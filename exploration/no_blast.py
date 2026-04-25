from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, TextIO, Tuple

import pandas as pd

from training.data.go_utils import ASPECT_ROOTS


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_PREDICTIONS_DIR = ROOT_DIR / "predictions"
DEFAULT_WEIGHTS_PATH = ROOT_DIR / "models" / "fusion_weights.csv"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build no-BLAST fusion predictions from component TSV files")
    parser.add_argument("--esm2", type=Path, default=DEFAULT_PREDICTIONS_DIR / "ProteinExt3.1-ESM2.tsv")
    parser.add_argument("--esm2-l20", dest="esm2_l20", type=Path, default=DEFAULT_PREDICTIONS_DIR / "ProteinExt3.1-ESM2-L20.tsv")
    parser.add_argument("--prott5", type=Path, default=DEFAULT_PREDICTIONS_DIR / "ProteinExt3.1-ProtT5.tsv")
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS_PATH)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_PREDICTIONS_DIR / "ProteinExt3.1-noBLAST.tsv")
    parser.add_argument("--apply-th", action="store_true", help="Apply per-aspect fusion threshold from weights.csv")
    parser.add_argument("--renormalize", action="store_true", help="Renormalize neural weights after dropping BLAST")
    return parser.parse_args()


def load_weights(path: Path) -> Dict[str, Dict[str, float]]:
    frame = pd.read_csv(path)
    required = {"aspect", "thr", "w_last", "w_l20", "w_t5"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    weights: Dict[str, Dict[str, float]] = {}
    for row in frame.to_dict(orient="records"):
        aspect = str(row["aspect"])
        weights[aspect] = {
            "esm2": float(row["w_last"]),
            "esm2_l20": float(row["w_l20"]),
            "prott5": float(row["w_t5"]),
            "threshold": float(row["thr"]),
        }
    return weights


def infer_term_aspects(terms: Iterable[str], obo_path: Path) -> Dict[str, str]:
    namespace_to_aspect = {
        "biological_process": "P",
        "molecular_function": "F",
        "cellular_component": "C",
    }
    requested_terms = set(terms)
    resolved: Dict[str, str] = {}
    current_id: str | None = None
    in_term = False
    with obo_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("["):
                in_term = line == "[Term]"
                current_id = None
                continue
            if not in_term:
                continue
            if line.startswith("id: "):
                current_id = line[4:]
            elif current_id in requested_terms and line.startswith("namespace: "):
                namespace = line[11:]
                aspect = namespace_to_aspect.get(namespace)
                if aspect is None:
                    raise ValueError(f"Unsupported namespace for GO term {current_id}: {namespace}")
                resolved[current_id] = aspect
                if len(resolved) == len(requested_terms):
                    break
    missing = requested_terms - set(resolved)
    root_to_aspect = {root: aspect for aspect, root in ASPECT_ROOTS.items()}
    for term in list(missing):
        aspect = root_to_aspect.get(term)
        if aspect is not None:
            resolved[term] = aspect
    missing = requested_terms - set(resolved)
    if missing:
        sample = ", ".join(sorted(list(missing))[:5])
        raise ValueError(f"Failed to resolve aspect for {len(missing)} GO terms, e.g. {sample}")
    return resolved


class PredictionBlockReader:
    def __init__(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Prediction file not found: {path}")
        self.path = path
        self.handle = path.open(encoding="utf-8", newline="")
        self.reader = csv.reader(self.handle, delimiter="\t")
        self.pending: List[str] | None = None
        self.line_no = 0

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> "PredictionBlockReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def blocks(self) -> Iterator[Tuple[str, Dict[str, float]]]:
        while True:
            row = self.pending
            if row is None:
                try:
                    row = next(self.reader)
                    self.line_no += 1
                except StopIteration:
                    return
            self.pending = None
            if len(row) != 3:
                raise ValueError(f"Expected 3 TSV columns in {self.path}:{self.line_no}, got {len(row)}")
            pid, term, score = row
            term_scores = {term: float(score)}
            while True:
                try:
                    next_row = next(self.reader)
                    self.line_no += 1
                except StopIteration:
                    yield pid, term_scores
                    return
                if len(next_row) != 3:
                    raise ValueError(f"Expected 3 TSV columns in {self.path}:{self.line_no}, got {len(next_row)}")
                next_pid, next_term, next_score = next_row
                if next_pid != pid:
                    self.pending = next_row
                    yield pid, term_scores
                    break
                term_scores[next_term] = float(next_score)


def collect_terms(paths: Iterable[Path]) -> set[str]:
    terms: set[str] = set()
    for path in paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            for line_no, row in enumerate(reader, start=1):
                if len(row) != 3:
                    raise ValueError(f"Expected 3 TSV columns in {path}:{line_no}, got {len(row)}")
                terms.add(row[1])
    return terms


def adjusted_weights(aspect_weights: Dict[str, float], renormalize: bool) -> Tuple[float, float, float]:
    esm2_weight = aspect_weights["esm2"]
    esm2_l20_weight = aspect_weights["esm2_l20"]
    prott5_weight = aspect_weights["prott5"]
    if renormalize:
        total = esm2_weight + esm2_l20_weight + prott5_weight
        if total <= 0:
            raise ValueError("Neural weight sum is non-positive")
        return esm2_weight / total, esm2_l20_weight / total, prott5_weight / total
    return esm2_weight, esm2_l20_weight, prott5_weight


def write_no_blast_predictions(
    output_handle: TextIO,
    component_paths: Dict[str, Path],
    weights_by_aspect: Dict[str, Dict[str, float]],
    term_aspect: Dict[str, str],
    apply_threshold: bool,
    renormalize: bool,
) -> int:
    writer = csv.writer(output_handle, delimiter="\t")
    row_count = 0
    with (
        PredictionBlockReader(component_paths["esm2"]) as esm2_reader,
        PredictionBlockReader(component_paths["esm2_l20"]) as esm2_l20_reader,
        PredictionBlockReader(component_paths["prott5"]) as prott5_reader,
    ):
        for esm2_block, esm2_l20_block, prott5_block in zip(
            esm2_reader.blocks(),
            esm2_l20_reader.blocks(),
            prott5_reader.blocks(),
            strict=True,
        ):
            pid = esm2_block[0]
            if esm2_l20_block[0] != pid or prott5_block[0] != pid:
                raise ValueError(
                    "Component prediction files are not aligned by protein ID: "
                    f"{pid}, {esm2_l20_block[0]}, {prott5_block[0]}"
                )
            merged_terms = (
                set(esm2_block[1])
                | set(esm2_l20_block[1])
                | set(prott5_block[1])
            )
            fused_rows: List[Tuple[str, float]] = []
            for term in merged_terms:
                aspect = term_aspect[term]
                aspect_weights = weights_by_aspect[aspect]
                esm2_weight, esm2_l20_weight, prott5_weight = adjusted_weights(aspect_weights, renormalize)
                score = (
                    esm2_weight * esm2_block[1].get(term, 0.0)
                    + esm2_l20_weight * esm2_l20_block[1].get(term, 0.0)
                    + prott5_weight * prott5_block[1].get(term, 0.0)
                )
                fused_rows.append((term, score))
            fused_rows.sort(key=lambda item: item[1], reverse=True)
            for term, score in fused_rows:
                if apply_threshold and score < weights_by_aspect[term_aspect[term]]["threshold"]:
                    break
                writer.writerow([pid, term, f"{score:.6f}"])
                row_count += 1
    return row_count


def main() -> None:
    args = parse_args()
    weights_by_aspect = load_weights(args.weights)
    component_paths = {
        "esm2": args.esm2,
        "esm2_l20": args.esm2_l20,
        "prott5": args.prott5,
    }
    all_terms = collect_terms(component_paths.values())
    term_aspect = infer_term_aspects(all_terms, args.obo)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        row_count = write_no_blast_predictions(
            handle,
            component_paths,
            weights_by_aspect,
            term_aspect,
            args.apply_th,
            args.renormalize,
        )
    print(f"Saved no-BLAST predictions to {args.out}")
    print(f"Rows written: {row_count}")
    print(f"Weight mode: {'renormalized' if args.renormalize else 'raw no-blast'}")


if __name__ == "__main__":
    main()
