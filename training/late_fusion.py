from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"
DEFAULT_MODEL_DIR = ROOT_DIR / "models"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "latefusion.csv"
METHOD_COLUMNS = {
    "esm2_last": "last",
    "esm2_l20": "l20",
    "prott5": "t5",
    "blast": "blast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build late-fusion weights from best OOF folds")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--methods", nargs="+", default=["esm2_last", "esm2_l20", "prott5", "blast"])
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def artifact_prefix(oof_dir: Path, method: str, aspect: str, fold: int) -> Path:
    aliases = {"esm2": "esm2_last", "t5": "prott5"}
    method = aliases.get(method, method)
    return oof_dir / f"{method}_{aspect}_fold_{fold}"


def load_metrics(oof_dir: Path, method: str, aspect: str, fold: int) -> dict:
    prefix = artifact_prefix(oof_dir, method, aspect, fold)
    path = prefix.with_name(prefix.name + "_metrics.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF metrics: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def model_path(model_dir: Path, method: str, aspect: str, fold: int) -> str:
    if method == "blast":
        return ""
    path = model_dir / f"{method}_{aspect}_fold_{fold}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {path}")
    return str(path.relative_to(ROOT_DIR))


def best_fold_record(oof_dir: Path, model_dir: Path, method: str, aspect: str, folds: List[int]) -> dict:
    best = None
    for fold in folds:
        metrics = load_metrics(oof_dir, method, aspect, fold)
        candidate = {
            "Aspect": aspect,
            "Method": method,
            "Best_Fold": fold,
            "OOF_FMAX": float(metrics["fmax"]),
            "OOF_THRESHOLD": float(metrics["fmax_threshold"]),
            "OOF_MICRO_F1": float(metrics["micro_f1"]),
            "MODEL_PATH": model_path(model_dir, method, aspect, fold),
        }
        if best is None or candidate["OOF_FMAX"] > best["OOF_FMAX"]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No OOF metrics found for method={method} aspect={aspect}")
    return best


def main() -> None:
    args = parse_args()
    rows = []
    summary: Dict[str, dict] = {}
    aliases = {"esm2": "esm2_last", "t5": "prott5"}
    methods = [aliases.get(method, method) for method in args.methods]
    required = {"esm2_last", "esm2_l20", "prott5", "blast"}
    if set(methods) != required:
        raise ValueError(f"Two-stage fusion requires methods {sorted(required)}, got {methods}")
    for aspect in args.aspect:
        records = [best_fold_record(args.oof_dir, args.model_dir, method, aspect, args.fold) for method in methods]
        fmax_values = [record["OOF_FMAX"] for record in records]
        fmax_sum = sum(fmax_values)
        if fmax_sum > 0:
            weights = [value / fmax_sum for value in fmax_values]
        else:
            weights = [1.0 / len(records)] * len(records)
        aspect_threshold = sum(weight * record["OOF_THRESHOLD"] for weight, record in zip(weights, records))
        summary[aspect] = {"threshold": aspect_threshold, "methods": {}}
        row = {"aspect": aspect, "thr": round(aspect_threshold, 2)}
        fold_columns = {}
        for weight, record in zip(weights, records):
            record["WEIGHT"] = weight
            record["THRESHOLD"] = aspect_threshold
            column = METHOD_COLUMNS[record["Method"]]
            row[f"w_{column}"] = round(weight, 2)
            fold_columns[f"fold_{column}"] = record["Best_Fold"]
            summary[aspect]["methods"][record["Method"]] = record
        row.update(fold_columns)
        rows.append(row)
        weight_text = " ".join(f"{record['Method']}={record['WEIGHT']:.4f}" for record in records)
        print(f"aspect={aspect} threshold={aspect_threshold:.4f} {weight_text}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    with args.output.with_name(args.output.stem + "_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Saved fusion weights to {args.output}")


if __name__ == "__main__":
    main()
