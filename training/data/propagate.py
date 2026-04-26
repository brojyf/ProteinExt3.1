from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.data.data_utils import load_fasta_sequences
from training.data.go_utils import parse_go_obo, propagate_terms


DEFAULT_RAW_DIR = ROOT_DIR / "training" / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "training" / "data" / "propagated"
DEFAULT_FASTA = DEFAULT_RAW_DIR / "training.fasta"
DEFAULT_LABELS = DEFAULT_RAW_DIR / "training.tsv"
DEFAULT_OBO = ROOT_DIR / "training" / "data" / "go-basic.obo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propagate raw GO labels and write a propagated training dataset.")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_labels(path: Path) -> dict[tuple[str, str], set[str]]:
    labels: dict[tuple[str, str], set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"EntryID", "term", "aspect"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Labels TSV must contain columns: {sorted(required)}")
        for row in reader:
            pid = row["EntryID"].strip()
            term = row["term"].strip()
            aspect = row["aspect"].strip()
            if not pid or not term or not aspect:
                continue
            labels.setdefault((pid, aspect), set()).add(term)
    return labels


def validate_inputs(sequences: dict[str, str], labels: dict[tuple[str, str], set[str]]) -> None:
    missing_pids = sorted({pid for pid, _ in labels} - set(sequences))
    if missing_pids:
        example = ", ".join(missing_pids[:5])
        raise ValueError(f"Labels contain proteins not present in FASTA, for example: {example}")


def write_labels(path: Path, labels: dict[tuple[str, str], set[str]], parents: dict[str, set[str]]) -> int:
    row_count = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["EntryID", "term", "aspect"], delimiter="\t")
        writer.writeheader()
        for pid, aspect in sorted(labels):
            for term in sorted(propagate_terms(labels[(pid, aspect)], parents)):
                writer.writerow({"EntryID": pid, "term": term, "aspect": aspect})
                row_count += 1
    return row_count


def propagate_dataset(args: argparse.Namespace) -> None:
    for path in (args.fasta, args.labels, args.obo):
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")

    sequences = load_fasta_sequences(args.fasta)
    labels = load_labels(args.labels)
    validate_inputs(sequences, labels)
    parents = parse_go_obo(args.obo)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fasta_out = args.out_dir / "training.fasta"
    labels_out = args.out_dir / "training.tsv"
    shutil.copyfile(args.fasta, fasta_out)
    row_count = write_labels(labels_out, labels, parents)

    print(f"saved fasta: {fasta_out}")
    print(f"saved labels: {labels_out}")
    print(f"proteins: {len(sequences)}")
    print(f"label rows: {row_count}")


def main() -> None:
    propagate_dataset(parse_args())


if __name__ == "__main__":
    main()
