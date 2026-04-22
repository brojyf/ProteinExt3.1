from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from training.data.data_utils import load_fasta_sequences

DEFAULT_RAW_DIR = ROOT_DIR / "training" / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "training" / "data" / "cv"
DEFAULT_FASTA = DEFAULT_RAW_DIR / "train.fasta"
DEFAULT_LABELS = DEFAULT_RAW_DIR / "labels.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fold_* CV files from raw FASTA and labels TSV")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = pd.read_csv(path, sep="\t")
    missing = {"EntryID", "term", "aspect"} - set(labels.columns)
    if missing:
        raise ValueError(f"Labels TSV must contain EntryID, term, aspect columns; missing {sorted(missing)}")
    return labels[["EntryID", "term", "aspect"]].copy()


def write_fasta(path: Path, pids: Sequence[str], sequences: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pid in pids:
            handle.write(f">{pid}\n")
            sequence = sequences[pid]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def split_pids(pids: Sequence[str], folds: int, seed: int) -> List[List[str]]:
    shuffled = list(pids)
    random.Random(seed).shuffle(shuffled)
    fold_pids = [[] for _ in range(folds)]
    for index, pid in enumerate(shuffled):
        fold_pids[index % folds].append(pid)
    return [sorted(items) for items in fold_pids]


def validate_inputs(sequences: Dict[str, str], labels: pd.DataFrame, folds: int) -> None:
    if folds < 2:
        raise ValueError(f"--folds must be at least 2, got {folds}")
    if not sequences:
        raise RuntimeError("No sequences found in FASTA")
    if len(sequences) < folds:
        raise ValueError(f"Cannot build {folds} folds from only {len(sequences)} sequences")
    label_pids = set(labels["EntryID"].astype(str))
    missing_sequences = sorted(label_pids - set(sequences))
    if missing_sequences:
        example = ", ".join(missing_sequences[:5])
        raise ValueError(f"Labels contain proteins not present in FASTA, for example: {example}")


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output directory already exists: {path}. Use --overwrite to replace it.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def make_cv(args: argparse.Namespace) -> None:
    sequences = load_fasta_sequences(args.fasta)
    labels = load_labels(args.labels)
    validate_inputs(sequences, labels, args.folds)
    prepare_output_dir(args.out, args.overwrite)

    all_pids = sorted(sequences)
    val_by_fold = split_pids(all_pids, args.folds, args.seed)
    all_pid_set = set(all_pids)

    for fold, val_pids in enumerate(val_by_fold):
        fold_dir = args.out / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        val_set = set(val_pids)
        train_pids = sorted(all_pid_set - val_set)

        write_fasta(fold_dir / "train.fasta", train_pids, sequences)
        write_fasta(fold_dir / "val.fasta", val_pids, sequences)

        labels[labels["EntryID"].isin(train_pids)].to_csv(fold_dir / "train_labels.tsv", sep="\t", index=False)
        labels[labels["EntryID"].isin(val_pids)].to_csv(fold_dir / "val_labels.tsv", sep="\t", index=False)

        print(
            f"fold_{fold}: train={len(train_pids)} val={len(val_pids)} "
            f"train_labels={labels['EntryID'].isin(train_pids).sum()} "
            f"val_labels={labels['EntryID'].isin(val_pids).sum()}"
        )

    print(f"Saved CV folds to {args.out}")


def main() -> None:
    make_cv(parse_args())


if __name__ == "__main__":
    main()
