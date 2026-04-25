from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd

from training.data.data_utils import load_fasta_sequences
from training.data.go_utils import parse_go_obo, propagate_terms

DEFAULT_RAW_DIR = ROOT_DIR / "training" / "data" / "raw"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "training" / "data" / "cv"
DEFAULT_FASTA = DEFAULT_RAW_DIR / "training.fasta"
DEFAULT_LABELS = DEFAULT_RAW_DIR / "training.tsv"
DEFAULT_OBO = ROOT_DIR / "data" / "go-basic.obo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fold_* CV files from raw FASTA and labels TSV")
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cd-hit-bin", default="cd-hit")
    parser.add_argument("--cd-hit-identity", type=float, default=0.5)
    parser.add_argument("--cd-hit-word-size", type=int, default=2)
    parser.add_argument("--mem", type=int, default=16000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_labels(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")
    labels = pd.read_csv(path, sep="\t")
    missing = {"EntryID", "term", "aspect"} - set(labels.columns)
    if missing:
        raise ValueError(f"Labels TSV must contain EntryID, term, aspect columns; missing {sorted(missing)}")
    labels = labels[["EntryID", "term", "aspect"]].copy()
    for column in ("EntryID", "term", "aspect"):
        labels[column] = labels[column].astype(str)
    return labels


def extract_cluster_pid(line: str) -> str:
    marker = line.find(">")
    if marker == -1:
        raise ValueError(f"Unexpected CD-HIT cluster line: {line.rstrip()}")
    token = line[marker + 1 :].split("...", maxsplit=1)[0].split()[0].rstrip(",")
    return token.split("|")[1] if "|" in token else token


def load_cd_hit_clusters(path: Path) -> List[List[str]]:
    if not path.exists():
        raise FileNotFoundError(f"CD-HIT cluster file not found: {path}")
    clusters: List[List[str]] = []
    current_cluster: List[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">Cluster"):
                if current_cluster:
                    clusters.append(current_cluster)
                current_cluster = []
            else:
                current_cluster.append(extract_cluster_pid(line))
    if current_cluster:
        clusters.append(current_cluster)
    if not clusters:
        raise RuntimeError(f"No clusters found in {path}")
    return clusters


def run_cd_hit(args: argparse.Namespace, work_dir: Path) -> List[List[str]]:
    cd_hit_bin = shutil.which(args.cd_hit_bin)
    if cd_hit_bin is None:
        raise FileNotFoundError(
            f"CD-HIT executable not found: {args.cd_hit_bin}. "
            "Install cd-hit or pass --cd-hit-bin with the executable path."
        )

    output_prefix = work_dir / "clustered"
    command = [
        cd_hit_bin,
        "-i",
        str(args.fasta),
        "-o",
        str(output_prefix),
        "-c",
        str(args.cd_hit_identity),
        "-n",
        str(args.cd_hit_word_size),
        "-d",
        "0",
        "-M",
        str(args.mem),
        "-T",
        str(args.threads),
    ]
    print("Running CD-HIT: " + " ".join(command))
    subprocess.run(command, check=True)
    return load_cd_hit_clusters(output_prefix.with_suffix(output_prefix.suffix + ".clstr"))


def propagate_labels(labels: pd.DataFrame, obo_path: Path) -> pd.DataFrame:
    if not obo_path.exists():
        raise FileNotFoundError(f"GO OBO file not found: {obo_path}")
    parents = parse_go_obo(obo_path)
    rows: List[Dict[str, str]] = []
    for (pid, aspect), group in labels.groupby(["EntryID", "aspect"], sort=False):
        for term in sorted(propagate_terms(group["term"], parents)):
            rows.append({"EntryID": pid, "term": term, "aspect": aspect})
    return pd.DataFrame(rows, columns=["EntryID", "term", "aspect"])


def write_fasta(path: Path, pids: Sequence[str], sequences: Dict[str, str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for pid in pids:
            handle.write(f">{pid}\n")
            sequence = sequences[pid]
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def split_clusters(clusters: Sequence[Sequence[str]], pids: Sequence[str], folds: int, seed: int) -> List[List[str]]:
    valid_pids = set(pids)
    seen_pids: set[str] = set()
    cluster_members: List[List[str]] = []
    repeated_pids: List[str] = []

    for cluster in clusters:
        members = sorted({pid for pid in cluster if pid in valid_pids})
        if not members:
            continue
        repeated_pids.extend(sorted(seen_pids.intersection(members)))
        seen_pids.update(members)
        cluster_members.append(members)

    if repeated_pids:
        example = ", ".join(sorted(set(repeated_pids))[:5])
        raise ValueError(f"CD-HIT clusters contain repeated proteins, for example: {example}")

    missing_pids = sorted(valid_pids - seen_pids)
    if missing_pids:
        print(
            f"Warning: {len(missing_pids)} FASTA proteins were not in CD-HIT clusters; "
            "assigning them as singleton clusters.",
            file=sys.stderr,
        )
        cluster_members.extend([pid] for pid in missing_pids)

    shuffled = list(cluster_members)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=len, reverse=True)

    fold_pids = [[] for _ in range(folds)]
    fold_sizes = [0] * folds
    for members in shuffled:
        fold = min(range(folds), key=lambda index: (fold_sizes[index], index))
        fold_pids[fold].extend(members)
        fold_sizes[fold] += len(members)
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
    labels = propagate_labels(load_labels(args.labels), args.obo)
    validate_inputs(sequences, labels, args.folds)
    prepare_output_dir(args.out, args.overwrite)

    all_pids = sorted(sequences)
    with tempfile.TemporaryDirectory(prefix="make_cv_cdhit_") as tmp_dir:
        clusters = run_cd_hit(args, Path(tmp_dir))
        val_by_fold = split_clusters(clusters, all_pids, args.folds, args.seed)
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
