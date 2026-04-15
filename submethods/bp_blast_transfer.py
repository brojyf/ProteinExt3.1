from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _require_blast() -> None:
    missing = [name for name in ("makeblastdb", "blastp") if shutil.which(name) is None]
    if missing:
        missing_list = ", ".join(missing)
        raise RuntimeError(
            f"BP BLAST enhancement requires local BLAST+ binaries, missing: {missing_list}"
        )


def _build_database(train_fasta: Path, db_dir: Path) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / "train_db"
    expected = db_prefix.with_suffix(".pin")
    if expected.exists():
        return db_prefix

    command = [
        "makeblastdb",
        "-in",
        str(train_fasta),
        "-dbtype",
        "prot",
        "-out",
        str(db_prefix),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    return db_prefix


def _run_blast(query_fasta: Path, db_prefix: Path, output_path: Path, max_hits: int, evalue: float, num_threads: int = 8) -> None:
    command = [
        "blastp",
        "-query",
        str(query_fasta),
        "-db",
        str(db_prefix),
        "-outfmt",
        "6 qseqid sseqid bitscore pident evalue",
        "-max_target_seqs",
        str(max_hits),
        "-num_threads",
        str(num_threads),
        "-evalue",
        str(evalue),
        "-out",
        str(output_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)


def _parse_blast_hits(path: Path) -> Dict[str, List[dict]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}

    columns = ["qseqid", "sseqid", "bitscore", "pident", "evalue"]
    frame = pd.read_csv(path, sep="\t", header=None, names=columns)

    hits: Dict[str, List[dict]] = {}
    for row in frame.itertuples(index=False):
        hits.setdefault(row.qseqid, []).append(
            {
                "subject": row.sseqid,
                "bitscore": float(row.bitscore),
                "pident": float(row.pident),
                "evalue": float(row.evalue),
            }
        )
    return hits


def _transfer_scores(
    query_pids: np.ndarray,
    hits_by_query: Dict[str, List[dict]],
    train_labels_df: pd.DataFrame,
    classes: np.ndarray,
    aspect: str | None = None,
) -> np.ndarray:
    df = train_labels_df if aspect is None else train_labels_df[train_labels_df["aspect"] == aspect]
    train_terms = df.groupby("EntryID")["term"].apply(list).to_dict()
    class_to_index = {term: idx for idx, term in enumerate(classes)}
    scores = np.zeros((len(query_pids), len(classes)), dtype=np.float32)

    for row_index, pid in enumerate(query_pids):
        hits = hits_by_query.get(pid, [])
        if not hits:
            continue

        total_weight = 0.0
        for hit in hits:
            terms = train_terms.get(hit["subject"])
            if not terms:
                continue
            weight = max(hit["bitscore"], 0.0) * max(hit["pident"], 0.0) / 100.0
            if weight <= 0:
                continue
            total_weight += weight
            for term in terms:
                class_index = class_to_index.get(term)
                if class_index is not None:
                    scores[row_index, class_index] += weight

        if total_weight > 0:
            scores[row_index] /= total_weight

    return scores


def maybe_run_bp_blast_enhancement(
    *,
    aspect: str,
    enabled: bool,
    fold_dir: Path,
    query_pids: np.ndarray,
    classes: np.ndarray,
    train_labels_df: pd.DataFrame,
) -> np.ndarray | None:
    if not enabled:
        return None

    _require_blast()

    blast_dir = fold_dir / "blast_cache"
    db_prefix = _build_database(fold_dir / "train.fasta", blast_dir)
    output_path = blast_dir / "val_vs_train.tsv"
    _run_blast(fold_dir / "val.fasta", db_prefix, output_path, max_hits=10, evalue=1e-3)
    hits = _parse_blast_hits(output_path)
    return _transfer_scores(query_pids, hits, train_labels_df, classes)
