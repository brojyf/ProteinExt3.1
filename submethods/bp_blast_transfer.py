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
        raise RuntimeError(f"BLAST+ binaries are required, missing: {', '.join(missing)}")


def _build_database(train_fasta: Path, db_dir: Path) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / "train_db"
    if db_prefix.with_suffix(".pin").exists():
        return db_prefix
    subprocess.run(
        ["makeblastdb", "-in", str(train_fasta), "-dbtype", "prot", "-out", str(db_prefix)],
        check=True,
        capture_output=True,
        text=True,
    )
    return db_prefix


def _run_blast(
    query_fasta: Path,
    db_prefix: Path,
    output_path: Path,
    *,
    max_hits: int,
    evalue: float,
    num_threads: int = 8,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "blastp", "-query", str(query_fasta), "-db", str(db_prefix),
            "-outfmt", "6 qseqid sseqid bitscore pident evalue",
            "-max_target_seqs", str(max_hits), "-num_threads", str(num_threads),
            "-evalue", str(evalue), "-out", str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _parse_blast_hits(path: Path) -> Dict[str, List[dict]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path, sep="\t", header=None, names=["qseqid", "sseqid", "bitscore", "pident", "evalue"])
    hits: Dict[str, List[dict]] = {}
    for row in frame.itertuples(index=False):
        hits.setdefault(row.qseqid, []).append({"subject": row.sseqid, "bitscore": float(row.bitscore)})
    return hits


def _transfer_scores(
    query_pids: np.ndarray,
    hits_by_query: Dict[str, List[dict]],
    train_labels_df: pd.DataFrame,
    classes: np.ndarray,
    *,
    tau: float = 50.0,
) -> np.ndarray:
    train_terms = train_labels_df.groupby("EntryID")["term"].apply(list).to_dict()
    class_to_index = {term: index for index, term in enumerate(classes.tolist())}
    scores = np.zeros((len(query_pids), len(classes)), dtype=np.float32)
    for row_index, pid in enumerate(query_pids.tolist()):
        hits = hits_by_query.get(pid, [])
        if not hits:
            continue
        bitscores = np.asarray([max(hit["bitscore"], 0.0) for hit in hits], dtype=np.float64)
        weights = np.exp((bitscores - bitscores.max()) / tau)
        weights /= weights.sum()
        for hit, weight in zip(hits, weights):
            for term in train_terms.get(hit["subject"], []):
                col = class_to_index.get(term)
                if col is not None:
                    scores[row_index, col] += float(weight)
    return np.clip(scores, 0.0, 1.0)
