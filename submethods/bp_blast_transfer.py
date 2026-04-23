from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

BLAST_COLUMNS = ["qseqid", "sseqid", "pident", "length", "qlen", "slen", "evalue", "bitscore"]


def _require_blast() -> None:
    missing = [name for name in ("makeblastdb", "blastp") if shutil.which(name) is None]
    if missing:
        raise RuntimeError(f"BLAST+ binaries are required, missing: {', '.join(missing)}")


def _build_database(train_fasta: Path, db_dir: Path) -> Path:
    db_dir.mkdir(parents=True, exist_ok=True)
    db_prefix = db_dir / "train_db"
    expected_files = [db_prefix.with_suffix(f".{suffix}") for suffix in ("pin", "phr", "psq")]
    if all(path.exists() for path in expected_files):
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
    num_threads = max(1, int(num_threads))
    subprocess.run(
        [
            "blastp", "-query", str(query_fasta), "-db", str(db_prefix),
            "-outfmt", "6 qseqid sseqid pident length qlen slen evalue bitscore",
            "-max_target_seqs", str(max_hits), "-num_threads", str(num_threads),
            "-evalue", str(evalue), "-out", str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def _normalize_protein_id(value: object) -> str:
    token = str(value).split()[0]
    parts = token.split("|")
    if len(parts) >= 3:
        return parts[1]
    return token


def _parse_blast_hits(path: Path) -> Dict[str, List[dict]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    frame = pd.read_csv(path, sep="\t", header=None, comment="#")
    if frame.empty:
        return {}
    if frame.shape[1] == 8:
        frame.columns = BLAST_COLUMNS
    elif frame.shape[1] == 5:
        frame.columns = ["qseqid", "sseqid", "bitscore", "pident", "evalue"]
        frame["length"] = 1.0
        frame["qlen"] = 1.0
        frame["slen"] = 1.0
        frame = frame[BLAST_COLUMNS]
    elif frame.shape[1] == 6:
        frame.columns = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore"]
        frame["qlen"] = frame["length"]
        frame["slen"] = frame["length"]
        frame = frame[BLAST_COLUMNS]
    else:
        raise ValueError(f"Unsupported BLAST TSV format with {frame.shape[1]} columns. Expected 5, 6, or 8 columns.")
    frame["qseqid"] = frame["qseqid"].map(_normalize_protein_id)
    frame["sseqid"] = frame["sseqid"].map(_normalize_protein_id)
    for col in ["pident", "length", "qlen", "slen", "evalue", "bitscore"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    frame = frame.sort_values(["qseqid", "evalue", "bitscore"], ascending=[True, True, False])
    hits: Dict[str, List[dict]] = {}
    for row in frame.itertuples(index=False):
        hits.setdefault(row.qseqid, []).append(
            {
                "subject": row.sseqid,
                "pident": float(row.pident),
                "length": float(row.length),
                "qlen": float(row.qlen),
                "slen": float(row.slen),
                "evalue": float(row.evalue),
                "bitscore": float(row.bitscore),
            }
        )
    return hits


def _transfer_scores(
    query_pids: np.ndarray,
    hits_by_query: Dict[str, List[dict]],
    train_labels_df: pd.DataFrame,
    classes: np.ndarray,
) -> np.ndarray:
    train_labels = train_labels_df.copy()
    train_labels["EntryID"] = train_labels["EntryID"].map(_normalize_protein_id)
    train_terms = train_labels.groupby("EntryID")["term"].apply(list).to_dict()
    class_to_index = {term: index for index, term in enumerate(classes.tolist())}
    scores = np.zeros((len(query_pids), len(classes)), dtype=np.float32)
    for row_index, raw_pid in enumerate(query_pids.tolist()):
        pid = _normalize_protein_id(raw_pid)
        hits = hits_by_query.get(pid, [])
        if not hits:
            continue
        term_weights: Dict[int, float] = {}
        total_weight = 0.0
        for hit in hits:
            coverage = min(float(hit.get("length", 0.0)) / max(float(hit.get("qlen", 1.0)), 1.0), 1.0)
            identity = max(min(float(hit.get("pident", 0.0)) / 100.0, 1.0), 0.0)
            weight = max(float(hit.get("bitscore", 0.0)), 0.0) * coverage * identity
            if weight <= 0:
                continue
            total_weight += weight
            for term in train_terms.get(hit["subject"], []):
                col = class_to_index.get(term)
                if col is not None:
                    term_weights[col] = term_weights.get(col, 0.0) + weight
        if total_weight <= 0:
            continue
        for col, weight in term_weights.items():
            scores[row_index, col] = min(weight / total_weight, 1.0)
    return np.clip(scores, 0.0, 1.0)
