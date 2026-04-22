from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from torch.utils.data import Dataset

from training.data.go_utils import build_label_space, propagate_terms


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = ROOT_DIR / "training" / "data"
FOLDS_DIR = TRAINING_DATA_DIR / "cv"
EMBEDDING_DIR = TRAINING_DATA_DIR / "embedding"
PROTEIN_FEATURES_DIR = TRAINING_DATA_DIR / "protein_features"
LABEL_SPACE_DIR = TRAINING_DATA_DIR / "label_space"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"

STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
HYDRO_MAP = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5,
    "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9,
    "M": 1.9, "F": 2.8, "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9,
    "Y": -1.3, "V": 4.2,
}
CHARGE_MAP = {"D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.1}
RESIDUE_GROUPS = {
    "polar": tuple("STNQCY"),
    "nonpolar": tuple("AILMFWVPG"),
    "aromatic": tuple("FWY"),
    "aliphatic": tuple("AILV"),
    "positive": tuple("KRH"),
    "negative": tuple("DE"),
    "charged": tuple("DEKRH"),
    "sulfur": tuple("CM"),
    "small": tuple("AGST"),
    "tiny": tuple("AGS"),
    "disorder": tuple("ARGQSPKE"),
    "order": tuple("NCILFWYV"),
}
PROTEIN_FEATURE_DIM = 63


@dataclass
class FoldData:
    fold_dir: Path
    aspect: str
    train_pids: List[str]
    val_pids: List[str]
    train_sequences: Dict[str, str]
    val_sequences: Dict[str, str]
    classes: np.ndarray
    train_matrix: sparse.csr_matrix
    val_matrix: sparse.csr_matrix
    train_labels_df: pd.DataFrame
    val_labels_df: pd.DataFrame


def load_fasta_sequences(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"FASTA not found: {path}")
    sequences: Dict[str, str] = {}
    current_id: str | None = None
    chunks: List[str] = []
    with Path(path).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    sequences[current_id] = "".join(chunks)
                token = line[1:].split()[0]
                current_id = token.split("|")[1] if "|" in token else token
                chunks = []
            else:
                chunks.append(line)
    if current_id is not None:
        sequences[current_id] = "".join(chunks)
    return sequences


def collect_unique_sequences_from_folds(folds: Sequence[int]) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    for fold in folds:
        fold_dir = FOLDS_DIR / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        for fasta_name in ("train.fasta", "val.fasta"):
            for pid, sequence in load_fasta_sequences(fold_dir / fasta_name).items():
                existing = sequences.get(pid)
                if existing is not None and existing != sequence:
                    raise ValueError(f"Conflicting sequences found for protein {pid}")
                sequences[pid] = sequence
    return sequences


def load_labels(path: Path, aspect: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Label TSV not found: {path}")
    frame = pd.read_csv(path, sep="\t")
    missing = {"EntryID", "term", "aspect"} - set(frame.columns)
    if missing:
        raise ValueError(f"Unexpected label schema in {path}; missing {sorted(missing)}")
    return frame[frame["aspect"] == aspect].copy()


def group_terms_by_pid(labels_df: pd.DataFrame) -> Dict[str, List[str]]:
    if labels_df.empty:
        return {}
    return labels_df.groupby("EntryID")["term"].apply(list).to_dict()


def load_or_build_global_label_space(
    *,
    folds: Sequence[int],
    aspect: str,
    parents: Dict[str, set[str]],
    min_count: int,
) -> np.ndarray:
    LABEL_SPACE_DIR.mkdir(parents=True, exist_ok=True)
    path = LABEL_SPACE_DIR / f"{aspect}_min{min_count}.npy"
    if path.exists():
        return np.load(path, allow_pickle=True)
    grouped_terms: List[List[str]] = []
    for fold in folds:
        fold_dir = FOLDS_DIR / f"fold_{fold}"
        grouped_terms.extend(group_terms_by_pid(load_labels(fold_dir / "train_labels.tsv", aspect)).values())
        grouped_terms.extend(group_terms_by_pid(load_labels(fold_dir / "val_labels.tsv", aspect)).values())
    classes = build_label_space(grouped_terms, parents, aspect=aspect, min_count=min_count)
    np.save(path, classes)
    return classes


def encode_labels(
    pids: Sequence[str],
    pid_to_terms: Dict[str, List[str]],
    classes: np.ndarray,
    parents: Dict[str, set[str]],
) -> sparse.csr_matrix:
    class_to_index = {term: index for index, term in enumerate(classes.tolist())}
    rows: List[int] = []
    cols: List[int] = []
    for row, pid in enumerate(pids):
        for term in propagate_terms(pid_to_terms.get(pid, ()), parents):
            col = class_to_index.get(term)
            if col is not None:
                rows.append(row)
                cols.append(col)
    return sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(pids), len(classes)),
        dtype=np.float32,
    )


def load_fold_data(*, fold: int, aspect: str, parents: Dict[str, set[str]], classes: np.ndarray) -> FoldData:
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
    train_sequences = load_fasta_sequences(fold_dir / "train.fasta")
    val_sequences = load_fasta_sequences(fold_dir / "val.fasta")
    train_labels_df = load_labels(fold_dir / "train_labels.tsv", aspect)
    val_labels_df = load_labels(fold_dir / "val_labels.tsv", aspect)
    train_pids = sorted(train_sequences)
    val_pids = sorted(val_sequences)
    return FoldData(
        fold_dir=fold_dir,
        aspect=aspect,
        train_pids=train_pids,
        val_pids=val_pids,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        classes=classes,
        train_matrix=encode_labels(train_pids, group_terms_by_pid(train_labels_df), classes, parents),
        val_matrix=encode_labels(val_pids, group_terms_by_pid(val_labels_df), classes, parents),
        train_labels_df=train_labels_df,
        val_labels_df=val_labels_df,
    )


def _distribution_stats(values: np.ndarray) -> List[float]:
    if values.size == 0:
        return [0.0] * 14
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    centered = values - mean
    skew = float((centered ** 3).mean() / (std ** 3 + 1e-8))
    kurt = float((centered ** 4).mean() / (std ** 4 + 1e-8))
    q25, median, q75 = np.quantile(values, [0.25, 0.5, 0.75])
    return [
        mean, std, float(values.min()), float(values.max()), float(median), float(q25),
        float(q75), skew, kurt, float((values > 0).mean()), float((values < 0).mean()),
        float(np.abs(values).mean()), float(values.sum()), float(values.sum() / max(values.size, 1)),
    ]


def build_sequence_protein_features(sequence: str) -> torch.Tensor:
    residues = [aa for aa in sequence.upper() if aa in STANDARD_AMINO_ACIDS]
    length = len(residues)
    counts = {aa: residues.count(aa) for aa in STANDARD_AMINO_ACIDS}
    features: List[float] = []
    features.extend(counts[aa] / max(length, 1) for aa in STANDARD_AMINO_ACIDS)
    features.extend([float(length), float(np.log1p(length)), float(np.sqrt(length))])
    charge_values = np.asarray([CHARGE_MAP.get(aa, 0.0) for aa in residues], dtype=np.float32)
    features.extend(_distribution_stats(charge_values))
    hydro_values = np.asarray([HYDRO_MAP.get(aa, 0.0) for aa in residues], dtype=np.float32)
    hydro_stats = _distribution_stats(hydro_values)
    if hydro_values.size:
        hydro_stats[9] = float((hydro_values > 1.6).mean())
        hydro_stats[10] = float((hydro_values < -1.6).mean())
        hydro_stats[12] = float(hydro_values.max() - hydro_values.min())
        hydro_stats[13] = float(np.quantile(hydro_values, 0.75) - np.quantile(hydro_values, 0.25))
    features.extend(hydro_stats)
    for group in RESIDUE_GROUPS.values():
        features.append(sum(counts[aa] for aa in group) / max(length, 1))
    if len(features) != PROTEIN_FEATURE_DIM:
        raise RuntimeError(f"Protein feature dimension mismatch: {len(features)} != {PROTEIN_FEATURE_DIM}")
    return torch.tensor(features, dtype=torch.float32)


def build_and_save_protein_features(sequences_by_pid: Dict[str, str], output_path: Path) -> Dict[str, torch.Tensor]:
    cache = {pid: build_sequence_protein_features(seq) for pid, seq in sorted(sequences_by_pid.items())}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def load_protein_features_cache(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


class MultiEmbeddingDataset(Dataset):
    def __init__(
        self,
        pids: Sequence[str],
        labels: sparse.csr_matrix | None,
        embedding_dir: Path,
        protein_features_cache: Dict[str, torch.Tensor],
        chain: str = "all",
    ) -> None:
        self.pids = list(pids)
        self.labels = labels
        self.embedding_dir = Path(embedding_dir)
        self.protein_features_cache = protein_features_cache
        self.chain = chain

    def __len__(self) -> int:
        return len(self.pids)

    def _load_embedding(self, pid: str, plm: str, layer: str) -> torch.Tensor:
        path = self.embedding_dir / plm / layer / f"{pid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing embedding for {pid}: {path}")
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor embedding at {path}")
        return tensor.float()

    def __getitem__(self, index: int):
        pid = self.pids[index]
        item = {"pid": pid, "protein_features": self.protein_features_cache[pid].float()}
        if self.chain == "esm2_last":
            item["token_embeddings"] = self._load_embedding(pid, "esm2", "last")
        elif self.chain == "esm2_l20":
            item["token_embeddings"] = self._load_embedding(pid, "esm2", "layer20")
        elif self.chain == "prott5":
            item["token_embeddings"] = self._load_embedding(pid, "t5", "last")
        elif self.chain == "all":
            item["esm2_layer20"] = self._load_embedding(pid, "esm2", "layer20")
            item["esm2_last"] = self._load_embedding(pid, "esm2", "last")
            item["t5_last"] = self._load_embedding(pid, "t5", "last")
        else:
            raise ValueError(f"Unsupported embedding chain: {self.chain}")
        if self.labels is not None:
            item["labels"] = torch.from_numpy(self.labels[index].toarray().ravel().astype(np.float32))
        return item


def _pad_token_embeddings(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(tensors)
    max_length = max(tensor.size(0) for tensor in tensors)
    dim = tensors[0].size(-1)
    padded = torch.zeros((batch_size, max_length, dim), dtype=torch.float32)
    mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    for index, tensor in enumerate(tensors):
        length = tensor.size(0)
        padded[index, :length] = tensor
        mask[index, :length] = 1
    return padded, mask


def collate_multi_embedding_batch(batch: Sequence[dict]):
    pids = [item["pid"] for item in batch]
    inputs = {}
    if "token_embeddings" in batch[0]:
        padded, mask = _pad_token_embeddings([item["token_embeddings"] for item in batch])
        inputs["token_embeddings"] = padded
        inputs["attention_mask"] = mask
    else:
        for key in ("esm2_layer20", "esm2_last", "t5_last"):
            padded, mask = _pad_token_embeddings([item[key] for item in batch])
            inputs[key] = padded
            inputs[f"{key}_mask"] = mask
    inputs["protein_features"] = torch.stack([item["protein_features"] for item in batch], dim=0)
    if "labels" not in batch[0]:
        return pids, inputs
    return pids, inputs, torch.stack([item["labels"] for item in batch], dim=0)
