from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset


ROOT_DIR = Path(__file__).resolve().parents[2]
TRAINING_DATA_DIR = ROOT_DIR / "training" / "data"
FOLDS_DIR = TRAINING_DATA_DIR / "cv"
EMBEDDING_DIR = TRAINING_DATA_DIR / "embedding"


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
    sequences: Dict[str, str] = {}
    for record in SeqIO.parse(str(path), "fasta"):
        pid = record.id.split("|")[1] if "|" in record.id else record.id.split()[0]
        sequences[pid] = str(record.seq)
    return sequences


def collect_unique_sequences_from_folds(folds: Sequence[int]) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    for fold in folds:
        fold_dir = FOLDS_DIR / f"fold_{fold}"
        if not fold_dir.exists():
            raise FileNotFoundError(f"Fold directory not found: {fold_dir}")
        for fasta_name in ("train.fasta", "val.fasta"):
            fold_sequences = load_fasta_sequences(fold_dir / fasta_name)
            for pid, sequence in fold_sequences.items():
                existing = sequences.get(pid)
                if existing is not None and existing != sequence:
                    raise ValueError(f"Conflicting sequences found for protein {pid}")
                sequences[pid] = sequence
    return sequences


def load_labels(path: Path, aspect: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if {"EntryID", "term", "aspect"} - set(df.columns):
        raise ValueError(f"Unexpected label schema in {path}")
    return df[df["aspect"] == aspect].copy()


def group_terms_by_pid(labels_df: pd.DataFrame) -> Dict[str, List[str]]:
    return labels_df.groupby("EntryID")["term"].apply(list).to_dict()


def encode_labels(
    pids: Sequence[str],
    pid_to_terms: Dict[str, List[str]],
    classes: np.ndarray | None = None,
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    label_lists = [pid_to_terms.get(pid, []) for pid in pids]
    if classes is None:
        mlb = MultiLabelBinarizer(sparse_output=True)
        matrix = mlb.fit_transform(label_lists).tocsr()
        return matrix, mlb.classes_

    mlb = MultiLabelBinarizer(classes=classes, sparse_output=True)
    matrix = mlb.fit_transform(label_lists).tocsr()
    return matrix, mlb.classes_


def load_fold_data(fold: int, aspect: str) -> FoldData:
    fold_dir = FOLDS_DIR / f"fold_{fold}"
    if not fold_dir.exists():
        raise FileNotFoundError(f"Fold directory not found: {fold_dir}")

    train_sequences = load_fasta_sequences(fold_dir / "train.fasta")
    val_sequences = load_fasta_sequences(fold_dir / "val.fasta")
    train_labels_df = load_labels(fold_dir / "train_labels.tsv", aspect)
    val_labels_df = load_labels(fold_dir / "val_labels.tsv", aspect)

    train_pids = sorted(train_sequences)
    val_pids = sorted(val_sequences)

    train_terms = group_terms_by_pid(train_labels_df)
    val_terms = group_terms_by_pid(val_labels_df)

    train_matrix, classes = encode_labels(train_pids, train_terms)
    val_matrix, _ = encode_labels(val_pids, val_terms, classes=classes)

    return FoldData(
        fold_dir=fold_dir,
        aspect=aspect,
        train_pids=train_pids,
        val_pids=val_pids,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        classes=classes,
        train_matrix=train_matrix,
        val_matrix=val_matrix,
        train_labels_df=train_labels_df,
        val_labels_df=val_labels_df,
    )


class ProteinSequenceDataset(Dataset):
    def __init__(
        self,
        pids: Sequence[str],
        sequences: Dict[str, str],
        labels: sparse.csr_matrix,
    ) -> None:
        self.pids = list(pids)
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, index: int) -> Tuple[str, str, torch.Tensor]:
        pid = self.pids[index]
        label = self.labels[index].toarray().ravel().astype(np.float32)
        return pid, self.sequences[pid], torch.from_numpy(label)


def collate_batch(batch: Sequence[Tuple[str, str, torch.Tensor]]):
    pids, sequences, labels = zip(*batch)
    return list(pids), list(sequences), torch.stack(list(labels), dim=0)


HYDRO_MAP = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}


def build_token_hydrophobicity_features(sequence: str, token_length: int, window_size: int) -> torch.Tensor:
    features = np.zeros((token_length, 2), dtype=np.float32)
    values = np.asarray([HYDRO_MAP.get(residue, 0.0) for residue in sequence], dtype=np.float32)
    if values.size == 0:
        return torch.from_numpy(features)

    half_window = max(window_size // 2, 1)
    local = np.zeros((values.shape[0], 2), dtype=np.float32)
    for residue_index in range(values.shape[0]):
        start = max(0, residue_index - half_window)
        end = min(values.shape[0], residue_index + half_window + 1)
        chunk = values[start:end]
        local[residue_index, 0] = float(chunk.mean())
        local[residue_index, 1] = float(chunk.var())

    usable_length = min(local.shape[0], max(token_length - 2, 0))
    if usable_length > 0:
        features[1 : 1 + usable_length] = local[:usable_length]
    return torch.from_numpy(features)


class ProteinTokenEmbeddingDataset(Dataset):
    def __init__(
        self,
        pids: Sequence[str],
        sequences: Dict[str, str],
        labels: sparse.csr_matrix,
        embedding_dir: Path,
        plm: str,
        layer: str,
        hydro_window_size: int | None = None,
    ) -> None:
        self.pids = list(pids)
        self.sequences = sequences
        self.labels = labels
        self.embedding_dir = Path(embedding_dir)
        self.plm = plm
        self.layer = layer
        self.hydro_window_size = hydro_window_size

    def __len__(self) -> int:
        return len(self.pids)

    def _embedding_path(self, pid: str) -> Path:
        return self.embedding_dir / self.plm / self.layer / f"{pid}.pt"

    def __getitem__(self, index: int):
        pid = self.pids[index]
        path = self._embedding_path(pid)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing embedding for protein {pid}: {path}. "
                "Run `python data/embedding.py --plm ...` first."
            )

        token_embeddings = torch.load(path, map_location="cpu")
        if not isinstance(token_embeddings, torch.Tensor):
            raise TypeError(f"Expected tensor embedding at {path}")
        token_embeddings = token_embeddings.float()

        label = torch.from_numpy(self.labels[index].toarray().ravel().astype(np.float32))
        hydro_features = None
        if self.hydro_window_size is not None:
            hydro_features = build_token_hydrophobicity_features(
                self.sequences[pid],
                token_length=token_embeddings.size(0),
                window_size=self.hydro_window_size,
            )

        return pid, token_embeddings, hydro_features, label


def collate_token_embedding_batch(batch):
    pids, token_embeddings, hydro_features, labels = zip(*batch)
    batch_size = len(batch)
    max_length = max(tensor.size(0) for tensor in token_embeddings)
    embedding_dim = token_embeddings[0].size(-1)

    padded_embeddings = torch.zeros((batch_size, max_length, embedding_dim), dtype=torch.float32)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)

    use_hydro = hydro_features[0] is not None
    padded_hydro = None
    if use_hydro:
        hydro_dim = hydro_features[0].size(-1)
        padded_hydro = torch.zeros((batch_size, max_length, hydro_dim), dtype=torch.float32)

    for index, token_tensor in enumerate(token_embeddings):
        length = token_tensor.size(0)
        padded_embeddings[index, :length] = token_tensor
        attention_mask[index, :length] = 1
        if use_hydro and hydro_features[index] is not None:
            padded_hydro[index, :length] = hydro_features[index]

    batch_inputs = {
        "token_embeddings": padded_embeddings,
        "attention_mask": attention_mask,
    }
    if padded_hydro is not None:
        batch_inputs["hydro_features"] = padded_hydro

    return list(pids), batch_inputs, torch.stack(list(labels), dim=0)
