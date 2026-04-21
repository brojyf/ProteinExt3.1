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
PROTEIN_FEATURES_DIR = TRAINING_DATA_DIR / "protein_features"


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

STANDARD_AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")

AA_GROUPS = {
    "hydrophobic": tuple("AILMFWVYC"),
    "polar": tuple("STNQCY"),
    "aromatic": tuple("FWY"),
    "small": tuple("AGST"),
    "tiny": tuple("AGS"),
    "acidic": tuple("DE"),
    "basic": tuple("KRH"),
    "charged": tuple("DEKRH"),
    "sulfur": tuple("CM"),
    "turn_favoring": tuple("NPGSD"),
}

AA_MASS = {
    "A": 89.09,
    "C": 121.16,
    "D": 133.10,
    "E": 147.13,
    "F": 165.19,
    "G": 75.07,
    "H": 155.16,
    "I": 131.17,
    "K": 146.19,
    "L": 131.17,
    "M": 149.21,
    "N": 132.12,
    "P": 115.13,
    "Q": 146.15,
    "R": 174.20,
    "S": 105.09,
    "T": 119.12,
    "V": 117.15,
    "W": 204.23,
    "Y": 181.19,
}

POSITIVE_PKA = {"N_TERM": 9.69, "K": 10.5, "R": 12.4, "H": 6.0}
NEGATIVE_PKA = {"C_TERM": 2.34, "D": 3.9, "E": 4.1, "C": 8.3, "Y": 10.1}
PROTEIN_FEATURE_WINDOWS = (7, 15, 25)
PROTEIN_FEATURE_DIM = (
    len(STANDARD_AMINO_ACIDS)
    + len(AA_GROUPS)
    + 2
    + 5
    + 6
    + len(PROTEIN_FEATURE_WINDOWS) * 6
    + 2
)


def _safe_fraction(count: float, length: int) -> float:
    return float(count) / float(length) if length > 0 else 0.0


def _net_charge_at_ph(sequence: str, ph: float = 7.0) -> float:
    counts = {aa: sequence.count(aa) for aa in STANDARD_AMINO_ACIDS}
    charge = 1.0 / (1.0 + 10.0 ** (ph - POSITIVE_PKA["N_TERM"]))
    charge -= 1.0 / (1.0 + 10.0 ** (NEGATIVE_PKA["C_TERM"] - ph))

    for aa in ("K", "R", "H"):
        charge += counts[aa] / (1.0 + 10.0 ** (ph - POSITIVE_PKA[aa]))
    for aa in ("D", "E", "C", "Y"):
        charge -= counts[aa] / (1.0 + 10.0 ** (NEGATIVE_PKA[aa] - ph))
    return float(charge)


def _rolling_stats(values: np.ndarray, window_size: int) -> List[float]:
    if values.size == 0:
        return [0.0] * 6

    rolled = pd.Series(values).rolling(window=window_size, min_periods=1, center=True)
    rolling_mean = rolled.mean().to_numpy(dtype=np.float32)
    rolling_var = rolled.var(ddof=0).fillna(0.0).to_numpy(dtype=np.float32)
    stderr = float(rolling_mean.std(ddof=0) / np.sqrt(max(rolling_mean.size, 1)))
    return [
        float(rolling_mean.mean()),
        float(rolling_mean.var(ddof=0)),
        stderr,
        float(rolling_mean.max()),
        float(rolling_var.mean()),
        float(rolling_var.max()),
    ]


def build_sequence_protein_features(sequence: str) -> torch.Tensor:
    sequence = "".join(residue for residue in sequence.upper() if residue.isalpha())
    valid = [residue for residue in sequence if residue in STANDARD_AMINO_ACIDS]
    length = len(valid)
    counts = {aa: valid.count(aa) for aa in STANDARD_AMINO_ACIDS}

    features: List[float] = []

    # Amino acid composition.
    features.extend(_safe_fraction(counts[aa], length) for aa in STANDARD_AMINO_ACIDS)

    # Broad residue-group composition useful for fold, localization, and binding signals.
    for residues in AA_GROUPS.values():
        features.append(_safe_fraction(sum(counts[aa] for aa in residues), length))

    clipped_length = min(length, 4096)
    features.append(float(np.log1p(length) / np.log1p(4096)))
    features.append(float(clipped_length / 4096.0))

    net_charge = _net_charge_at_ph("".join(valid)) if length > 0 else 0.0
    positive_fraction = _safe_fraction(sum(counts[aa] for aa in ("K", "R", "H")), length)
    negative_fraction = _safe_fraction(sum(counts[aa] for aa in ("D", "E")), length)
    features.extend(
        [
            float(net_charge / max(length, 1)),
            float(abs(net_charge) / max(length, 1)),
            positive_fraction,
            negative_fraction,
            float(net_charge / np.sqrt(max(length, 1))),
        ]
    )

    # Kyte-Doolittle hydrophobicity, scaled to roughly [-1, 1].
    hydro_values = np.asarray([HYDRO_MAP.get(residue, 0.0) / 5.0 for residue in valid], dtype=np.float32)
    if hydro_values.size == 0:
        features.extend([0.0] * 6)
    else:
        features.extend(
            [
                float(hydro_values.mean()),
                float(hydro_values.var(ddof=0)),
                float(hydro_values.std(ddof=0)),
                float(hydro_values.std(ddof=0) / np.sqrt(max(hydro_values.size, 1))),
                float(hydro_values.min()),
                float(hydro_values.max()),
            ]
        )

    for window_size in PROTEIN_FEATURE_WINDOWS:
        features.extend(_rolling_stats(hydro_values, window_size))

    if length == 0:
        features.extend([0.0, 0.0])
    else:
        average_mass = sum(counts[aa] * AA_MASS[aa] for aa in STANDARD_AMINO_ACIDS) / length
        aliphatic_index = (
            counts["A"] + 2.9 * counts["V"] + 3.9 * (counts["I"] + counts["L"])
        ) / length
        features.extend([float(average_mass / 200.0), float(aliphatic_index)])

    if len(features) != PROTEIN_FEATURE_DIM:
        raise RuntimeError(f"Protein feature dimension mismatch: {len(features)} != {PROTEIN_FEATURE_DIM}")
    return torch.tensor(features, dtype=torch.float32)


def build_and_save_protein_features(
    sequences_by_pid: Dict[str, str],
    output_path: Path,
) -> Dict[str, torch.Tensor]:
    cache: Dict[str, torch.Tensor] = {}
    for pid in sorted(sequences_by_pid):
        cache[pid] = build_sequence_protein_features(sequences_by_pid[pid])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, output_path)
    return cache


def load_protein_features_cache(path: Path) -> Dict[str, torch.Tensor]:
    return torch.load(path, map_location="cpu", weights_only=True)


def build_token_hydrophobicity_features(sequence: str, token_length: int, window_size: int) -> torch.Tensor:
    features = np.zeros((token_length, 2), dtype=np.float32)
    values = np.asarray([HYDRO_MAP.get(residue, 0.0) for residue in sequence], dtype=np.float32)
    if values.size == 0:
        return torch.from_numpy(features)

    half_window = max(window_size // 2, 1)
    local = np.zeros((values.shape[0], 2), dtype=np.float32)
    
    window = half_window * 2 + 1
    rolled = pd.Series(values).rolling(window=window, min_periods=1, center=True)
    local[:, 0] = rolled.mean().to_numpy()
    local[:, 1] = rolled.var(ddof=0).fillna(0.0).to_numpy()

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
        include_protein_features: bool = False,
        protein_features_cache: Dict[str, torch.Tensor] | None = None,
    ) -> None:
        self.pids = list(pids)
        self.sequences = sequences
        self.labels = labels
        self.embedding_dir = Path(embedding_dir)
        self.plm = plm
        self.layer = layer
        self.hydro_window_size = hydro_window_size
        self.include_protein_features = include_protein_features
        self._protein_feature_cache: Dict[str, torch.Tensor] = protein_features_cache or {}

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

        token_embeddings = torch.load(path, map_location="cpu", weights_only=True)
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

        protein_features = None
        if self.include_protein_features:
            protein_features = self._protein_feature_cache.get(pid)
            if protein_features is None:
                protein_features = build_sequence_protein_features(self.sequences[pid])
                self._protein_feature_cache[pid] = protein_features

        return pid, token_embeddings, hydro_features, protein_features, label


def collate_token_embedding_batch(batch):
    pids, token_embeddings, hydro_features, protein_features, labels = zip(*batch)
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

    use_protein_features = protein_features[0] is not None
    stacked_protein_features = None
    if use_protein_features:
        stacked_protein_features = torch.stack(list(protein_features), dim=0)

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
    if stacked_protein_features is not None:
        batch_inputs["protein_features"] = stacked_protein_features

    if protein_features[0] is not None:
        batch_inputs["protein_features"] = torch.stack(list(protein_features), dim=0)

    return list(pids), batch_inputs, torch.stack(list(labels), dim=0)
