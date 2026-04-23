from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Sequence

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from submethods import MODEL_BUILDERS
from submethods.bp_blast_transfer import _build_database, _parse_blast_hits, _require_blast, _run_blast, _transfer_scores
from training.data.data_utils import (
    MultiEmbeddingDataset,
    build_sequence_protein_features,
    collate_multi_embedding_batch,
    load_fasta_sequences,
)
from training.data.embedding import (
    DEFAULT_ESM2_INTERMEDIATE_LAYER,
    DEFAULT_ESM2_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_T5_NAME,
    esm2_layer_dir,
    extract_esm2_embeddings,
    extract_t5_embeddings,
)
from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores
from training.train import resolve_device
from training.trainer import predict as predict_batches

DEFAULT_OUTPUT_PATH = ROOT_DIR / "predictions" / "ProteinExt3.1.tsv"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"
PREDICT_EMBEDDING_DIR = ROOT_DIR / "data" / "embedding"
FUSION_COMPONENTS = {
    "esm2_last": "last",
    "esm2_l20": "l20",
    "prott5": "t5",
    "blast": "blast",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict GO terms from protein FASTA")
    parser.add_argument("--in", dest="input", type=Path, default=ROOT_DIR / "data" / "test.fasta")
    parser.add_argument("--out", dest="output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--method", choices=["fusion", "esm2_last", "esm2_l20", "prott5", "blast", "esm2", "t5"], default="fusion")
    parser.add_argument("--aspect", choices=["P", "F", "C", "PFC"], default="PFC")
    parser.add_argument("--batch-size", "--batchsize", dest="batch_size", type=int, default=2)
    parser.add_argument("--cpu", type=int, default=8)
    parser.add_argument("--weight", "--weights", dest="weights", type=Path, default=DEFAULT_MODELS_DIR / "fusion_weights.csv")
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--no-propagate", action="store_true")
    parser.add_argument("--apply-th", action="store_true", help="Apply fusion threshold to filter predictions")
    return parser.parse_args()


def normalize_aspects(aspect: str) -> List[str]:
    return ["P", "F", "C"] if aspect == "PFC" else [aspect]


def ensure_embeddings(sequences_by_pid: Dict[str, str], batch_size: int, device: torch.device, methods: Sequence[str]) -> None:
    needs_esm2 = any(method in {"esm2_last", "esm2_l20"} for method in methods)
    needs_t5 = "prott5" in methods
    missing_esm2 = [
        pid for pid in sequences_by_pid
        if not ((PREDICT_EMBEDDING_DIR / "esm2" / "last" / f"{pid}.pt").exists()
                and (PREDICT_EMBEDDING_DIR / "esm2" / esm2_layer_dir(DEFAULT_ESM2_INTERMEDIATE_LAYER) / f"{pid}.pt").exists())
    ] if needs_esm2 else []
    missing_t5 = [pid for pid in sequences_by_pid if not (PREDICT_EMBEDDING_DIR / "t5" / "last" / f"{pid}.pt").exists()] if needs_t5 else []
    if missing_esm2 or missing_t5:
        print(f"Embedding cache: {PREDICT_EMBEDDING_DIR}")
        print(f"Embedding extraction device: {device.type}")
    if missing_esm2:
        extract_esm2_embeddings(
            sequences_by_pid={pid: sequences_by_pid[pid] for pid in missing_esm2},
            output_dir=PREDICT_EMBEDDING_DIR,
            pretrained_name=DEFAULT_ESM2_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
            layer_index=DEFAULT_ESM2_INTERMEDIATE_LAYER,
        )
    if missing_t5:
        extract_t5_embeddings(
            sequences_by_pid={pid: sequences_by_pid[pid] for pid in missing_t5},
            output_dir=PREDICT_EMBEDDING_DIR,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
        )


def normalize_method(method: str) -> str:
    return {"esm2": "esm2_last", "t5": "prott5"}.get(method, method)


def checkpoint_path(method: str, aspect: str) -> Path:
    return DEFAULT_MODELS_DIR / f"{method}_{aspect}.pt"


def model_args_from_checkpoint(payload: dict) -> SimpleNamespace:
    saved = dict(payload.get("args", {}))
    return SimpleNamespace(
        esm2_embedding_dim=saved.get("esm2_embedding_dim", 1280),
        t5_embedding_dim=saved.get("t5_embedding_dim", 1024),
        protein_feature_dim=saved.get("protein_feature_dim", 63),
        hidden_dim=saved.get("hidden_dim", 2048),
        bottleneck=saved.get("bottleneck", 1024),
        dropout=saved.get("dropout", 0.3),
    )


def run_chain_inference(
    method: str,
    aspect: str,
    sequences_by_pid: Dict[str, str],
    batch_size: int,
    device: torch.device,
) -> dict:
    path = checkpoint_path(method, aspect)
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint for method={method} aspect={aspect}: {path}")
    pids = sorted(sequences_by_pid)
    features = {pid: build_sequence_protein_features(seq) for pid, seq in sequences_by_pid.items()}
    dataset = MultiEmbeddingDataset(pids, None, PREDICT_EMBEDDING_DIR, features, chain=method)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_multi_embedding_batch)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    classes = np.asarray(payload["classes"])
    model = MODEL_BUILDERS[method](model_args_from_checkpoint(payload), len(classes))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    result = predict_batches(model, loader, device, f"predict {path.stem}")
    return {"pids": result["pids"], "classes": classes, "probs": result["probs"].astype(np.float32)}


def load_fusion_weights(path: Path, aspect: str) -> dict:
    frame = pd.read_csv(path)
    row = frame[frame["aspect"] == aspect]
    if row.empty:
        raise ValueError(f"No fusion weights for aspect={aspect} in {path}")
    item = row.iloc[0].to_dict()
    return {
        "esm2_last": float(item.get("w_last", 0.0)),
        "esm2_l20": float(item.get("w_l20", 0.0)),
        "prott5": float(item.get("w_t5", 0.0)),
        "blast": float(item.get("w_blast", 0.0)),
        "threshold": float(item.get("thr", 0.5)),
    }


def align_probs(reference_pids: np.ndarray, reference_classes: np.ndarray, result: dict) -> np.ndarray:
    pid_to_index = {pid: index for index, pid in enumerate(result["pids"].tolist())}
    class_to_index = {term: index for index, term in enumerate(result["classes"].tolist())}
    aligned = np.zeros((len(reference_pids), len(reference_classes)), dtype=np.float32)
    rows = [pid_to_index[pid] for pid in reference_pids.tolist()]
    for col, term in enumerate(reference_classes.tolist()):
        source_col = class_to_index.get(term)
        if source_col is not None:
            aligned[:, col] = result["probs"][rows, source_col]
    return aligned


def load_blast_labels() -> pd.DataFrame:
    labels_path = ROOT_DIR / "data" / "blast" / "blast.tsv"
    if not labels_path.exists():
        raise FileNotFoundError("BLAST inference requires data/blast/blast.tsv")
    return pd.read_csv(labels_path, sep="\t")


def run_blast_search(input_fasta: Path, cpu: int) -> Dict[str, List[dict]]:
    _require_blast()
    blast_dir = ROOT_DIR / "data" / "blast"
    fasta_path = blast_dir / "blast.fasta"
    if not fasta_path.exists():
        raise FileNotFoundError("BLAST inference requires data/blast/blast.fasta")
    db_prefix = _build_database(fasta_path, blast_dir / "cache")
    output_path = blast_dir / "cache" / "query_vs_blast.tsv"
    _run_blast(input_fasta, db_prefix, output_path, max_hits=15, evalue=1e-3, num_threads=cpu)
    return _parse_blast_hits(output_path)


def blast_classes(labels_df: pd.DataFrame, aspect: str) -> np.ndarray:
    return np.asarray(sorted(labels_df[labels_df["aspect"] == aspect]["term"].unique()), dtype=object)


def transfer_blast_scores(
    hits: Dict[str, List[dict]],
    labels_df: pd.DataFrame,
    pids: np.ndarray,
    classes: np.ndarray,
    aspect: str,
) -> np.ndarray:
    labels_df = labels_df[labels_df["aspect"] == aspect]
    return _transfer_scores(pids, hits, labels_df, classes)


def run_blast_inference(input_fasta: Path, pids: np.ndarray, classes: np.ndarray, aspect: str, cpu: int) -> np.ndarray:
    labels_df = load_blast_labels()
    return transfer_blast_scores(run_blast_search(input_fasta, cpu), labels_df, pids, classes, aspect)


def collect_rows(pids: np.ndarray, classes: np.ndarray, probs: np.ndarray, threshold: float | None = None) -> List[tuple[str, str, float]]:
    rows: List[tuple[str, str, float]] = []
    for row_index, pid in enumerate(pids.tolist()):
        order = np.argsort(probs[row_index])[::-1]
        for col in order.tolist():
            score = float(probs[row_index, col])
            if threshold is not None and score < threshold:
                break
            rows.append((pid, str(classes[col]), score))
    return rows


def write_predictions(path: Path, rows: Sequence[tuple[str, str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for pid, term, score in rows:
            writer.writerow([pid, term, f"{score:.6f}"])


def component_output_path(output_path: Path, suffix: str) -> Path:
    NAME_MAP = {
        "last": "ESM2",
        "l20": "ESM2-L20",
        "t5": "ProtT5",
        "blast": "BLAST"
    }

    return output_path.with_name(
        f"{output_path.stem}-{NAME_MAP.get(suffix, suffix)}{output_path.suffix}"
    )


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input FASTA not found: {args.input}")
    sequences = load_fasta_sequences(args.input)
    if not sequences:
        raise RuntimeError(f"No sequences found in {args.input}")
    aspects = normalize_aspects(args.aspect)
    parents = parse_go_obo(args.obo) if not args.no_propagate else None
    device = resolve_device(args.device)
    rows: List[tuple[str, str, float]] = []
    method = normalize_method(args.method)
    component_rows: Dict[str, List[tuple[str, str, float]]] = {
        suffix: [] for suffix in FUSION_COMPONENTS.values()
    } if method == "fusion" else {}
    neural_methods = ["esm2_last", "esm2_l20", "prott5"] if method == "fusion" else [method]
    if method != "blast":
        ensure_embeddings(sequences, args.batch_size, device, neural_methods)
    blast_labels = load_blast_labels() if method in {"blast", "fusion"} else None
    blast_hits = run_blast_search(args.input, args.cpu) if method in {"blast", "fusion"} else None
    for aspect in tqdm(aspects, desc="aspects", dynamic_ncols=True):
        if method == "blast":
            if blast_hits is None or blast_labels is None:
                raise RuntimeError("BLAST hits were not initialized")
            classes = blast_classes(blast_labels, aspect)
            pids = np.asarray(sorted(sequences))
            probs = transfer_blast_scores(blast_hits, blast_labels, pids, classes, aspect)
        else:
            if method == "fusion":
                if blast_hits is None or blast_labels is None:
                    raise RuntimeError("BLAST hits were not initialized")
                weights = load_fusion_weights(args.weights, aspect)
                component_results = {
                    chain: run_chain_inference(chain, aspect, sequences, args.batch_size, device)
                    for chain in ("esm2_last", "esm2_l20", "prott5")
                }
                if not component_results:
                    raise RuntimeError(f"Fusion weights contain no neural component for aspect={aspect}")
                first = next(iter(component_results.values()))
                pids = first["pids"]
                classes = first["classes"]
                neural_probs = np.zeros_like(first["probs"], dtype=np.float32)
                component_probs = {}
                for chain, result in component_results.items():
                    aligned = align_probs(pids, classes, result)
                    component_probs[FUSION_COMPONENTS[chain]] = aligned
                    neural_probs += weights[chain] * aligned
                blast_probs = transfer_blast_scores(blast_hits, blast_labels, pids, classes, aspect)
                component_probs["blast"] = blast_probs
                probs = neural_probs + weights["blast"] * blast_probs
            else:
                component_probs = {}
                result = run_chain_inference(method, aspect, sequences, args.batch_size, device)
                pids = result["pids"]
                classes = result["classes"]
                probs = result["probs"]
        if parents is not None:
            prop_indices = build_propagation_indices(classes, parents)
            if method == "fusion":
                component_probs = {
                    name: propagate_scores(values, prop_indices)
                    for name, values in component_probs.items()
                }
            probs = propagate_scores(probs, prop_indices)
        if method == "fusion":
            for name, values in component_probs.items():
                component_rows[name].extend(collect_rows(pids, classes, values))
        th = weights["threshold"] if method == "fusion" and args.apply_th else None
        rows.extend(collect_rows(pids, classes, probs, threshold=th))
    write_predictions(args.output, rows)
    for suffix, method_rows in component_rows.items():
        path = component_output_path(args.output, suffix)
        write_predictions(path, method_rows)
        print(f"Saved {suffix} predictions to {path}")
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
