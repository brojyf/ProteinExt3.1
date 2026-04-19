from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from training.data.data_utils import build_token_hydrophobicity_features, load_fasta_sequences
from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores
from training.data.embedding import (
    DEFAULT_ESM2_INTERMEDIATE_LAYER,
    DEFAULT_ESM2_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_T5_NAME,
    esm2_layer_dir,
    extract_esm2_embeddings,
    extract_t5_embeddings,
)
from submethods import MODEL_BUILDERS
from submethods.bp_blast_transfer import (
    _build_database,
    _parse_blast_hits,
    _require_blast,
    _run_blast,
    _transfer_scores,
)


DEFAULT_OUTPUT_PATH = ROOT_DIR / "predictions" / "predictions.tsv"
DEFAULT_MODELS_DIR = ROOT_DIR / "models"
DEFAULT_EMBEDDING_DIR = ROOT_DIR / "data" / "embedding"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"
DEFAULT_BATCH_SIZE = 2
DEFAULT_THRESHOLD = 0.5
DEFAULT_ESM2_EMBEDDING_DIM = 1280
DEFAULT_T5_EMBEDDING_DIM = 1024
DEFAULT_POOLING = "mean"
DEFAULT_DROPOUT = 0.4
DEFAULT_HIDDEN_DIM = 1024


class ProteinTokenInferenceDataset(Dataset):
    def __init__(
        self,
        pids: Sequence[str],
        sequences: Dict[str, str],
        embedding_dir: Path,
        plm: str,
        layer: str,
        hydro_window_size: int | None = None,
    ) -> None:
        self.pids = list(pids)
        self.sequences = sequences
        self.embedding_dir = Path(embedding_dir)
        self.plm = plm
        self.layer = layer
        self.hydro_window_size = hydro_window_size

    def __len__(self) -> int:
        return len(self.pids)

    def __getitem__(self, index: int):
        pid = self.pids[index]
        path = self.embedding_dir / self.plm / self.layer / f"{pid}.pt"
        if not path.exists():
            raise FileNotFoundError(f"Missing inference embedding for protein {pid}: {path}")

        token_embeddings = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(token_embeddings, torch.Tensor):
            raise TypeError(f"Expected tensor embedding at {path}")
        token_embeddings = token_embeddings.float()

        hydro_features = None
        if self.hydro_window_size is not None:
            hydro_features = build_token_hydrophobicity_features(
                self.sequences[pid],
                token_length=token_embeddings.size(0),
                window_size=self.hydro_window_size,
            )

        return pid, token_embeddings, hydro_features


def collate_inference_batch(batch):
    pids, token_embeddings, hydro_features = zip(*batch)
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

    return list(pids), batch_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProteinExt3 prediction pipeline")
    parser.add_argument("--method", choices=["esm2", "t5", "cnn", "blast"])
    parser.add_argument("--aspect", default="PFC", choices=["P", "F", "C", "PFC"])
    parser.add_argument("--in", dest="input", default=ROOT_DIR / "data" / "test.fasta", type=Path)
    parser.add_argument("--out", dest="output", default=DEFAULT_OUTPUT_PATH, type=Path)
    parser.add_argument("--batchsize", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--cpu", type=int, default=8, help="Number of CPU cores for BLAST and dataloader tasks")
    parser.add_argument(
        "--obo",
        type=Path,
        default=DEFAULT_OBO_PATH,
        help="Path to go-basic.obo if --propagate is enabled (default: data/go-basic.obo)",
    )
    parser.add_argument(
        "--propagate",
        action="store_true",
        help="Apply GO upward propagation before writing predictions. Disabled by default.",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_MODELS_DIR / "fusion_weights.csv",
        help="Specific path to a fusion weights CSV file (default: models/fusion_weights.csv)",
    )
    return parser.parse_args()


def is_mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if is_mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def print_device_summary(device: torch.device) -> None:
    print(
        "Device Check | "
        f"cuda_available={torch.cuda.is_available()} | "
        f"mps_available={is_mps_available()} | "
        "cpu_available=True"
    )
    if device.type == "cuda":
        print(f"Using device: cuda ({torch.cuda.get_device_name(device)})")
    elif device.type == "mps":
        print("Using device: mps (Apple Metal)")
    else:
        print("Using device: cpu")


def normalize_aspects(aspect_arg: str) -> List[str]:
    if aspect_arg == "PFC":
        return ["P", "F", "C"]
    return [aspect_arg]


def ensure_input_exists(input_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {input_path}")


def load_sequences(input_path: Path) -> Dict[str, str]:
    sequences = load_fasta_sequences(input_path)
    if not sequences:
        raise RuntimeError(f"No sequences found in input FASTA: {input_path}")
    return sequences


def methods_needed(method: str | None) -> List[str]:
    if method is None:
        return ["esm2", "t5", "cnn"]
    return [method]


def prepare_embeddings(
    *,
    sequences_by_pid: Dict[str, str],
    needed_methods: Sequence[str],
    embedding_dir: Path,
    batch_size: int,
    device: torch.device,
) -> None:
    need_esm2 = any(method in {"esm2", "cnn"} for method in needed_methods)
    need_t5 = any(method == "t5" for method in needed_methods)

    if need_esm2:
        extract_esm2_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=embedding_dir,
            pretrained_name=DEFAULT_ESM2_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
            layer_index=DEFAULT_ESM2_INTERMEDIATE_LAYER,
        )
    if need_t5:
        extract_t5_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=embedding_dir,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
        )


def checkpoint_candidates(models_dir: Path, method: str, aspect: str) -> List[Path]:
    legacy_name = "esm2_pca" if method == "cnn" else method
    return [
        models_dir / f"best_{method}_{aspect}.pt",
        models_dir / f"final_model_{legacy_name}_{aspect}.pth",
    ]


def load_checkpoint(models_dir: Path, method: str, aspect: str) -> dict:
    for candidate in checkpoint_candidates(models_dir, method, aspect):
        if candidate.exists():
            # Local training checkpoints include numpy metadata in addition to
            # tensors, which PyTorch 2.6+ rejects under the default
            # weights_only=True loader.
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            return {"payload": payload, "path": candidate}
    searched = ", ".join(str(path.name) for path in checkpoint_candidates(models_dir, method, aspect))
    raise FileNotFoundError(f"Model checkpoint not found for {method} aspect={aspect}. Searched: {searched}")


def build_args_from_checkpoint(checkpoint_payload: dict, method: str) -> SimpleNamespace:
    saved_args = dict(checkpoint_payload.get("args", {}))
    hidden_dim = saved_args.get("hidden_dim", saved_args.get("hidden", DEFAULT_HIDDEN_DIM))
    cnn_hidden_dim = saved_args.get("cnn_hidden_dim", saved_args.get("hidden", DEFAULT_HIDDEN_DIM))
    dropout = saved_args.get("dropout", DEFAULT_DROPOUT)
    pooling = saved_args.get("pooling", DEFAULT_POOLING)
    window_size = saved_args.get("window_size", 20)
    return SimpleNamespace(
        method=method,
        hidden_dim=hidden_dim,
        cnn_hidden_dim=cnn_hidden_dim,
        dropout=dropout,
        pooling=pooling,
        window_size=window_size,
        esm2_embedding_dim=saved_args.get("esm2_embedding_dim", DEFAULT_ESM2_EMBEDDING_DIM),
        t5_embedding_dim=saved_args.get("t5_embedding_dim", DEFAULT_T5_EMBEDDING_DIM),
    )


def load_classes_for_checkpoint(models_dir: Path, checkpoint_info: dict, method: str, aspect: str) -> np.ndarray:
    payload = checkpoint_info["payload"]
    if isinstance(payload, dict) and "classes" in payload:
        return np.asarray(payload["classes"])

    candidates = [
        models_dir / f"best_{method}_{aspect}_classes.npy",
        models_dir / f"classes_{aspect}.npy",
        models_dir / f"classes_{aspect}.pkl",
    ]
    for candidate in candidates:
        if not candidate.exists():
            continue
        if candidate.suffix == ".npy":
            return np.load(candidate, allow_pickle=True)
        return np.asarray(pd.read_pickle(candidate))

    raise FileNotFoundError(
        f"Classes metadata not found for {method} aspect={aspect}. "
        f"Searched checkpoint payload and {[path.name for path in candidates]}"
    )


def dataset_kwargs_for_method(method: str, args_ns: SimpleNamespace) -> dict:
    if method == "esm2":
        return {"plm": "esm2", "layer": "last", "hydro_window_size": None}
    if method == "t5":
        return {"plm": "t5", "layer": "last", "hydro_window_size": None}
    if method == "cnn":
        return {
            "plm": "esm2",
            "layer": esm2_layer_dir(DEFAULT_ESM2_INTERMEDIATE_LAYER),
            "hydro_window_size": args_ns.window_size,
        }
    raise ValueError(f"Unsupported inference method: {method}")


@torch.inference_mode()
def run_model_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_desc: str,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_pids: List[str] = []
    all_probs: List[np.ndarray] = []
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    for pids, batch_inputs in DataLoaderProgress(loader, progress_desc):
        batch_inputs = move_batch_to_device(batch_inputs, device)
        with torch.amp.autocast(autocast_device, enabled=device.type == "cuda"):
            logits = model(batch_inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_pids.extend(pids)
        all_probs.append(probs)

    return {
        "pids": np.asarray(all_pids),
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32),
    }


def move_batch_to_device(batch_inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch_inputs.items():
        moved[key] = value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
    return moved


def DataLoaderProgress(loader: DataLoader, desc: str):
    return tqdm(loader, desc=desc, leave=True, dynamic_ncols=True)


def align_probs_to_reference(
    *,
    pids: np.ndarray,
    source_pids: np.ndarray,
    source_probs: np.ndarray,
    reference_classes: np.ndarray,
    source_classes: np.ndarray,
) -> np.ndarray:
    pid_to_index = {pid: index for index, pid in enumerate(source_pids.tolist())}
    row_index = [pid_to_index[pid] for pid in pids.tolist()]

    source_class_to_index = {term: index for index, term in enumerate(source_classes.tolist())}
    aligned = np.zeros((len(pids), len(reference_classes)), dtype=np.float32)
    for col_index, term in enumerate(reference_classes.tolist()):
        source_col = source_class_to_index.get(term)
        if source_col is not None:
            aligned[:, col_index] = source_probs[row_index, source_col]
    return aligned


def predict_for_method(
    *,
    method: str,
    aspect: str,
    models_dir: Path,
    embedding_dir: Path,
    sequences_by_pid: Dict[str, str],
    batch_size: int,
    device: torch.device,
    num_workers: int = 0,
) -> dict:
    checkpoint_info = load_checkpoint(models_dir, method, aspect)
    checkpoint_payload = checkpoint_info["payload"]
    model_args = build_args_from_checkpoint(checkpoint_payload, method)
    classes = load_classes_for_checkpoint(models_dir, checkpoint_info, method, aspect)
    model = MODEL_BUILDERS[method](model_args, num_classes=len(classes))
    state_dict = (
        checkpoint_payload["model_state_dict"]
        if isinstance(checkpoint_payload, dict) and "model_state_dict" in checkpoint_payload
        else checkpoint_payload
    )
    model.load_state_dict(state_dict)
    model.to(device)

    pids = sorted(sequences_by_pid)
    dataset = ProteinTokenInferenceDataset(
        pids=pids,
        sequences=sequences_by_pid,
        embedding_dir=embedding_dir,
        **dataset_kwargs_for_method(method, model_args),
    )
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": collate_inference_batch,
        "pin_memory": device.type == "cuda",
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(
        dataset,
        **loader_kwargs,
    )
    predictions = run_model_inference(
        model,
        loader,
        device,
        progress_desc=f"Predicting {method} aspect={aspect}",
    )
    threshold = float(
        checkpoint_payload.get("metrics", {}).get("fmax_threshold", DEFAULT_THRESHOLD)
        if isinstance(checkpoint_payload, dict)
        else DEFAULT_THRESHOLD
    )
    del model, state_dict, checkpoint_payload
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {
        "pids": predictions["pids"],
        "probs": predictions["probs"],
        "classes": classes,
        "threshold": threshold,
    }


def union_classes(results_by_method: Dict[str, dict]) -> np.ndarray:
    seen = set()
    ordered: List[str] = []
    for method in ("esm2", "t5", "cnn"):
        result = results_by_method.get(method)
        if result is None:
            continue
        for term in result["classes"].tolist():
            if term not in seen:
                seen.add(term)
                ordered.append(term)
    return np.asarray(ordered, dtype=object)


def load_fusion_weights(path: Path, aspect: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Fusion weights not found: {path}")
    frame = pd.read_csv(path)
    row = frame[frame["Aspect"] == aspect]
    if row.empty:
        raise ValueError(f"No fusion weights found for aspect={aspect} in {path}")
    item = row.iloc[0].to_dict()
    return {
        "esm2": float(item.get("W_ESM2", 0.0)),
        "t5": float(item.get("W_ProtT5", 0.0)),
        "cnn": float(item.get("W_PCACNN", 0.0)),
        "blast": float(item.get("W_BLAST", 0.0)),
        "threshold": float(item.get("THRESHOLD", DEFAULT_THRESHOLD)),
    }


def run_blast_inference(
    *,
    query_fasta: Path,
    query_pids: np.ndarray,
    classes: np.ndarray,
    aspect: str,
    num_threads: int = 8,
) -> np.ndarray:
    _require_blast()
    blast_dir = ROOT_DIR / "data" / "blast"
    db_prefix = _build_database(blast_dir / "blast.fasta", blast_dir / "cache")
    output_path = blast_dir / "cache" / "query_vs_blast.tsv"
    _run_blast(query_fasta, db_prefix, output_path, max_hits=10, evalue=1e-3, num_threads=num_threads)
    hits = _parse_blast_hits(output_path)
    labels_df = pd.read_csv(blast_dir / "blast.tsv", sep="\t")
    return _transfer_scores(query_pids, hits, labels_df, classes, aspect=aspect)


def collect_prediction_rows(
    *,
    pids: np.ndarray,
    probs: np.ndarray,
    classes: np.ndarray,
    progress_desc: str = "Collecting predictions",
) -> List[tuple[str, str, float]]:
    rows: List[tuple[str, str, float]] = []
    pid_list = pids.tolist()
    for row_index, pid in tqdm(
        enumerate(pid_list),
        total=len(pid_list),
        desc=progress_desc,
        leave=True,
        dynamic_ncols=True,
    ):
        indices = np.arange(probs.shape[1], dtype=np.int64)
        indices = indices[np.argsort(probs[row_index, indices])[::-1]]
        for class_index in indices.tolist():
            rows.append((pid, str(classes[class_index]), float(probs[row_index, class_index])))
    return rows


def load_go_parents(obo_path: Path) -> dict | None:
    """Load GO ontology for propagation. Returns None (with a warning) if file not found."""
    if not obo_path.exists():
        print(f"[warn] go-basic.obo not found at {obo_path}; skipping GO propagation.")
        return None
    print(f"Loading GO ontology from {obo_path} ...")
    parents = parse_go_obo(obo_path)
    print(f"  Loaded {len(parents)} GO terms.")
    return parents


def apply_go_propagation(
    probs: np.ndarray,
    classes: np.ndarray,
    go_parents: dict,
) -> np.ndarray:
    """Propagate scores upward through GO DAG: parent = max(parent, max(children))."""
    prop_indices = build_propagation_indices(classes, go_parents)
    return propagate_scores(probs, prop_indices)


def write_predictions(output_path: Path, rows: Sequence[tuple[str, str, float]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for pid, term, score in tqdm(rows, desc="Writing predictions", leave=True, dynamic_ncols=True):
            writer.writerow([pid, term, f"{score:.6f}"])


def main() -> None:
    args = parse_args()
    ensure_input_exists(args.input)

    sequences_by_pid = load_sequences(args.input)
    aspects = normalize_aspects(args.aspect)
    output_rows: List[tuple[str, str, float]] = []
    go_parents = load_go_parents(args.obo) if args.propagate else None

    if args.method == "blast":
        # Blast-only inference: no embeddings or model checkpoints needed
        blast_dir = ROOT_DIR / "data" / "blast"
        labels_df = pd.read_csv(blast_dir / "blast.tsv", sep="\t")
        for aspect in aspects:
            query_pids = np.asarray(sorted(sequences_by_pid))
            classes = np.asarray(sorted(labels_df[labels_df["aspect"] == aspect]["term"].unique()))
            blast_scores = run_blast_inference(
                query_fasta=args.input,
                query_pids=query_pids,
                classes=classes,
                aspect=aspect,
                num_threads=args.cpu,
            )
            if go_parents is not None:
                blast_scores = apply_go_propagation(blast_scores, classes, go_parents)
            output_rows.extend(
                collect_prediction_rows(
                    pids=query_pids,
                    probs=blast_scores,
                    classes=classes,
                    progress_desc=f"Collecting blast aspect={aspect}",
                )
            )
    else:
        device = resolve_device()
        print_device_summary(device)
        fusion_configs = {}
        if args.method is None:
            fusion_configs = {aspect: load_fusion_weights(args.weights, aspect) for aspect in aspects}
            needed_methods = [
                method
                for method in ("esm2", "t5", "cnn")
                if any(config.get(method, 0.0) > 0 for config in fusion_configs.values())
            ]
        else:
            needed_methods = methods_needed(args.method)

        embedding_dir = DEFAULT_EMBEDDING_DIR
        embedding_dir.mkdir(parents=True, exist_ok=True)
        prepare_embeddings(
            sequences_by_pid=sequences_by_pid,
            needed_methods=needed_methods,
            embedding_dir=embedding_dir,
            batch_size=args.batchsize,
            device=device,
        )

        for aspect in tqdm(aspects, desc="Predicting aspects", leave=True, dynamic_ncols=True):
            if args.method is not None:
                # Single neural method
                result = predict_for_method(
                    method=args.method,
                    aspect=aspect,
                    models_dir=DEFAULT_MODELS_DIR,
                    embedding_dir=embedding_dir,
                    sequences_by_pid=sequences_by_pid,
                    batch_size=args.batchsize,
                    device=device,
                    num_workers=args.cpu,
                )
                if go_parents is not None:
                    print(f"Applying GO propagation for {args.method} aspect={aspect} ...")
                    result["probs"] = apply_go_propagation(
                        result["probs"], result["classes"], go_parents
                    )
                output_rows.extend(
                    collect_prediction_rows(
                        pids=result["pids"],
                        probs=result["probs"],
                        classes=result["classes"],
                        progress_desc=f"Collecting {args.method} aspect={aspect}",
                    )
                )
                continue

            # Full late fusion: esm2 + t5 + cnn + blast
            fusion_cfg = fusion_configs[aspect]
            neural_methods = [
                method for method in ("esm2", "t5", "cnn") if fusion_cfg.get(method, 0.0) > 0
            ]
            component_results = {}
            for method in tqdm(
                neural_methods,
                desc=f"Neural methods aspect={aspect}",
                leave=True,
                dynamic_ncols=True,
            ):
                component_results[method] = predict_for_method(
                    method=method,
                    aspect=aspect,
                    models_dir=DEFAULT_MODELS_DIR,
                    embedding_dir=embedding_dir,
                    sequences_by_pid=sequences_by_pid,
                    batch_size=args.batchsize,
                    device=device,
                    num_workers=args.cpu,
                )

            if component_results:
                reference_method = neural_methods[0]
                reference_pids = component_results[reference_method]["pids"]
                fusion_classes = union_classes(component_results)
            else:
                labels_df = pd.read_csv(ROOT_DIR / "data" / "blast" / "blast.tsv", sep="\t")
                reference_pids = np.asarray(sorted(sequences_by_pid))
                fusion_classes = np.asarray(sorted(labels_df[labels_df["aspect"] == aspect]["term"].unique()))

            fused_probs = np.zeros((len(reference_pids), len(fusion_classes)), dtype=np.float32)
            total_weight = 0.0
            for method in tqdm(
                neural_methods,
                desc=f"Fusing neural methods aspect={aspect}",
                leave=True,
                dynamic_ncols=True,
            ):
                weight = fusion_cfg[method]
                if weight <= 0:
                    continue
                aligned_probs = align_probs_to_reference(
                    pids=reference_pids,
                    source_pids=component_results[method]["pids"],
                    source_probs=component_results[method]["probs"],
                    reference_classes=fusion_classes,
                    source_classes=component_results[method]["classes"],
                )
                fused_probs += float(weight) * aligned_probs
                total_weight += float(weight)

            blast_weight = fusion_cfg.get("blast", 0.0)
            if blast_weight > 0:
                blast_scores = run_blast_inference(
                    query_fasta=args.input,
                    query_pids=reference_pids,
                    classes=fusion_classes,
                    aspect=aspect,
                    num_threads=args.cpu,
                )
                fused_probs += blast_weight * blast_scores
                total_weight += blast_weight

            if total_weight <= 0:
                raise RuntimeError(f"Fusion weights for aspect={aspect} sum to zero")
            fused_probs /= total_weight

            if go_parents is not None:
                print(f"Applying GO propagation for late fusion aspect={aspect} ...")
                fused_probs = apply_go_propagation(fused_probs, fusion_classes, go_parents)
            output_rows.extend(
                collect_prediction_rows(
                    pids=reference_pids,
                    probs=fused_probs,
                    classes=fusion_classes,
                    progress_desc=f"Collecting late fusion aspect={aspect}",
                )
            )
            del component_results, fused_probs
            if device.type == "cuda":
                torch.cuda.empty_cache()

    write_predictions(args.output, output_rows)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
