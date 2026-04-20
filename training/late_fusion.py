from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from training.data.go_utils import build_propagation_indices, parse_go_obo


DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "models"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"
DEFAULT_METHODS = ["esm2", "t5", "cnn", "blast"]
METHOD_COLUMN_NAMES = {
    "esm2": "W_ESM2",
    "t5": "W_ProtT5",
    "cnn": "W_PCACNN",
    "blast": "W_BLAST",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search late fusion weights from OOF predictions")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "fusion_weights.csv",
                        help="Exact file path to save the output CSV. A JSON summary will be saved alongside it.")
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--threshold-start", type=float, default=0.1)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--save-fused-oof", action="store_true")
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH,
                        help="Path to go-basic.obo for propagation-aware Fmax (default: data/go-basic.obo)")
    parser.add_argument("--cores", type=int, default=1,
                        help="Kept for CLI compatibility; chunked fusion is single-process to cap RAM.")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Number of proteins to evaluate per block (default: 256). Lower this if RAM is tight.")
    parser.add_argument("--cache-dir", type=Path, default=None,
                        help="Directory for temporary memmap files (default: <oof-dir>/.late_fusion_cache).")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu",
                        help="Device for grid search evaluation (default: cpu). Use cuda to enable GPU acceleration.")
    parser.add_argument("--weight-batch-size", type=int, default=8,
                        help="Number of weight candidates evaluated together on CUDA (default: 8).")
    parser.add_argument("--prop-pair-batch-size", type=int, default=20000,
                        help="Number of GO propagation ancestor/descendant pairs per CUDA scatter batch.")
    return parser.parse_args()


def fold_artifact_paths(oof_dir: Path, method: str, aspect: str, fold: int) -> dict:
    prefix = oof_dir / f"{method}_{aspect}_fold{fold}"
    return {
        "pids": prefix.with_name(prefix.name + "_pids.npy"),
        "labels": prefix.with_name(prefix.name + "_labels.npy"),
        "probs": prefix.with_name(prefix.name + "_probs.npy"),
        "classes": prefix.with_name(prefix.name + "_classes.npy"),
    }


def load_fold_artifact(
    oof_dir: Path,
    method: str,
    aspect: str,
    fold: int,
    *,
    mmap_arrays: bool = False,
    metadata_only: bool = False,
) -> dict:
    required = fold_artifact_paths(oof_dir, method, aspect, fold)
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing OOF artifacts for {method} {aspect} fold {fold}: {missing_list}")

    artifact = {
        "pids": np.load(required["pids"], allow_pickle=True),
        "classes": np.load(required["classes"], allow_pickle=True),
    }
    if metadata_only:
        return artifact

    mmap_mode = "r" if mmap_arrays else None
    artifact["labels"] = np.load(required["labels"], mmap_mode=mmap_mode)
    artifact["probs"] = np.load(required["probs"], mmap_mode=mmap_mode)
    return artifact


def align_artifact(reference: dict, candidate: dict, method: str, aspect: str, fold: int) -> np.ndarray:
    if np.array_equal(reference["pids"], candidate["pids"]) and np.array_equal(reference["classes"], candidate["classes"]):
        if not np.array_equal(reference["labels"], candidate["labels"]):
            raise ValueError(f"Label mismatch for {method} {aspect} fold {fold}")
        return candidate["probs"]

    reference_pids = reference["pids"].tolist()
    candidate_pid_to_index = {pid: index for index, pid in enumerate(candidate["pids"].tolist())}
    try:
        row_index = [candidate_pid_to_index[pid] for pid in reference_pids]
    except KeyError as exc:
        raise ValueError(f"PID mismatch for {method} {aspect} fold {fold}: missing {exc}") from exc

    candidate_classes = candidate["classes"].tolist()
    class_to_index = {term: index for index, term in enumerate(candidate_classes)}
    try:
        col_index = [class_to_index[term] for term in reference["classes"].tolist()]
    except KeyError as exc:
        raise ValueError(f"Class mismatch for {method} {aspect} fold {fold}: missing {exc}") from exc

    aligned_probs = candidate["probs"][row_index][:, col_index]
    aligned_labels = candidate["labels"][row_index][:, col_index]

    if not np.array_equal(reference["labels"], aligned_labels):
        raise ValueError(f"Label mismatch after alignment for {method} {aspect} fold {fold}")

    return aligned_probs


def is_identity_indices(indices: Sequence[int], size: int) -> bool:
    return len(indices) == size and all(index == expected for expected, index in enumerate(indices))


def assign_fold_block(target: np.ndarray, rows: slice, columns: List[int], values: np.ndarray) -> None:
    if is_identity_indices(columns, target.shape[1]):
        target[rows] = values
    else:
        target[rows, columns] = values


def collect_oof_predictions(
    oof_dir: Path,
    methods: List[str],
    aspect: str,
    folds: List[int],
    cache_dir: Path,
) -> dict:
    # First pass: load only small metadata, not multi-GB probability matrices.
    union_classes_set = set()
    total_samples = 0
    for fold in folds:
        ref = load_fold_artifact(oof_dir, methods[0], aspect, fold, metadata_only=True)
        union_classes_set.update(ref["classes"].tolist())
        total_samples += len(ref["pids"])

    classes = np.array(sorted(list(union_classes_set)))
    class_to_idx = {term: i for i, term in enumerate(classes)}

    n_samples = total_samples
    n_classes = len(classes)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmpdir = tempfile.TemporaryDirectory(prefix=f"late_fusion_{aspect}_", dir=cache_dir)
    try:
        tmp_path = Path(tmpdir.name)
        stacked_pids = np.empty(n_samples, dtype=object)
        stacked_labels = np.lib.format.open_memmap(
            tmp_path / "labels.npy",
            mode="w+",
            dtype=np.int8,
            shape=(n_samples, n_classes),
        )
        stacked_labels[:] = 0
        stacked_probs = {
            method: np.lib.format.open_memmap(
                tmp_path / f"{method}_probs.npy",
                mode="w+",
                dtype=np.float16,
                shape=(n_samples, n_classes),
            )
            for method in methods
        }
        for probs in stacked_probs.values():
            probs[:] = 0.0

        current_idx = 0
        for fold in folds:
            reference = load_fold_artifact(oof_dir, methods[0], aspect, fold, mmap_arrays=True)
            n_fold_samples = len(reference["pids"])
            ref_col_idx = [class_to_idx[term] for term in reference["classes"]]
            fold_slice = slice(current_idx, current_idx + n_fold_samples)

            print(f"  Caching fold {fold}: rows {current_idx}-{current_idx + n_fold_samples - 1}")
            stacked_pids[fold_slice] = reference["pids"]
            assign_fold_block(
                stacked_labels,
                fold_slice,
                ref_col_idx,
                reference["labels"].astype(np.int8, copy=False),
            )
            assign_fold_block(
                stacked_probs[methods[0]],
                fold_slice,
                ref_col_idx,
                reference["probs"].astype(np.float16, copy=False),
            )

            for method in methods[1:]:
                candidate = load_fold_artifact(oof_dir, method, aspect, fold, mmap_arrays=True)
                aligned_to_ref_probs = align_artifact(reference, candidate, method, aspect, fold)
                assign_fold_block(
                    stacked_probs[method],
                    fold_slice,
                    ref_col_idx,
                    aligned_to_ref_probs.astype(np.float16, copy=False),
                )

            current_idx += n_fold_samples

        stacked_labels.flush()
        for probs in stacked_probs.values():
            probs.flush()

        return {
            "pids": stacked_pids,
            "labels": stacked_labels,
            "classes": classes,
            "probs": stacked_probs,
            "_tmpdir": tmpdir,
        }
    except Exception:
        tmpdir.cleanup()
        raise


def calibrate_temperature(
    probs: np.ndarray,
    labels: np.ndarray,
    chunk_size: int = 4096,
) -> float:
    """
    Fit a single temperature parameter T that minimizes NLL on the given
    labels. calibrated = sigmoid(logit(prob) / T).

    Operates in chunks to avoid materialising full float64 copies of large
    memmap arrays. Returns T >= 0.1.
    """
    from scipy.optimize import minimize_scalar

    def nll_at_temperature(log_t: float) -> float:
        t = np.exp(log_t)
        total_nll = 0.0
        n_total = 0
        for start in range(0, probs.shape[0], chunk_size):
            end = min(start + chunk_size, probs.shape[0])
            p = np.asarray(probs[start:end], dtype=np.float64).clip(1e-7, 1.0 - 1e-7)
            y = np.asarray(labels[start:end], dtype=np.float64)
            logit_p = np.log(p / (1.0 - p))
            scaled = 1.0 / (1.0 + np.exp(-logit_p / t))
            nll = -(y * np.log(scaled + 1e-12) + (1.0 - y) * np.log(1.0 - scaled + 1e-12))
            total_nll += float(nll.sum())
            n_total += nll.size
        return total_nll / max(n_total, 1)

    result = minimize_scalar(nll_at_temperature, bounds=(np.log(0.1), np.log(10.0)), method="bounded")
    return float(np.exp(result.x))


def apply_temperature_inplace(probs: np.ndarray, temperature: float, chunk_size: int = 4096) -> None:
    """Apply temperature scaling to a memmap probability array in-place."""
    if abs(temperature - 1.0) < 1e-6:
        return
    for start in range(0, probs.shape[0], chunk_size):
        end = min(start + chunk_size, probs.shape[0])
        p = np.asarray(probs[start:end], dtype=np.float32).clip(1e-7, 1.0 - 1e-7)
        logit_p = np.log(p / (1.0 - p))
        calibrated = 1.0 / (1.0 + np.exp(-logit_p / temperature))
        probs[start:end] = calibrated.astype(probs.dtype)
    if hasattr(probs, "flush"):
        probs.flush()


def generate_weight_grid(methods: List[str], step: float) -> Iterable[Dict[str, float]]:
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError(f"weight_step must evenly divide 1.0, got {step}")

    def _simplex_points(n_remaining: int, units_remaining: int):
        if n_remaining == 1:
            yield (units_remaining,)
            return
        for first in range(units_remaining + 1):
            for rest in _simplex_points(n_remaining - 1, units_remaining - first):
                yield (first,) + rest

    for point in _simplex_points(len(methods), units):
        yield {method: round(v * step, 10) for method, v in zip(methods, point)}


def generate_thresholds(start: float, stop: float, step: float) -> List[float]:
    values = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def chunk_slices(n_samples: int, chunk_size: int) -> Iterable[slice]:
    for start in range(0, n_samples, chunk_size):
        yield slice(start, min(start + chunk_size, n_samples))


def propagate_scores_inplace(scores: np.ndarray, descendant_indices: List[List[int]]) -> np.ndarray:
    for anc_idx, desc_idxs in enumerate(descendant_indices):
        if desc_idxs:
            np.maximum(
                scores[:, anc_idx],
                scores[:, desc_idxs].max(axis=1),
                out=scores[:, anc_idx],
            )
    return scores


def build_propagation_pairs(descendant_indices: List[List[int]]) -> tuple[np.ndarray, np.ndarray]:
    ancestor_indices = []
    child_indices = []
    for anc_idx, desc_idxs in enumerate(descendant_indices):
        ancestor_indices.extend([anc_idx] * len(desc_idxs))
        child_indices.extend(desc_idxs)
    return (
        np.asarray(ancestor_indices, dtype=np.int64),
        np.asarray(child_indices, dtype=np.int64),
    )


def fuse_probability_chunk(
    method_chunks: Dict[str, np.ndarray],
    methods: List[str],
    weights: Dict[str, float],
    out: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    out.fill(0.0)
    wrote_first = False
    for method in methods:
        weight = float(weights[method])
        if weight == 0.0:
            continue
        source = method_chunks[method]
        if not wrote_first:
            np.multiply(source, weight, out=out, casting="unsafe")
            wrote_first = True
        else:
            np.multiply(source, weight, out=temp, casting="unsafe")
            np.add(out, temp, out=out)
    return out


def resolve_torch_device(device_name: str):
    import torch

    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but torch.cuda.is_available() is false")
    return torch.device(device_name)


def propagate_scores_torch(
    scores,
    ancestor_indices,
    child_indices,
    pair_batch_size: int,
):
    if ancestor_indices.numel() == 0:
        return scores

    source = scores.clone()
    batch_size, rows, _ = scores.shape
    for start in range(0, ancestor_indices.numel(), pair_batch_size):
        end = min(start + pair_batch_size, ancestor_indices.numel())
        ancestors = ancestor_indices[start:end]
        children = child_indices[start:end]
        child_scores = source.index_select(2, children)
        scatter_index = ancestors.view(1, 1, -1).expand(batch_size, rows, -1)
        scores.scatter_reduce_(2, scatter_index, child_scores, reduce="amax", include_self=True)
    return scores


def accumulate_threshold_counts_cuda(
    payload: dict,
    methods: List[str],
    weights_grid: List[Dict[str, float]],
    thresholds: Sequence[float],
    prop_indices: List[List[int]],
    chunk_size: int,
    *,
    progress_label: str,
    device_name: str,
    weight_batch_size: int,
    prop_pair_batch_size: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    import torch

    device = resolve_torch_device(device_name)
    n_samples = payload["labels"].shape[0]
    threshold_values = np.asarray(thresholds, dtype=np.float32)
    true_positives = np.zeros((len(weights_grid), len(threshold_values)), dtype=np.int64)
    pred_positives = np.zeros_like(true_positives)
    total_true = 0

    weight_array = np.asarray(
        [[weights[method] for method in methods] for weights in weights_grid],
        dtype=np.float32,
    )
    thresholds_t = torch.as_tensor(threshold_values, device=device, dtype=torch.float32)
    ancestor_np, child_np = build_propagation_pairs(prop_indices)
    ancestor_t = torch.as_tensor(ancestor_np, device=device, dtype=torch.long)
    child_t = torch.as_tensor(child_np, device=device, dtype=torch.long)

    for chunk_number, rows in enumerate(chunk_slices(n_samples, chunk_size), start=1):
        labels_chunk = np.array(payload["labels"][rows], dtype=np.int8, copy=True)
        propagate_scores_inplace(labels_chunk, prop_indices)
        total_true += int(np.count_nonzero(labels_chunk))
        labels_t = torch.as_tensor(labels_chunk.astype(np.bool_, copy=False), device=device)

        method_tensors = [
            torch.as_tensor(np.asarray(payload["probs"][method][rows]), device=device, dtype=torch.float32)
            for method in methods
        ]
        method_stack = torch.stack(method_tensors, dim=0)
        labels_expanded = labels_t.unsqueeze(0)

        for start in range(0, len(weights_grid), weight_batch_size):
            end = min(start + weight_batch_size, len(weights_grid))
            weights_t = torch.as_tensor(weight_array[start:end], device=device, dtype=torch.float32)
            fused = torch.einsum("mrc,bm->brc", method_stack, weights_t)
            propagate_scores_torch(fused, ancestor_t, child_t, prop_pair_batch_size)

            for threshold_idx, threshold in enumerate(thresholds_t):
                pred = fused >= threshold
                pred_positives[start:end, threshold_idx] += pred.sum(dim=(1, 2)).cpu().numpy()
                true_positives[start:end, threshold_idx] += (pred & labels_expanded).sum(dim=(1, 2)).cpu().numpy()

            del fused, weights_t

        del labels_t, labels_expanded, method_stack, method_tensors
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  {progress_label}: processed chunk {chunk_number} ({rows.stop}/{n_samples} rows)")

    return true_positives, pred_positives, total_true


def accumulate_threshold_counts(
    payload: dict,
    methods: List[str],
    weights_grid: List[Dict[str, float]],
    thresholds: Sequence[float],
    prop_indices: List[List[int]],
    chunk_size: int,
    *,
    progress_label: str,
    device_name: str = "cpu",
    weight_batch_size: int = 1,
    prop_pair_batch_size: int = 20000,
) -> tuple[np.ndarray, np.ndarray, int]:
    if device_name != "cpu":
        return accumulate_threshold_counts_cuda(
            payload,
            methods,
            weights_grid,
            thresholds,
            prop_indices,
            chunk_size,
            progress_label=progress_label,
            device_name=device_name,
            weight_batch_size=weight_batch_size,
            prop_pair_batch_size=prop_pair_batch_size,
        )

    n_samples = payload["labels"].shape[0]
    threshold_values = np.asarray(thresholds, dtype=np.float32)
    true_positives = np.zeros((len(weights_grid), len(threshold_values)), dtype=np.int64)
    pred_positives = np.zeros_like(true_positives)
    total_true = 0

    for chunk_number, rows in enumerate(chunk_slices(n_samples, chunk_size), start=1):
        labels_chunk = np.array(payload["labels"][rows], dtype=np.int8, copy=True)
        propagate_scores_inplace(labels_chunk, prop_indices)
        total_true += int(np.count_nonzero(labels_chunk))

        method_chunks = {method: payload["probs"][method][rows] for method in methods}
        fused = np.empty(labels_chunk.shape, dtype=np.float32)
        temp = np.empty_like(fused)
        pred_mask = np.empty(labels_chunk.shape, dtype=bool)

        for weight_idx, weights in enumerate(weights_grid):
            fuse_probability_chunk(method_chunks, methods, weights, fused, temp)
            propagate_scores_inplace(fused, prop_indices)
            for threshold_idx, threshold in enumerate(threshold_values):
                np.greater_equal(fused, threshold, out=pred_mask)
                pred_positives[weight_idx, threshold_idx] += int(np.count_nonzero(pred_mask))
                np.logical_and(pred_mask, labels_chunk, out=pred_mask)
                true_positives[weight_idx, threshold_idx] += int(np.count_nonzero(pred_mask))

        print(f"  {progress_label}: processed chunk {chunk_number} ({rows.stop}/{n_samples} rows)")

    return true_positives, pred_positives, total_true


def metrics_from_counts(
    true_positives: int,
    pred_positives: int,
    total_true: int,
    threshold: float,
    *,
    fmax: float = 0.0,
    fmax_threshold: float | None = None,
) -> Dict[str, float]:
    denom_f1 = pred_positives + total_true
    return {
        "micro_f1": float((2 * true_positives) / denom_f1) if denom_f1 > 0 else 0.0,
        "micro_precision": float(true_positives / pred_positives) if pred_positives > 0 else 0.0,
        "micro_recall": float(true_positives / total_true) if total_true > 0 else 0.0,
        "fmax": float(fmax),
        "fmax_threshold": float(threshold if fmax_threshold is None else fmax_threshold),
    }


def search_best_fusion(
    payload: dict,
    methods: List[str],
    thresholds: List[float],
    weight_step: float,
    prop_indices: List[List[int]],
    chunk_size: int,
    cores: int = 1,
    device_name: str = "cpu",
    weight_batch_size: int = 1,
    prop_pair_batch_size: int = 20000,
) -> dict:
    """
    Grid-search fusion weights with bounded RAM.

    The evaluator streams row blocks from memmapped OOF predictions. It keeps
    only one fused block, one temp block, and one boolean mask in memory.
    """
    if cores != 1 and device_name == "cpu":
        print("  Note: chunked fusion ignores --cores to avoid parallel copies of large matrices.")
    grid = list(generate_weight_grid(methods, weight_step))
    if device_name == "cpu":
        print(f"  Evaluating {len(grid)} fusion candidates in chunks of {chunk_size} rows...")
    else:
        print(
            f"  Evaluating {len(grid)} fusion candidates on {device_name} "
            f"in chunks of {chunk_size} rows, {weight_batch_size} weights/batch..."
        )

    true_positives, pred_positives, total_true = accumulate_threshold_counts(
        payload,
        methods,
        grid,
        thresholds,
        prop_indices,
        chunk_size,
        progress_label="Grid search",
        device_name=device_name,
        weight_batch_size=weight_batch_size,
        prop_pair_batch_size=prop_pair_batch_size,
    )

    best = None
    for weight_idx, weights in enumerate(grid):
        # Find fmax (best F1 across all thresholds) for this weight combination
        weight_fmax = 0.0
        weight_best_threshold = float(thresholds[0])
        for threshold_idx, threshold in enumerate(thresholds):
            tp = int(true_positives[weight_idx, threshold_idx])
            pp = int(pred_positives[weight_idx, threshold_idx])
            denom = pp + total_true
            f1 = float((2 * tp) / denom) if denom > 0 else 0.0
            if f1 > weight_fmax:
                weight_fmax = f1
                weight_best_threshold = float(threshold)

        candidate = {
            "weights": weights,
            "threshold": weight_best_threshold,
            "metrics": {"fmax": weight_fmax},
        }
        if best is None or weight_fmax > best["metrics"]["fmax"]:
            best = candidate

    if best is None:
        raise RuntimeError("No fusion candidates were evaluated")

    fmax_thresholds = [round(float(value), 10) for value in np.linspace(0.01, 0.99, 99)]
    eval_thresholds = sorted(set(fmax_thresholds + [round(float(best["threshold"]), 10)]))
    best_tp, best_pred, best_total_true = accumulate_threshold_counts(
        payload,
        methods,
        [best["weights"]],
        eval_thresholds,
        prop_indices,
        chunk_size,
        progress_label="Best-weight Fmax",
        device_name=device_name,
        weight_batch_size=1,
        prop_pair_batch_size=prop_pair_batch_size,
    )
    threshold_to_idx = {round(float(value), 10): idx for idx, value in enumerate(eval_thresholds)}
    selected_idx = threshold_to_idx[round(float(best["threshold"]), 10)]

    fmax = 0.0
    fmax_threshold = float(best["threshold"])
    for threshold in fmax_thresholds:
        idx = threshold_to_idx[round(float(threshold), 10)]
        denom = int(best_pred[0, idx]) + best_total_true
        candidate_f1 = float((2 * int(best_tp[0, idx])) / denom) if denom > 0 else 0.0
        if candidate_f1 > fmax:
            fmax = candidate_f1
            fmax_threshold = float(threshold)

    best["metrics"] = metrics_from_counts(
        int(best_tp[0, selected_idx]),
        int(best_pred[0, selected_idx]),
        best_total_true,
        float(best["threshold"]),
        fmax=fmax,
        fmax_threshold=fmax_threshold,
    )

    return best


def format_weight_row(aspect: str, best: dict) -> dict:
    row = {"Aspect": aspect}
    for method, column_name in METHOD_COLUMN_NAMES.items():
        row[column_name] = float(best["weights"].get(method, 0.0))
    row["THRESHOLD"] = float(best["threshold"])
    row["F1"] = float(best["metrics"]["micro_f1"])
    row["MICRO_PRECISION"] = float(best["metrics"]["micro_precision"])
    row["MICRO_RECALL"] = float(best["metrics"]["micro_recall"])
    return row


def save_fused_oof(
    output_dir: Path,
    aspect: str,
    payload: dict,
    methods: List[str],
    best: dict,
    chunk_size: int,
) -> None:
    prefix = output_dir / f"late_fusion_{aspect}"
    np.save(prefix.with_name(prefix.name + "_pids.npy"), payload["pids"])
    np.save(prefix.with_name(prefix.name + "_labels.npy"), payload["labels"])
    np.save(prefix.with_name(prefix.name + "_classes.npy"), payload["classes"])
    probs_path = prefix.with_name(prefix.name + "_probs.npy")
    fused_out = np.lib.format.open_memmap(
        probs_path,
        mode="w+",
        dtype=np.float32,
        shape=payload["labels"].shape,
    )
    for rows in chunk_slices(payload["labels"].shape[0], chunk_size):
        method_chunks = {method: payload["probs"][method][rows] for method in methods}
        fused = np.empty((rows.stop - rows.start, payload["labels"].shape[1]), dtype=np.float32)
        temp = np.empty_like(fused)
        fuse_probability_chunk(method_chunks, methods, best["weights"], fused, temp)
        fused_out[rows] = fused
    fused_out.flush()


def main() -> None:
    args = parse_args()
    args.oof_dir = Path(args.oof_dir)
    args.output = Path(args.output)
    args.cache_dir = Path(args.cache_dir) if args.cache_dir is not None else args.oof_dir / ".late_fusion_cache"
    if args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive, got {args.chunk_size}")
    if args.weight_batch_size <= 0:
        raise ValueError(f"--weight-batch-size must be positive, got {args.weight_batch_size}")
    if args.prop_pair_batch_size <= 0:
        raise ValueError(f"--prop-pair-batch-size must be positive, got {args.prop_pair_batch_size}")
    thresholds = generate_thresholds(args.threshold_start, args.threshold_stop, args.threshold_step)

    obo_path = Path(args.obo)
    if not obo_path.exists():
        raise FileNotFoundError(
            f"GO OBO file not found: {obo_path}. "
            "Provide the correct path with --obo, or place go-basic.obo at the default location."
        )
    print(f"Loading GO ontology from {obo_path} ...")
    go_parents = parse_go_obo(obo_path)
    print(f"  Loaded {len(go_parents)} GO terms.")

    rows = []
    summary = {}
    for aspect in args.aspect:
        print(f"\nSearching late fusion for aspect={aspect} using methods={args.methods}")
        payload = collect_oof_predictions(args.oof_dir, args.methods, aspect, args.fold, args.cache_dir)
        try:
            # Temperature calibration per method
            temperatures = {}
            for method in args.methods:
                t = calibrate_temperature(payload["probs"][method], payload["labels"], chunk_size=args.chunk_size)
                temperatures[method] = t
                print(f"  Temperature for {method}: {t:.4f}")
                apply_temperature_inplace(payload["probs"][method], t, chunk_size=args.chunk_size)

            print(f"  Building propagation indices for {len(payload['classes'])} classes ...")
            prop_indices = build_propagation_indices(payload["classes"], go_parents)

            best = search_best_fusion(
                payload,
                args.methods,
                thresholds,
                args.weight_step,
                prop_indices,
                args.chunk_size,
                cores=args.cores,
                device_name=args.device,
                weight_batch_size=args.weight_batch_size,
                prop_pair_batch_size=args.prop_pair_batch_size,
            )

            rows.append(format_weight_row(aspect, best))
            summary[aspect] = {
                "weights": best["weights"],
                "threshold": best["threshold"],
                "metrics": best["metrics"],
                "temperatures": temperatures,
                "num_proteins": int(payload["labels"].shape[0]),
                "num_classes": int(payload["labels"].shape[1]),
            }
            if args.save_fused_oof:
                save_fused_oof(args.oof_dir, aspect, payload, args.methods, best, args.chunk_size)

            print(
                f"  best weights={best['weights']} threshold={best['threshold']:.2f} "
                f"propagated_fmax={best['metrics']['fmax']:.4f} "
                f"micro_f1={best['metrics']['micro_f1']:.4f}"
            )
        finally:
            payload["_tmpdir"].cleanup()

    csv_path = args.output
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    
    json_path = csv_path.with_name(csv_path.stem + "_summary.json")
    
    frame.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    print(f"\nSaved fusion weights to {csv_path}")


if __name__ == "__main__":
    main()
