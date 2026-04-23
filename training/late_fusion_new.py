from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores

DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "latefusion_new.csv"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"
METHODS = ("esm2_last", "esm2_l20", "prott5", "blast")
METHOD_COLUMNS = {
    "esm2_last": "last",
    "esm2_l20": "l20",
    "prott5": "t5",
    "blast": "blast",
}
FMAX_THRESHOLDS = np.linspace(0.01, 0.99, 99)
CPU_BATCH_BYTES = 500_000_000
CUDA_WORKING_BYTES_FRACTION = 0.7
CUDA_PROPAGATION_CLASS_LIMIT = 5_000
MPS_BATCH_BYTES = 6_000_000_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OOF late-fusion weights by fused FMAX")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--jobs", type=int, default=0, help="Parallel aspect jobs; 0 uses one job per requested aspect")
    return parser.parse_args()


def artifact_prefix(oof_dir: Path, method: str, aspect: str, fold: int) -> Path:
    return oof_dir / f"{method}_{aspect}_fold_{fold}"


def load_array(oof_dir: Path, method: str, aspect: str, fold: int, kind: str, mmap_mode: str | None = None) -> np.ndarray:
    prefix = artifact_prefix(oof_dir, method, aspect, fold)
    path = prefix.with_name(prefix.name + f"_{kind}.npy")
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF artifact: {path}")
    return np.load(path, allow_pickle=True, mmap_mode=mmap_mode)


def load_metrics(oof_dir: Path, method: str, aspect: str, fold: int) -> dict:
    prefix = artifact_prefix(oof_dir, method, aspect, fold)
    path = prefix.with_name(prefix.name + "_metrics.json")
    if not path.exists():
        raise FileNotFoundError(f"Missing OOF metrics: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_step(step: float) -> None:
    if step <= 0 or step > 1:
        raise ValueError(f"step must be in (0, 1], got {step}")
    units = round(1.0 / step)
    if not np.isclose(units * step, 1.0, atol=1e-8):
        raise ValueError(f"step must divide 1.0 exactly, got {step}")


def simplex_grid(methods: Iterable[str], step: float) -> List[Dict[str, float]]:
    validate_step(step)
    method_list = list(methods)
    units = int(round(1.0 / step))

    def points(n_remaining: int, units_remaining: int):
        if n_remaining == 1:
            yield (units_remaining,)
            return
        for first in range(units_remaining + 1):
            for rest in points(n_remaining - 1, units_remaining - first):
                yield (first,) + rest

    return [
        {method: round(value * step, 10) for method, value in zip(method_list, point)}
        for point in points(len(method_list), units)
    ]


def neighborhood_grid(center: Dict[str, float], methods: List[str], step: float, radius: float) -> List[Dict[str, float]]:
    """Simplex grid points within L-inf *radius* of *center*."""
    validate_step(step)
    units = int(round(1.0 / step))
    n = len(methods)
    bounds = []
    for m in methods:
        lo = max(0, int(np.ceil((center[m] - radius) / step - 1e-9)))
        hi = min(units, int(np.floor((center[m] + radius) / step + 1e-9)))
        bounds.append((lo, hi))

    def _enum(idx: int, remaining: int):
        if idx == n - 1:
            lo, hi = bounds[idx]
            if lo <= remaining <= hi:
                yield (remaining,)
            return
        lo, hi = bounds[idx]
        for val in range(max(lo, 0), min(hi, remaining) + 1):
            yield from ((val,) + rest for rest in _enum(idx + 1, remaining - val))

    return [
        {m: round(v * step, 10) for m, v in zip(methods, point)}
        for point in _enum(0, units)
    ]


def best_fold(oof_dir: Path, method: str, aspect: str, folds: List[int]) -> int:
    best = None
    for fold in folds:
        metrics = load_metrics(oof_dir, method, aspect, fold)
        candidate = (float(metrics["fmax"]), fold)
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(f"No OOF metrics found for method={method} aspect={aspect}")
    return best[1]


def load_aspect_references(oof_dir: Path, aspect: str, folds: List[int], parents: dict) -> List[dict]:
    references = []
    classes = None
    prop_indices = None
    for fold in folds:
        reference_pids = load_array(oof_dir, METHODS[0], aspect, fold, "pids")
        reference_classes = load_array(oof_dir, METHODS[0], aspect, fold, "classes")
        labels = load_array(oof_dir, METHODS[0], aspect, fold, "labels", mmap_mode="r").astype(np.float32, copy=False)
        if classes is None:
            classes = reference_classes
            prop_indices = build_propagation_indices(classes, parents)
        elif not np.array_equal(classes, reference_classes):
            raise ValueError(f"Class mismatch on aspect={aspect} fold={fold}")
        for method in METHODS[1:]:
            pids = load_array(oof_dir, method, aspect, fold, "pids")
            method_classes = load_array(oof_dir, method, aspect, fold, "classes")
            if not np.array_equal(reference_pids, pids):
                raise ValueError(f"PID mismatch on method={method} aspect={aspect} fold={fold}")
            if not np.array_equal(reference_classes, method_classes):
                raise ValueError(f"Class mismatch on method={method} aspect={aspect} fold={fold}")
        propagated_labels = propagate_scores(labels, prop_indices).astype(bool, copy=False)
        true_per = propagated_labels.sum(axis=1)
        references.append(
            {
                "aspect": aspect,
                "fold": fold,
                "shape": labels.shape,
                "prop_indices": prop_indices,
                "labels": propagated_labels,
                "true_per": true_per,
                "has_label": true_per > 0,
            }
        )
    return references


def empty_metric_accumulator() -> dict:
    return {
        "precision_sum": np.zeros(len(FMAX_THRESHOLDS), dtype=np.float64),
        "precision_count": np.zeros(len(FMAX_THRESHOLDS), dtype=np.int64),
        "recall_sum": np.zeros(len(FMAX_THRESHOLDS), dtype=np.float64),
        "recall_count": 0,
        "default_tp": 0,
        "default_pred_pos": 0,
        "true_pos": 0,
    }


def update_metric_accumulator(accumulator: dict, labels: np.ndarray, true_per: np.ndarray, has_label: np.ndarray, probs: np.ndarray) -> None:
    n_prot, n_cls = probs.shape
    label_count = int(has_label.sum())
    accumulator["recall_count"] += label_count
    accumulator["true_pos"] += int(true_per.sum())

    # Vectorize: process multiple thresholds per chunk via broadcasting
    n_thr = len(FMAX_THRESHOLDS)
    chunk = max(1, min(n_thr, 500_000_000 // max(n_prot * n_cls, 1)))
    has_label_row = has_label[None, :]
    true_per_labeled = true_per[has_label]

    for t0 in range(0, n_thr, chunk):
        t1 = min(t0 + chunk, n_thr)
        pred = probs[None, :, :] >= FMAX_THRESHOLDS[t0:t1, None, None]  # (K, N, C)
        pred_per = pred.sum(axis=2)                                      # (K, N)
        pred &= labels[None, :, :]                                       # in-place AND
        tp_per = pred.sum(axis=2)                                        # (K, N)
        del pred

        mask = (pred_per > 0) & has_label_row
        safe_denom = np.where(pred_per > 0, pred_per, 1)
        prec_vals = tp_per / safe_denom
        accumulator["precision_sum"][t0:t1] += np.where(mask, prec_vals, 0.0).sum(axis=1)
        accumulator["precision_count"][t0:t1] += mask.sum(axis=1)

        if label_count:
            accumulator["recall_sum"][t0:t1] += (tp_per[:, has_label] / true_per_labeled[None, :]).sum(axis=1)

    pred05 = probs >= 0.5
    accumulator["default_tp"] += int(np.logical_and(pred05, labels).sum())
    accumulator["default_pred_pos"] += int(pred05.sum())


def finalize_metric_accumulator(accumulator: dict) -> Dict[str, float]:
    best_fmax = 0.0
    best_threshold = 0.5
    for index, threshold in enumerate(FMAX_THRESHOLDS):
        precision = (
            float(accumulator["precision_sum"][index] / accumulator["precision_count"][index])
            if accumulator["precision_count"][index] > 0
            else 0.0
        )
        recall = (
            float(accumulator["recall_sum"][index] / accumulator["recall_count"])
            if accumulator["recall_count"] > 0
            else 0.0
        )
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = float(threshold)
    pred_pos = accumulator["default_pred_pos"]
    true_pos = accumulator["true_pos"]
    tp = accumulator["default_tp"]
    return {
        "micro_f1": float((2 * tp) / (pred_pos + true_pos)) if pred_pos + true_pos > 0 else 0.0,
        "micro_precision": float(tp / pred_pos) if pred_pos > 0 else 0.0,
        "micro_recall": float(tp / true_pos) if true_pos > 0 else 0.0,
        "fmax": best_fmax,
        "fmax_threshold": best_threshold,
    }


def load_fold_probs(oof_dir: Path, aspect: str, fold: int) -> Dict[str, np.ndarray]:
    return {
        method: load_array(oof_dir, method, aspect, fold, "probs", mmap_mode="r")
        for method in METHODS
    }


def update_progress(progress: tqdm | None) -> None:
    if progress is not None:
        progress.update(1)


def is_mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built())


def score_matrix_bytes(shape: tuple[int, int]) -> int:
    n_prot, n_cls = shape
    return n_prot * n_cls * 4


def cpu_batch_size(n_candidates: int, shape: tuple[int, int]) -> int:
    return max(1, min(n_candidates, CPU_BATCH_BYTES // max(score_matrix_bytes(shape), 1)))


def cuda_batch_size(n_candidates: int, shape: tuple[int, int]) -> int:
    bytes_per = score_matrix_bytes(shape)
    free_bytes, _ = torch.cuda.mem_get_info()
    budget = int(free_bytes * CUDA_WORKING_BYTES_FRACTION) - len(METHODS) * bytes_per
    if budget < bytes_per:
        return 0
    return max(1, min(n_candidates, budget // max(bytes_per, 1)))


def mps_batch_size(n_candidates: int, shape: tuple[int, int]) -> int:
    return max(1, min(n_candidates, MPS_BATCH_BYTES // max(score_matrix_bytes(shape), 1)))


def select_fusion_backend(reference: dict, n_candidates: int, requested_device: str) -> dict:
    shape = reference["shape"]
    bytes_per = score_matrix_bytes(shape)
    n_cls = shape[1]
    cpu_batch = cpu_batch_size(n_candidates, shape)
    mps_batch = mps_batch_size(n_candidates, shape)
    mps_fits = (len(METHODS) + 1) * bytes_per <= MPS_BATCH_BYTES

    if requested_device == "cpu":
        return {"kind": "numpy", "device": torch.device("cpu"), "batch": cpu_batch}

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested cuda, but CUDA is not available")
        cuda_batch = cuda_batch_size(n_candidates, shape)
        if cuda_batch <= 0:
            raise RuntimeError("Requested cuda, but the estimated free GPU memory is not enough for one fusion batch")
        return {"kind": "torch", "device": torch.device("cuda"), "batch": cuda_batch}

    if requested_device == "mps":
        if not is_mps_available():
            raise RuntimeError("Requested mps, but MPS is not available")
        if not mps_fits:
            raise RuntimeError("Requested mps, but the estimated working set exceeds the configured MPS budget")
        return {"kind": "torch", "device": torch.device("mps"), "batch": mps_batch}

    if requested_device != "auto":
        raise ValueError(f"Unsupported device={requested_device}")

    if torch.cuda.is_available():
        cuda_batch = cuda_batch_size(n_candidates, shape)
        if cuda_batch > 0 and n_cls <= CUDA_PROPAGATION_CLASS_LIMIT:
            return {"kind": "torch", "device": torch.device("cuda"), "batch": cuda_batch}

    if is_mps_available() and mps_fits:
        return {"kind": "torch", "device": torch.device("mps"), "batch": mps_batch}

    return {"kind": "numpy", "device": torch.device("cpu"), "batch": cpu_batch}


def build_torch_descendant_indices(descendant_indices: List[List[int]], device: torch.device) -> List[torch.Tensor | None]:
    return [
        torch.as_tensor(children, device=device, dtype=torch.long) if children else None
        for children in descendant_indices
    ]


def propagate_scores_torch(scores: torch.Tensor, descendant_indices: List[torch.Tensor | None]) -> torch.Tensor:
    propagated = scores.clone()
    for parent_index, children in enumerate(descendant_indices):
        if children is None:
            continue
        child_max = torch.index_select(scores, 1, children).amax(dim=1)
        propagated[:, parent_index] = torch.maximum(propagated[:, parent_index], child_max)
    return propagated


def update_candidates_for_fold(
    reference: dict,
    fold_probs: Dict[str, np.ndarray],
    candidates: List[Dict[str, float]],
    accumulators: List[dict],
    backend: dict,
    *,
    progress: tqdm | None = None,
) -> None:
    # Pre-stack probs: (n_methods, N, C); build weight matrix: (n_candidates, n_methods)
    weight_matrix = np.array(
        [[w.get(m, 0.0) for m in METHODS] for w in candidates],
        dtype=np.float32,
    )
    n_cand = len(candidates)
    batch = backend["batch"]
    prop_indices = reference["prop_indices"]
    labels = reference["labels"]
    true_per = reference["true_per"]
    has_label = reference["has_label"]

    if backend["kind"] == "torch":
        probs_stack_np = np.stack([np.asarray(fold_probs[m], dtype=np.float32) for m in METHODS])
        probs_stack = torch.from_numpy(probs_stack_np).to(backend["device"])
        weight_matrix_torch = torch.from_numpy(weight_matrix).to(backend["device"])
        torch_prop_indices = backend["torch_prop_indices"]

        for c0 in range(0, n_cand, batch):
            c1 = min(c0 + batch, n_cand)
            fused_batch = torch.einsum("cm,mnk->cnk", weight_matrix_torch[c0:c1], probs_stack)
            for i in range(c1 - c0):
                propagated = propagate_scores_torch(fused_batch[i], torch_prop_indices)
                update_metric_accumulator(
                    accumulators[c0 + i],
                    labels,
                    true_per,
                    has_label,
                    propagated.detach().cpu().numpy(),
                )
                update_progress(progress)
        return

    probs_stack = np.stack([np.asarray(fold_probs[m], dtype=np.float32) for m in METHODS])
    for c0 in range(0, n_cand, batch):
        c1 = min(c0 + batch, n_cand)
        fused_batch = np.einsum("cm,mnk->cnk", weight_matrix[c0:c1], probs_stack)
        for i in range(c1 - c0):
            propagated = propagate_scores(fused_batch[i], prop_indices)
            update_metric_accumulator(accumulators[c0 + i], labels, true_per, has_label, propagated)
            update_progress(progress)


def best_candidate(candidates: List[Dict[str, float]], accumulators: List[dict]) -> dict:
    best = None
    for weights, accumulator in zip(candidates, accumulators):
        candidate = {"weights": weights, "metrics": finalize_metric_accumulator(accumulator)}
        if best is None or candidate["metrics"]["fmax"] > best["metrics"]["fmax"]:
            best = candidate
    if best is None:
        raise RuntimeError("No fusion candidate was evaluated")
    return best


def top_candidates(candidates: List[Dict[str, float]], accumulators: List[dict], k: int) -> List[Dict[str, float]]:
    scored = [
        (finalize_metric_accumulator(acc)["fmax"], w)
        for w, acc in zip(candidates, accumulators)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [w for _, w in scored[:k]]


def neighborhood_union(centers: List[Dict[str, float]], methods: List[str], step: float, radius: float) -> List[Dict[str, float]]:
    seen: set = set()
    result: List[Dict[str, float]] = []
    for center in centers:
        for candidate in neighborhood_grid(center, methods, step, radius):
            key = tuple(candidate[m] for m in methods)
            if key not in seen:
                seen.add(key)
                result.append(candidate)
    return result


def _eval_grid(
    references: List[dict],
    oof_dir: Path,
    aspect: str,
    candidates: List[Dict[str, float]],
    *,
    desc: str,
    position: int,
    requested_device: str,
) -> List[dict]:
    accumulators = [empty_metric_accumulator() for _ in candidates]
    backend = select_fusion_backend(references[0], len(candidates), requested_device)
    if backend["kind"] == "torch":
        backend["torch_prop_indices"] = build_torch_descendant_indices(references[0]["prop_indices"], backend["device"])
    print(
        f"{desc}: device={backend['device'].type} batch={backend['batch']} "
        f"shape={references[0]['shape'][0]}x{references[0]['shape'][1]}"
    )
    progress = tqdm(total=len(candidates) * len(references), desc=desc, dynamic_ncols=True, position=position)
    try:
        for ref in references:
            fold_probs = load_fold_probs(oof_dir, aspect, ref["fold"])
            update_candidates_for_fold(ref, fold_probs, candidates, accumulators, backend, progress=progress)
            del fold_probs
    finally:
        progress.close()
    return accumulators


def search_aspect(
    oof_dir: Path,
    aspect: str,
    folds: List[int],
    step: float,
    parents: dict,
    *,
    position: int = 0,
    requested_device: str = "auto",
) -> dict:
    references = load_aspect_references(oof_dir, aspect, folds, parents)

    coarse_step = step * 2
    try:
        validate_step(coarse_step)
        use_two_stage = True
    except ValueError:
        use_two_stage = False

    if use_two_stage:
        # Stage 1: coarse grid
        coarse_cands = simplex_grid(METHODS, coarse_step)
        coarse_accs = _eval_grid(
            references, oof_dir, aspect, coarse_cands, desc=f"{aspect} coarse", position=position, requested_device=requested_device
        )

        # Stage 2: refine around top 2
        tops = top_candidates(coarse_cands, coarse_accs, k=2)
        fine_cands = neighborhood_union(tops, list(METHODS), step, radius=coarse_step)
        fine_accs = _eval_grid(
            references, oof_dir, aspect, fine_cands, desc=f"{aspect} refine", position=position, requested_device=requested_device
        )
        best = best_candidate(fine_cands, fine_accs)
    else:
        candidates = simplex_grid(METHODS, step)
        accs = _eval_grid(
            references, oof_dir, aspect, candidates, desc=f"{aspect} fusion", position=position, requested_device=requested_device
        )
        best = best_candidate(candidates, accs)

    return best


def search_aspect_worker(
    oof_dir: Path,
    aspect: str,
    folds: List[int],
    step: float,
    obo_path: Path,
    position: int,
    requested_device: str,
) -> tuple[str, dict]:
    parents = parse_go_obo(obo_path)
    return aspect, search_aspect(oof_dir, aspect, folds, step, parents, position=position, requested_device=requested_device)


def resolve_jobs(requested_jobs: int, aspect_count: int) -> int:
    if aspect_count <= 1:
        return 1
    if requested_jobs > 0:
        return max(1, min(requested_jobs, aspect_count))
    cpu_count = os.cpu_count() or 1
    return max(1, min(aspect_count, cpu_count))


def run_aspects_parallel(args: argparse.Namespace, jobs: int) -> Dict[str, dict]:
    executor_cls = ProcessPoolExecutor
    results: Dict[str, dict] = {}

    def collect_results(executor) -> None:
        futures = {
            executor.submit(search_aspect_worker, args.oof_dir, aspect, args.fold, args.step, args.obo, index, args.device): aspect
            for index, aspect in enumerate(args.aspect)
        }
        for future in as_completed(futures):
            aspect, best = future.result()
            results[aspect] = best

    try:
        with executor_cls(max_workers=jobs) as executor:
            collect_results(executor)
    except PermissionError:
        print("Process pool unavailable; falling back to threads.")
        results.clear()
        with ThreadPoolExecutor(max_workers=jobs) as executor:
            collect_results(executor)
    return results


def run_aspects_sequential(args: argparse.Namespace) -> Dict[str, dict]:
    parents = parse_go_obo(args.obo)
    results = {}
    for index, aspect in enumerate(args.aspect):
        results[aspect] = search_aspect(
            args.oof_dir, aspect, args.fold, args.step, parents, position=index, requested_device=args.device
        )
    return results


def build_output(args: argparse.Namespace, results: Dict[str, dict]) -> tuple[List[dict], Dict[str, dict]]:
    rows = []
    summary = {}
    for aspect in args.aspect:
        best = results[aspect]
        weights = best["weights"]
        metrics = best["metrics"]
        row = {"aspect": aspect, "thr": round(metrics["fmax_threshold"], 2)}
        for method in METHODS:
            row[f"w_{METHOD_COLUMNS[method]}"] = round(weights[method], 2)
        for method in METHODS:
            row[f"fold_{METHOD_COLUMNS[method]}"] = best_fold(args.oof_dir, method, aspect, args.fold)
        rows.append(row)
        summary[aspect] = {
            "weights": weights,
            "metrics": metrics,
            "folds": {method: row[f"fold_{METHOD_COLUMNS[method]}"] for method in METHODS},
        }
        print(f"aspect={aspect} best_weights={weights} fmax={metrics['fmax']:.4f} threshold={metrics['fmax_threshold']:.2f}")
    return rows, summary


def main() -> None:
    args = parse_args()
    validate_step(args.step)
    jobs = resolve_jobs(args.jobs, len(args.aspect))
    if jobs > 1:
        results = run_aspects_parallel(args, jobs)
    else:
        results = run_aspects_sequential(args)
    rows, summary = build_output(args, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    with args.output.with_name(args.output.stem + "_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Saved fusion weights to {args.output}")


if __name__ == "__main__":
    main()
