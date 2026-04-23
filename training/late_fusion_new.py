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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search OOF late-fusion weights by fused FMAX")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--step", type=float, default=0.1)
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
    label_count = int(has_label.sum())
    accumulator["recall_count"] += label_count
    accumulator["true_pos"] += int(true_per.sum())
    for index, threshold in enumerate(FMAX_THRESHOLDS):
        pred = probs >= threshold
        pred_per = pred.sum(axis=1)
        tp_per = np.logical_and(pred, labels).sum(axis=1)
        has_pred_and_label = (pred_per > 0) & has_label
        if has_pred_and_label.any():
            accumulator["precision_sum"][index] += float((tp_per[has_pred_and_label] / pred_per[has_pred_and_label]).sum())
            accumulator["precision_count"][index] += int(has_pred_and_label.sum())
        if label_count:
            accumulator["recall_sum"][index] += float((tp_per[has_label] / true_per[has_label]).sum())
    pred = probs >= 0.5
    accumulator["default_tp"] += int(np.logical_and(pred, labels).sum())
    accumulator["default_pred_pos"] += int(pred.sum())


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


def update_candidates_for_fold(
    reference: dict,
    fold_probs: Dict[str, np.ndarray],
    candidates: List[Dict[str, float]],
    accumulators: List[dict],
    *,
    progress: tqdm | None = None,
) -> None:
    for weights, accumulator in zip(candidates, accumulators):
        fused = np.zeros(reference["shape"], dtype=np.float32)
        for method, weight in weights.items():
            if weight == 0:
                continue
            fused += np.float32(weight) * fold_probs[method]
        propagated_probs = propagate_scores(fused, reference["prop_indices"])
        update_metric_accumulator(
            accumulator,
            reference["labels"],
            reference["true_per"],
            reference["has_label"],
            propagated_probs,
        )
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


def search_aspect(
    oof_dir: Path,
    aspect: str,
    folds: List[int],
    step: float,
    parents: dict,
    *,
    position: int = 0,
) -> dict:
    references = load_aspect_references(oof_dir, aspect, folds, parents)
    candidates = simplex_grid(METHODS, step)
    accumulators = [empty_metric_accumulator() for _ in candidates]
    progress = tqdm(total=len(candidates) * len(references), desc=f"{aspect} fusion", dynamic_ncols=True, position=position)
    try:
        for reference in references:
            fold_probs = load_fold_probs(oof_dir, aspect, reference["fold"])
            update_candidates_for_fold(
                reference,
                fold_probs,
                candidates,
                accumulators,
                progress=progress,
            )
            del fold_probs
        best = best_candidate(candidates, accumulators)
        if progress is not None:
            progress.set_postfix(best=f"{best['metrics']['fmax']:.4f}")
    finally:
        if progress is not None:
            progress.close()
    return best


def search_aspect_worker(oof_dir: Path, aspect: str, folds: List[int], step: float, obo_path: Path, position: int) -> tuple[str, dict]:
    parents = parse_go_obo(obo_path)
    return aspect, search_aspect(oof_dir, aspect, folds, step, parents, position=position)


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
            executor.submit(search_aspect_worker, args.oof_dir, aspect, args.fold, args.step, args.obo, index): aspect
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
        results[aspect] = search_aspect(args.oof_dir, aspect, args.fold, args.step, parents, position=index)
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
