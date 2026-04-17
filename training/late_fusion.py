from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores
from training.trainer import compute_multilabel_metrics


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
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--threshold-start", type=float, default=0.1)
    parser.add_argument("--threshold-stop", type=float, default=0.9)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--save-fused-oof", action="store_true")
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH,
                        help="Path to go-basic.obo for propagation-aware Fmax (default: data/go-basic.obo)")
    return parser.parse_args()


def load_fold_artifact(oof_dir: Path, method: str, aspect: str, fold: int) -> dict:
    prefix = oof_dir / f"{method}_{aspect}_fold{fold}"
    required = {
        "pids": prefix.with_name(prefix.name + "_pids.npy"),
        "labels": prefix.with_name(prefix.name + "_labels.npy"),
        "probs": prefix.with_name(prefix.name + "_probs.npy"),
        "classes": prefix.with_name(prefix.name + "_classes.npy"),
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        missing_list = ", ".join(missing)
        raise FileNotFoundError(f"Missing OOF artifacts for {method} {aspect} fold {fold}: {missing_list}")

    return {
        "pids": np.load(required["pids"], allow_pickle=True),
        "labels": np.load(required["labels"], allow_pickle=True),
        "probs": np.load(required["probs"], allow_pickle=True),
        "classes": np.load(required["classes"], allow_pickle=True),
    }


def align_artifact(reference: dict, candidate: dict, method: str, aspect: str, fold: int) -> np.ndarray:
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


def collect_oof_predictions(oof_dir: Path, methods: List[str], aspect: str, folds: List[int]) -> dict:
    all_pids: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_probs: Dict[str, List[np.ndarray]] = {method: [] for method in methods}
    classes = None

    for fold in folds:
        reference = load_fold_artifact(oof_dir, methods[0], aspect, fold)
        if classes is None:
            classes = reference["classes"]
        elif not np.array_equal(classes, reference["classes"]):
            raise ValueError(f"Reference class mismatch across folds for aspect {aspect}")

        all_pids.append(reference["pids"])
        all_labels.append(reference["labels"])
        all_probs[methods[0]].append(reference["probs"])

        for method in methods[1:]:
            candidate = load_fold_artifact(oof_dir, method, aspect, fold)
            aligned_probs = align_artifact(reference, candidate, method, aspect, fold)
            all_probs[method].append(aligned_probs)

    stacked = {
        "pids": np.concatenate(all_pids, axis=0),
        "labels": np.concatenate(all_labels, axis=0),
        "classes": classes,
        "probs": {method: np.concatenate(prob_list, axis=0) for method, prob_list in all_probs.items()},
    }
    return stacked


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


def fuse_probabilities(probs_by_method: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    first = next(iter(probs_by_method.values()))
    fused = np.zeros_like(first, dtype=np.float32)
    for method, probs in probs_by_method.items():
        fused += float(weights[method]) * probs
    return fused


def search_best_fusion(
    payload: dict,
    methods: List[str],
    thresholds: List[float],
    weight_step: float,
    prop_indices: List[List[int]],
    propagated_labels: np.ndarray,
) -> dict:
    """
    Grid-search fusion weights optimising propagation-aware Fmax.

    Labels and predictions are propagated upward through the GO DAG before
    each Fmax evaluation so that the objective matches the CAFA evaluation
    convention: predicting a specific GO term implicitly predicts all ancestors.

    ``prop_indices`` and ``propagated_labels`` are precomputed per aspect in
    ``main()`` so propagation cost is paid once per weight combination, not
    once per (weight, threshold) pair.
    """
    best = None

    for weights in generate_weight_grid(methods, weight_step):
        fused_probs = fuse_probabilities(payload["probs"], weights)
        propagated_probs = propagate_scores(fused_probs, prop_indices)
        for threshold in thresholds:
            metrics = compute_multilabel_metrics(propagated_labels, propagated_probs, threshold=threshold)
            candidate = {
                "weights": weights,
                "threshold": float(threshold),
                "metrics": metrics,
                "probs": fused_probs,  # store un-propagated; propagation is for eval only
            }
            if best is None:
                best = candidate
                continue

            current_key = (
                metrics["micro_f1"],
                metrics["micro_precision"],
                metrics["micro_recall"],
            )
            best_key = (
                best["metrics"]["micro_f1"],
                best["metrics"]["micro_precision"],
                best["metrics"]["micro_recall"],
            )
            if current_key > best_key:
                best = candidate

    if best is None:
        raise RuntimeError("No fusion candidates were evaluated")
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


def save_fused_oof(output_dir: Path, aspect: str, payload: dict, best: dict) -> None:
    prefix = output_dir / f"late_fusion_{aspect}"
    np.save(prefix.with_name(prefix.name + "_pids.npy"), payload["pids"])
    np.save(prefix.with_name(prefix.name + "_labels.npy"), payload["labels"])
    np.save(prefix.with_name(prefix.name + "_classes.npy"), payload["classes"])
    np.save(prefix.with_name(prefix.name + "_probs.npy"), best["probs"])


def main() -> None:
    args = parse_args()
    args.oof_dir = Path(args.oof_dir)
    args.output = Path(args.output)
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
        payload = collect_oof_predictions(args.oof_dir, args.methods, aspect, args.fold)

        # Precompute GO propagation indices for this aspect's class set (done once per aspect)
        print(f"  Building propagation indices for {len(payload['classes'])} classes ...")
        prop_indices = build_propagation_indices(payload["classes"], go_parents)

        # Propagate labels once; fused probs are propagated per weight combo inside search
        propagated_labels = propagate_scores(payload["labels"].astype(np.float32), prop_indices)

        best = search_best_fusion(
            payload,
            args.methods,
            thresholds,
            args.weight_step,
            prop_indices,
            propagated_labels,
        )

        rows.append(format_weight_row(aspect, best))
        summary[aspect] = {
            "weights": best["weights"],
            "threshold": best["threshold"],
            "metrics": best["metrics"],
            "num_proteins": int(payload["labels"].shape[0]),
            "num_classes": int(payload["labels"].shape[1]),
        }
        if args.save_fused_oof:
            save_fused_oof(args.oof_dir, aspect, payload, best)

        print(
            f"  best weights={best['weights']} threshold={best['threshold']:.2f} "
            f"propagated_fmax={best['metrics']['fmax']:.4f} "
            f"micro_f1={best['metrics']['micro_f1']:.4f}"
        )

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
