from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

from training.data.go_utils import build_propagation_indices, parse_go_obo, propagate_scores
from training.trainer import compute_multilabel_metrics

DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"
DEFAULT_OUTPUT = ROOT_DIR / "models" / "fusion_weights.csv"
DEFAULT_OBO_PATH = ROOT_DIR / "data" / "go-basic.obo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search two-stage OOF fusion weights")
    parser.add_argument("--aspect", nargs="+", default=["P", "F", "C"], choices=["P", "F", "C"])
    parser.add_argument("--methods", nargs="+", default=["esm2_last", "esm2_l20", "prott5", "blast"])
    parser.add_argument("--fold", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--weight-step", "--alpha-step", dest="weight_step", type=float, default=0.05)
    parser.add_argument("--cores", type=int, default=1, help="Accepted for README compatibility; not used.")
    return parser.parse_args()


def artifact_prefix(oof_dir: Path, method: str, aspect: str, fold: int) -> Path:
    aliases = {"esm2": "esm2_last", "t5": "prott5", "cnn": "esm2_l20"}
    method = aliases.get(method, method)
    return oof_dir / f"{method}_{aspect}_fold_{fold}"


def load_artifact(oof_dir: Path, method: str, aspect: str, fold: int) -> dict:
    prefix = artifact_prefix(oof_dir, method, aspect, fold)
    paths = {
        "pids": prefix.with_name(prefix.name + "_pids.npy"),
        "labels": prefix.with_name(prefix.name + "_labels.npy"),
        "probs": prefix.with_name(prefix.name + "_probs.npy"),
        "classes": prefix.with_name(prefix.name + "_classes.npy"),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing OOF artifacts: {missing}")
    return {key: np.load(path, allow_pickle=True) for key, path in paths.items()}


def align_to_reference(reference: dict, candidate: dict) -> np.ndarray:
    pid_to_index = {pid: index for index, pid in enumerate(candidate["pids"].tolist())}
    class_to_index = {term: index for index, term in enumerate(candidate["classes"].tolist())}
    rows = [pid_to_index[pid] for pid in reference["pids"].tolist()]
    aligned = np.zeros_like(reference["probs"], dtype=np.float32)
    for out_col, term in enumerate(reference["classes"].tolist()):
        in_col = class_to_index.get(term)
        if in_col is not None:
            aligned[:, out_col] = candidate["probs"][rows, in_col]
    return aligned


def collect_oof(oof_dir: Path, aspect: str, folds: List[int], methods: List[str]) -> dict:
    pids = []
    labels = []
    probs = {method: [] for method in methods}
    classes = None
    for fold in folds:
        reference = load_artifact(oof_dir, methods[0], aspect, fold)
        if classes is None:
            classes = reference["classes"]
        elif not np.array_equal(classes, reference["classes"]):
            raise ValueError(f"Class mismatch on fold {fold}")
        pids.append(reference["pids"])
        labels.append(reference["labels"])
        probs[methods[0]].append(reference["probs"].astype(np.float32))
        for method in methods[1:]:
            probs[method].append(align_to_reference(reference, load_artifact(oof_dir, method, aspect, fold)))
    return {
        "pids": np.concatenate(pids),
        "labels": np.concatenate(labels),
        "probs": {method: np.concatenate(values).astype(np.float32) for method, values in probs.items()},
        "classes": classes,
    }


def alpha_grid(step: float) -> List[float]:
    count = int(round(1.0 / step))
    return [round(index * step, 10) for index in range(count + 1)]


def simplex_grid(methods: List[str], step: float) -> List[Dict[str, float]]:
    units = int(round(1.0 / step))

    def points(n_remaining: int, units_remaining: int):
        if n_remaining == 1:
            yield (units_remaining,)
            return
        for first in range(units_remaining + 1):
            for rest in points(n_remaining - 1, units_remaining - first):
                yield (first,) + rest

    return [
        {method: round(value * step, 10) for method, value in zip(methods, point)}
        for point in points(len(methods), units)
    ]


def weighted_sum(probs_by_method: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    result = None
    for method, weight in weights.items():
        if weight == 0:
            continue
        contribution = float(weight) * probs_by_method[method]
        result = contribution.copy() if result is None else result + contribution
    if result is None:
        first = next(iter(probs_by_method.values()))
        return np.zeros_like(first, dtype=np.float32)
    return result.astype(np.float32)


def search_two_stage_fusion(payload: dict, parents: dict, step: float) -> dict:
    prop_indices = build_propagation_indices(payload["classes"], parents)
    labels = propagate_scores(payload["labels"].astype(np.float32), prop_indices)
    chain_methods = ["esm2_last", "esm2_l20", "prott5"]
    best_neural = None
    for weights in simplex_grid(chain_methods, step):
        fused = weighted_sum(payload["probs"], weights)
        fused = propagate_scores(fused, prop_indices)
        metrics = compute_multilabel_metrics(labels, fused)
        candidate = {"weights": weights, "metrics": metrics, "probs": fused}
        if best_neural is None or metrics["fmax"] > best_neural["metrics"]["fmax"]:
            best_neural = candidate
    if best_neural is None:
        raise RuntimeError("No neural fusion candidate was evaluated")
    best_final = None
    for beta in alpha_grid(step):
        fused = beta * best_neural["probs"] + (1.0 - beta) * payload["probs"]["blast"]
        fused = propagate_scores(fused, prop_indices)
        metrics = compute_multilabel_metrics(labels, fused)
        candidate = {"beta": beta, "metrics": metrics}
        if best_final is None or metrics["fmax"] > best_final["metrics"]["fmax"]:
            best_final = candidate
    return {"neural": best_neural, "final": best_final}


def main() -> None:
    args = parse_args()
    parents = parse_go_obo(args.obo)
    rows = []
    summary: Dict[str, dict] = {}
    aliases = {"esm2": "esm2_last", "t5": "prott5", "cnn": "esm2_l20"}
    methods = [aliases.get(method, method) for method in args.methods]
    required = {"esm2_last", "esm2_l20", "prott5", "blast"}
    if set(methods) != required:
        raise ValueError(f"Two-stage fusion requires methods {sorted(required)}, got {methods}")
    for aspect in args.aspect:
        payload = collect_oof(args.oof_dir, aspect, args.fold, methods)
        best = search_two_stage_fusion(payload, parents, args.weight_step)
        chain_weights = best["neural"]["weights"]
        final = best["final"]
        rows.append(
            {
                "Aspect": aspect,
                "W_ESM2_LAST": chain_weights["esm2_last"],
                "W_ESM2_L20": chain_weights["esm2_l20"],
                "W_PROTT5": chain_weights["prott5"],
                "BETA_NEURAL": final["beta"],
                "W_BLAST": 1.0 - final["beta"],
                "THRESHOLD": final["metrics"]["fmax_threshold"],
                "FMAX": final["metrics"]["fmax"],
                "MICRO_F1": final["metrics"]["micro_f1"],
            }
        )
        summary[aspect] = {
            "neural_weights": chain_weights,
            "neural_metrics": best["neural"]["metrics"],
            "beta": final["beta"],
            "final_metrics": final["metrics"],
        }
        print(f"aspect={aspect} chain={chain_weights} beta={final['beta']:.2f} fmax={final['metrics']['fmax']:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    with args.output.with_name(args.output.stem + "_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"Saved fusion weights to {args.output}")


if __name__ == "__main__":
    main()
