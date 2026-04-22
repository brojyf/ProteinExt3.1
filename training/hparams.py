from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


COMMON_TRAINING_CONFIG: Dict[str, object] = {
    "method": "esm2_last",
    "aspect": "P",
    "fold": [0, 1, 2, 3, 4],
    "epochs": 20,
    "batch_size": 16,
    "num_workers": 0,
    "threshold": 0.5,
    "min_count": 20,
    "device": "auto",
    "lr": 3e-4,
    "lr_factor": 0.5,
    "lr_patience": 2,
    "min_lr": 5e-5,
    "lr_scheduler": "plateau",
    "early_stop_patience": 6,
    "early_stop_min_delta": 1e-4,
    "weight_decay": 2e-4,
    "hidden_dim": 2048,
    "bottleneck": 1024,
    "dropout": 0.3,
    "esm2_embedding_dim": 1280,
    "t5_embedding_dim": 1024,
    "blast_top_k": 30,
    "blast_tau": 50.0,
}

COSINE_TRAINING_CONFIG: Dict[str, object] = {
    "lr_scheduler": "cosine",
}

TRAINING_RUNS: List[Dict[str, object]] = [
    {
        "method": "esm2_last",
        "aspect": "P",
        "epochs": 24,
        "min_count": 30,
        "lr": 2e-4,
        "weight_decay": 3e-4,
        "hidden_dim": 2048,
        "bottleneck": 1024,
        "dropout": 0.35,
    },
    {
        "method": "esm2_last",
        "aspect": "F",
        "epochs": 22,
        "min_count": 15,
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 2048,
        "bottleneck": 1024,
        "dropout": 0.3,
    },
    {
        "method": "esm2_last",
        "aspect": "C",
        "epochs": 18,
        "min_count": 10,
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 1536,
        "bottleneck": 768,
        "dropout": 0.25,
    },
    {
        "method": "esm2_l20",
        "aspect": "P",
        "epochs": 22,
        "min_count": 30,
        "lr": 3e-4,
        "weight_decay": 3e-4,
        "hidden_dim": 1536,
        "bottleneck": 768,
        "dropout": 0.35,
    },
    {
        "method": "esm2_l20",
        "aspect": "F",
        "epochs": 20,
        "min_count": 15,
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 1536,
        "bottleneck": 768,
        "dropout": 0.3,
    },
    {
        "method": "esm2_l20",
        "aspect": "C",
        "epochs": 18,
        "min_count": 10,
        "lr": 3e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 1024,
        "bottleneck": 512,
        "dropout": 0.25,
    },
    {
        "method": "prott5",
        "aspect": "P",
        "epochs": 24,
        "min_count": 30,
        "lr": 2e-4,
        "weight_decay": 3e-4,
        "hidden_dim": 2048,
        "bottleneck": 1024,
        "dropout": 0.4,
    },
    {
        "method": "prott5",
        "aspect": "F",
        "epochs": 22,
        "min_count": 15,
        "lr": 2.5e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 1536,
        "bottleneck": 768,
        "dropout": 0.35,
    },
    {
        "method": "prott5",
        "aspect": "C",
        "epochs": 18,
        "min_count": 10,
        "lr": 2.5e-4,
        "weight_decay": 2e-4,
        "hidden_dim": 1024,
        "bottleneck": 512,
        "dropout": 0.3,
    },
    {"method": "blast", "aspect": "P", "min_count": 30, "blast_top_k": 50, "blast_tau": 75.0},
    {"method": "blast", "aspect": "F", "min_count": 15, "blast_top_k": 30, "blast_tau": 40.0},
    {"method": "blast", "aspect": "C", "min_count": 10, "blast_top_k": 40, "blast_tau": 65.0},
]


def resolve_training_run(run_config: Dict[str, object], use_cosine_lr: bool = False) -> Dict[str, object]:
    resolved = deepcopy(COMMON_TRAINING_CONFIG)
    if use_cosine_lr:
        resolved.update(deepcopy(COSINE_TRAINING_CONFIG))
    resolved.update(deepcopy(run_config))
    aliases = {"esm2": "esm2_last", "t5": "prott5"}
    resolved["method"] = aliases.get(str(resolved["method"]), resolved["method"])
    return resolved


def get_training_runs(use_cosine_lr: bool = False) -> List[Dict[str, object]]:
    return [resolve_training_run(run, use_cosine_lr) for run in TRAINING_RUNS if bool(run.get("enabled", True))]


def resolve_matching_training_run(method: str, aspect: str, use_cosine_lr: bool = False) -> Dict[str, object]:
    target = resolve_training_run({"method": method, "aspect": aspect}, use_cosine_lr)
    for run in get_training_runs(use_cosine_lr):
        if run["method"] == target["method"] and run["aspect"] == target["aspect"]:
            return run
    return target
