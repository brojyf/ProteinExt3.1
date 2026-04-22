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
    "weight_decay": 2e-4,
    "hidden_dim": 2048,
    "bottleneck": 1024,
    "dropout": 0.3,
    "esm2_embedding_dim": 1280,
    "t5_embedding_dim": 1024,
    "blast_top_k": 30,
    "blast_tau": 50.0,
}

TRAINING_RUNS: List[Dict[str, object]] = [
    {"method": "esm2_last", "aspect": "P"},
    {"method": "esm2_last", "aspect": "F"},
    {"method": "esm2_last", "aspect": "C"},
    {"method": "esm2_l20", "aspect": "P"},
    {"method": "esm2_l20", "aspect": "F"},
    {"method": "esm2_l20", "aspect": "C"},
    {"method": "prott5", "aspect": "P"},
    {"method": "prott5", "aspect": "F"},
    {"method": "prott5", "aspect": "C"},
    {"method": "blast", "aspect": "P"},
    {"method": "blast", "aspect": "F"},
    {"method": "blast", "aspect": "C"},
]


def resolve_training_run(run_config: Dict[str, object]) -> Dict[str, object]:
    resolved = deepcopy(COMMON_TRAINING_CONFIG)
    resolved.update(deepcopy(run_config))
    aliases = {"esm2": "esm2_last", "t5": "prott5", "cnn": "esm2_l20"}
    resolved["method"] = aliases.get(str(resolved["method"]), resolved["method"])
    return resolved


def get_training_runs() -> List[Dict[str, object]]:
    return [resolve_training_run(run) for run in TRAINING_RUNS if bool(run.get("enabled", True))]
