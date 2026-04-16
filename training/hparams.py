from __future__ import annotations

from copy import deepcopy
from typing import Dict, List


COMMON_TRAINING_CONFIG: Dict[str, object] = {
    "fold": [0, 1, 2, 3, 4],
    "epochs": 20,
    "batch_size": 16,
    "weight_decay": 2e-4,
    "num_workers": 4,
    "threshold": 0.5,
    "save_fold_artifacts": False,
    "device": "auto",
    "go_term_loss_weight": 0.25,
    "esm2_embedding_dim": 1280,
    "t5_embedding_dim": 1024,
    # --- Performance enhancements ---
    "use_amp": True,                     # Mixed precision (FP16) on CUDA
    "gradient_accumulation_steps": 2,    # Effective batch_size = batch_size * 2
    "focal_gamma": 2.0,                 # Focal loss γ; 0 = standard BCE
    "label_smoothing": 0.05,            # Shift hard 0/1 labels toward 0.5
    "max_grad_norm": 1.0,               # Gradient clipping; 0 = disabled
    "scheduler": {
        "warmup_ratio": 0.08,
        "min_lr_ratio": 0.1,
    },
}


METHOD_HPARAMS: Dict[str, Dict[str, object]] = {
    "esm2": {
        "pooling": "attention",
        "hidden_dim": 768,
        "dropout": 0.35,
        "window_size": 20,
        "go_term_loss_weight": 0.30,
        "optimizer": {
            "attention_lr": 2e-4,
            "classifier_lr": 6e-4,
        },
    },
    "t5": {
        "pooling": "attention",
        "hidden_dim": 768,
        "dropout": 0.4,
        "window_size": 20,
        "go_term_loss_weight": 0.20,
        "optimizer": {
            "attention_lr": 1.5e-4,
            "classifier_lr": 5e-4,
        },
    },
    "cnn": {
        "pooling": "mean",
        "hidden_dim": 768,
        "cnn_hidden_dim": 768,
        "dropout": 0.3,
        "window_size": 25,
        "go_term_loss_weight": 0.15,
        "optimizer": {
            "lr": 6e-4,
        },
    },
    "blast": {},  # similarity-based; no trainable parameters
}


TRAINING_RUNS: List[Dict[str, object]] = [
    {"method": "esm2", "aspect": "P"},
    {"method": "esm2", "aspect": "F"},
    {"method": "esm2", "aspect": "C"},
    {"method": "t5", "aspect": "P"},
    {"method": "t5", "aspect": "F"},
    {"method": "t5", "aspect": "C"},
    {"method": "cnn", "aspect": "P"},
    {"method": "cnn", "aspect": "F"},
    {"method": "cnn", "aspect": "C"},
    {"method": "blast", "aspect": "P"},
    {"method": "blast", "aspect": "F"},
    {"method": "blast", "aspect": "C"},
]


def _merge_dicts(base: Dict[str, object], override: Dict[str, object]) -> Dict[str, object]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_training_run(run_config: Dict[str, object]) -> Dict[str, object]:
    method = str(run_config["method"])
    if method not in METHOD_HPARAMS:
        raise ValueError(f"Unsupported training method in hparams: {method}")

    resolved = _merge_dicts(COMMON_TRAINING_CONFIG, METHOD_HPARAMS[method])
    resolved = _merge_dicts(resolved, run_config)
    resolved.setdefault("hidden_dim", 1024)
    resolved.setdefault("cnn_hidden_dim", resolved["hidden_dim"])
    return resolved


def get_training_runs() -> List[Dict[str, object]]:
    return [
        resolve_training_run(run_config)
        for run_config in TRAINING_RUNS
        if bool(run_config.get("enabled", True))
    ]
