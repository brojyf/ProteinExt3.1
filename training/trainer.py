from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochResult:
    loss: float


def move_batch_to_device(batch_inputs: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=device.type == "cuda") if isinstance(value, torch.Tensor) else value
        for key, value in batch_inputs.items()
    }


def train_one_epoch(
    model,
    loader: DataLoader,
    optimizer,
    device: torch.device,
    progress_desc: str,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_examples = 0
    for _, batch_inputs, labels in tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True):
        batch_inputs = move_batch_to_device(batch_inputs, device)
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch_inputs)
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        if scaler is None:
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        total_loss += float(loss.item()) * labels.size(0)
        total_examples += labels.size(0)
    return EpochResult(loss=total_loss / max(total_examples, 1))


@torch.inference_mode()
def predict(
    model,
    loader: DataLoader,
    device: torch.device,
    progress_desc: str,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_pids: List[str] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    for batch in tqdm(loader, desc=progress_desc, leave=False, dynamic_ncols=True):
        if len(batch) == 3:
            pids, batch_inputs, labels = batch
            all_labels.append(labels.numpy())
        else:
            pids, batch_inputs = batch
        batch_inputs = move_batch_to_device(batch_inputs, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(batch_inputs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_pids.extend(pids)
        all_probs.append(probs)
    return {
        "pids": np.asarray(all_pids),
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0), dtype=np.float32),
        "labels": np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, 0), dtype=np.float32),
    }


def compute_multilabel_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    if y_true.size == 0 or y_prob.size == 0:
        return {"micro_f1": 0.0, "micro_precision": 0.0, "micro_recall": 0.0, "fmax": 0.0, "fmax_threshold": threshold}
    y_true_bool = y_true.astype(bool)
    best_fmax = 0.0
    best_threshold = threshold
    for candidate in np.linspace(0.01, 0.99, 99):
        pred = y_prob >= candidate
        true_per = y_true_bool.sum(axis=1)
        pred_per = pred.sum(axis=1)
        tp_per = np.logical_and(pred, y_true_bool).sum(axis=1)
        has_label = true_per > 0
        has_pred_and_label = (pred_per > 0) & has_label
        precision = float((tp_per[has_pred_and_label] / pred_per[has_pred_and_label]).mean()) if has_pred_and_label.any() else 0.0
        recall = float((tp_per[has_label] / true_per[has_label]).mean()) if has_label.any() else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
        if f1 > best_fmax:
            best_fmax = f1
            best_threshold = float(candidate)
    pred = y_prob >= threshold
    tp = int(np.logical_and(pred, y_true_bool).sum())
    pred_pos = int(pred.sum())
    true_pos = int(y_true_bool.sum())
    return {
        "micro_f1": float((2 * tp) / (pred_pos + true_pos)) if pred_pos + true_pos > 0 else 0.0,
        "micro_precision": float(tp / pred_pos) if pred_pos > 0 else 0.0,
        "micro_recall": float(tp / true_pos) if true_pos > 0 else 0.0,
        "fmax": best_fmax,
        "fmax_threshold": best_threshold,
    }
