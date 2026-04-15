from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


@dataclass
class EpochResult:
    loss: float
    bce_loss: float
    go_term_loss: float
    go_term_soft_f1: float


def move_batch_to_device(batch_inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in batch_inputs.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def focal_bce_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma: float = 2.0,
) -> torch.Tensor:
    """
    Focal loss for multi-label classification (numerically stable).

    Reduces the contribution of easy-to-classify negatives so the model
    focuses on hard positives — critical for GO prediction where the vast
    majority of labels are negative.
    """
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * labels + (1.0 - p) * (1.0 - labels)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    go_term_loss_weight: float = 0.0,
    progress_desc: str | None = None,
    scaler: torch.amp.GradScaler | None = None,
    gradient_accumulation_steps: int = 1,
    focal_gamma: float = 0.0,
    label_smoothing: float = 0.0,
    max_grad_norm: float = 0.0,
) -> EpochResult:
    model.train()
    use_amp = scaler is not None
    total_loss = 0.0
    total_bce_loss = 0.0
    total_go_term_loss = 0.0
    total_go_term_soft_f1 = 0.0
    total_examples = 0

    progress = tqdm(
        loader,
        desc=progress_desc or "Training",
        leave=False,
        dynamic_ncols=True,
    )

    optimizer.zero_grad(set_to_none=True)

    for step, (_, batch_inputs, labels) in enumerate(progress):
        batch_inputs = move_batch_to_device(batch_inputs, device)
        labels = labels.to(device)

        # Label smoothing: shift hard 0/1 targets toward 0.5
        if label_smoothing > 0:
            labels = labels * (1.0 - label_smoothing) + label_smoothing * 0.5

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_inputs)

            if focal_gamma > 0:
                bce_loss = focal_bce_with_logits(logits, labels, gamma=focal_gamma)
            else:
                bce_loss = F.binary_cross_entropy_with_logits(logits, labels)

            go_term_loss, go_term_soft_f1 = compute_go_term_soft_f1_loss(logits, labels)
            loss = bce_loss + float(go_term_loss_weight) * go_term_loss
            loss = loss / gradient_accumulation_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Step optimizer every N accumulation steps or at the end of the epoch
        if (step + 1) % gradient_accumulation_steps == 0 or (step + 1) == len(loader):
            if max_grad_norm > 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        batch_size = labels.size(0)
        total_examples += batch_size
        # Undo accumulation division for logging
        total_loss += loss.item() * gradient_accumulation_steps * batch_size
        total_bce_loss += bce_loss.item() * batch_size
        total_go_term_loss += go_term_loss.item() * batch_size
        total_go_term_soft_f1 += go_term_soft_f1.item() * batch_size
        progress.set_postfix(
            loss=f"{(total_loss / max(total_examples, 1)):.4f}",
            go_f1=f"{(total_go_term_soft_f1 / max(total_examples, 1)):.4f}",
        )

    loss_value = total_loss / max(total_examples, 1)
    return EpochResult(
        loss=loss_value,
        bce_loss=total_bce_loss / max(total_examples, 1),
        go_term_loss=total_go_term_loss / max(total_examples, 1),
        go_term_soft_f1=total_go_term_soft_f1 / max(total_examples, 1),
    )


def compute_go_term_soft_f1_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.sigmoid(logits)
    labels = labels.to(dtype=probs.dtype)

    tp = (probs * labels).sum(dim=0)
    fp = (probs * (1.0 - labels)).sum(dim=0)
    fn = ((1.0 - probs) * labels).sum(dim=0)

    soft_f1_per_term = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    go_term_soft_f1 = soft_f1_per_term.mean()
    go_term_loss = 1.0 - go_term_soft_f1
    return go_term_loss, go_term_soft_f1


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    progress_desc: str | None = None,
    use_amp: bool = False,
) -> Dict[str, np.ndarray]:
    model.eval()
    all_pids: List[str] = []
    all_probs: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    progress = tqdm(
        loader,
        desc=progress_desc or "Validation",
        leave=False,
        dynamic_ncols=True,
    )

    for pids, batch_inputs, labels in progress:
        batch_inputs = move_batch_to_device(batch_inputs, device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_inputs)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        all_pids.extend(pids)
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return {
        "pids": np.asarray(all_pids),
        "probs": np.concatenate(all_probs, axis=0) if all_probs else np.empty((0, 0)),
        "labels": np.concatenate(all_labels, axis=0) if all_labels else np.empty((0, 0)),
    }


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    progress_desc: str | None = None,
) -> Dict[str, float]:
    if y_true.size == 0 or y_prob.size == 0:
        return {
            "micro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "fmax": 0.0,
            "fmax_threshold": threshold,
        }

    y_pred = (y_prob >= threshold).astype(np.int32)
    best_f1 = 0.0
    best_threshold = threshold
    thresholds = np.linspace(0.01, 0.99, 99)
    progress = tqdm(
        thresholds,
        desc=progress_desc or "Fmax Search",
        leave=False,
        dynamic_ncols=True,
    )
    for candidate in progress:
        candidate_pred = (y_prob >= candidate).astype(np.int32)
        candidate_f1 = float(f1_score(y_true, candidate_pred, average="micro", zero_division=0))
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = float(candidate)

    return {
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_precision": float(precision_score(y_true, y_pred, average="micro", zero_division=0)),
        "micro_recall": float(recall_score(y_true, y_pred, average="micro", zero_division=0)),
        "fmax": best_f1,
        "fmax_threshold": best_threshold,
    }
