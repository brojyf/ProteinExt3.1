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
    # Math trick: bce without reduction is mathematically equal to -log(p_t)
    p_t = torch.exp(-bce)
    focal_weight = (1.0 - p_t) ** gamma
    return (focal_weight * bce).mean()


def asymmetric_loss_with_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    gamma_neg: float = 4.0,
    gamma_pos: float = 0.0,
    clip: float = 0.05,
) -> torch.Tensor:
    """
    Asymmetric Loss for multi-label classification (Ridnik et al., ICCV 2021).

    Applies different focusing parameters for positive and negative samples,
    plus probability shifting on negatives to further suppress easy negatives.
    Designed for extreme label imbalance (95%+ negatives) typical in GO annotation.
    """
    probs = torch.sigmoid(logits)

    # Probability shifting: reduce negative probabilities to suppress easy negatives
    probs_neg = (probs - clip).clamp(min=0.0)

    # Separate positive and negative log-probabilities
    log_pos = torch.log(probs.clamp(min=1e-8))
    log_neg = torch.log((1.0 - probs_neg).clamp(min=1e-8))

    # Asymmetric focal weighting
    if gamma_pos > 0:
        pos_weight = (1.0 - probs) ** gamma_pos
        log_pos = log_pos * pos_weight
    if gamma_neg > 0:
        neg_weight = probs_neg ** gamma_neg
        log_neg = log_neg * neg_weight

    loss = -(labels * log_pos + (1.0 - labels) * log_neg)
    return loss.mean()


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
    asl_gamma_neg: float = 0.0,
    asl_gamma_pos: float = 0.0,
    asl_clip: float = 0.05,
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

        # Preserve hard labels for soft F1 loss (must not be smoothed)
        hard_labels = labels

        # Label smoothing: shift hard 0/1 targets toward 0.5 (only for BCE/focal)
        if label_smoothing > 0:
            smoothed_labels = labels * (1.0 - label_smoothing) + label_smoothing * 0.5
        else:
            smoothed_labels = labels

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_inputs)

            if asl_gamma_neg > 0:
                bce_loss = asymmetric_loss_with_logits(
                    logits, smoothed_labels,
                    gamma_neg=asl_gamma_neg, gamma_pos=asl_gamma_pos, clip=asl_clip,
                )
            elif focal_gamma > 0:
                bce_loss = focal_bce_with_logits(logits, smoothed_labels, gamma=focal_gamma)
            else:
                bce_loss = F.binary_cross_entropy_with_logits(logits, smoothed_labels)

            go_term_loss, go_term_soft_f1 = compute_go_term_soft_f1_loss(logits, hard_labels)
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
    pred_pos = probs.sum(dim=0)
    true_pos = labels.sum(dim=0)

    # 2*TP + FP + FN is mathematically equivalent to pred_pos + true_pos
    soft_f1_per_term = (2.0 * tp + eps) / (pred_pos + true_pos + eps)

    # Only average over terms that have at least one positive in the batch.
    # Terms with zero positives penalize any positive prediction (soft_f1→0),
    # pushing the model toward all-negative outputs (fmax → 0).
    active = true_pos > 0
    if active.any():
        go_term_soft_f1 = soft_f1_per_term[active].mean()
    else:
        go_term_soft_f1 = soft_f1_per_term.new_ones(())

    go_term_loss = 1.0 - go_term_soft_f1
    return go_term_loss, go_term_soft_f1


@torch.inference_mode()
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

    y_pred = y_prob >= threshold
    y_true_bool = y_true.astype(bool)
    total_true = y_true_bool.sum()
    
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
        candidate_pred = y_prob >= candidate
        tp = np.logical_and(candidate_pred, y_true_bool).sum()
        pred_positives = candidate_pred.sum()
        
        # Math trick: 2*TP + FP + FN == pred_positives + total_true
        denom = pred_positives + total_true
        candidate_f1 = float((2 * tp) / denom) if denom > 0 else 0.0
        
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = float(candidate)

    tp_base = np.logical_and(y_pred, y_true_bool).sum()
    pred_positives_base = y_pred.sum()

    denom_f1 = pred_positives_base + total_true
    denom_p = pred_positives_base
    # Recall denom = TP + FN = total_true
    denom_r = total_true

    return {
        "micro_f1": float((2 * tp_base) / denom_f1) if denom_f1 > 0 else 0.0,
        "micro_precision": float(tp_base / denom_p) if denom_p > 0 else 0.0,
        "micro_recall": float(tp_base / denom_r) if denom_r > 0 else 0.0,
        "fmax": best_f1,
        "fmax_threshold": best_threshold,
    }


@torch.inference_mode()
def evaluate_multilabel_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
    progress_desc: str | None = None,
    use_amp: bool = False,
) -> Dict[str, float]:
    model.eval()
    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_tp = np.zeros(thresholds.shape[0], dtype=np.int64)
    threshold_pred_positives = np.zeros(thresholds.shape[0], dtype=np.int64)
    total_true = 0
    base_tp = 0
    base_pred_positives = 0

    progress = tqdm(
        loader,
        desc=progress_desc or "Validation Metrics",
        leave=False,
        dynamic_ncols=True,
    )

    for _, batch_inputs, labels in progress:
        batch_inputs = move_batch_to_device(batch_inputs, device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(batch_inputs)
        probs = torch.sigmoid(logits).float().cpu().numpy()
        labels_bool = labels.numpy().astype(bool, copy=False)
        total_true += int(labels_bool.sum())

        base_pred = probs >= threshold
        base_tp += int(np.logical_and(base_pred, labels_bool).sum())
        base_pred_positives += int(base_pred.sum())

        for index, candidate in enumerate(thresholds):
            candidate_pred = probs >= candidate
            threshold_tp[index] += int(np.logical_and(candidate_pred, labels_bool).sum())
            threshold_pred_positives[index] += int(candidate_pred.sum())

    best_f1 = 0.0
    best_threshold = threshold
    for candidate, tp, pred_positives in zip(thresholds, threshold_tp, threshold_pred_positives):
        denom = int(pred_positives) + total_true
        candidate_f1 = float((2 * int(tp)) / denom) if denom > 0 else 0.0
        if candidate_f1 > best_f1:
            best_f1 = candidate_f1
            best_threshold = float(candidate)

    denom_f1 = base_pred_positives + total_true
    return {
        "micro_f1": float((2 * base_tp) / denom_f1) if denom_f1 > 0 else 0.0,
        "micro_precision": float(base_tp / base_pred_positives) if base_pred_positives > 0 else 0.0,
        "micro_recall": float(base_tp / total_true) if total_true > 0 else 0.0,
        "fmax": best_f1,
        "fmax_threshold": best_threshold,
    }
