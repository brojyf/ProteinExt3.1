from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

import torch
import torch.nn as nn


@dataclass
class ModelBatch:
    sequences: List[str]
    labels: Optional[torch.Tensor] = None


class MeanPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return summed / denom


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.query(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(attention_mask > 0, weights, torch.zeros_like(weights))
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        bottleneck = max(hidden_dim // 2, 1)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck),
            nn.LayerNorm(bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ConvClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        kernel_size: int = 5,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = token_features.transpose(1, 2)
        x = self.encoder(x)
        mask = attention_mask.unsqueeze(1).to(x.dtype)
        x = x * mask

        denom = mask.sum(dim=-1).clamp_min(1.0)
        avg_pool = x.sum(dim=-1) / denom

        max_mask = attention_mask.unsqueeze(1).expand_as(x)
        max_fill = torch.full_like(x, float("-inf"))
        max_pool = torch.where(max_mask > 0, x, max_fill).amax(dim=-1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        features = torch.cat([avg_pool, max_pool], dim=-1)
        return self.classifier(features)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def masked_sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1)
