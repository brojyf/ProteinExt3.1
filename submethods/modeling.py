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


class MaxPooling(nn.Module):
    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand_as(hidden_states)
        fill = torch.full_like(hidden_states, float("-inf"))
        pooled = torch.where(mask > 0, hidden_states, fill).amax(dim=1)
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


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

        # Normalize raw input features of differing scales (PLM poolings + raw count features)
        self.input_norm = nn.LayerNorm(input_dim)
        
        # Project input to hidden_dim (also serves as residual shortcut)
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # First residual block (hidden_dim -> hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Bottleneck + output (no residual — intentional dimensionality reduction)
        self.output_head = nn.Sequential(
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
        x_norm = self.input_norm(inputs)
        x = self.input_proj(x_norm)
        x = x + self.block1(x)
        return self.output_head(x)


class ConvClassifierHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 512,
        kernel_sizes: tuple[int, ...] = (3, 5, 7),
        dropout: float = 0.3,
        protein_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        self.protein_feature_dim = protein_feature_dim
        branch_dim = hidden_dim // len(kernel_sizes)
        remainder = hidden_dim - branch_dim * len(kernel_sizes)

        # Multi-scale parallel conv branches
        self.branches = nn.ModuleList()
        for idx, ks in enumerate(kernel_sizes):
            out_ch = branch_dim + (1 if idx < remainder else 0)
            self.branches.append(nn.Sequential(
                nn.Conv1d(input_dim, out_ch, kernel_size=ks, padding=ks // 2),
                nn.BatchNorm1d(out_ch),
                nn.GELU(),
                nn.Dropout(dropout),
            ))

        # Second conv layer on fused multi-scale features + residual
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1) if input_dim != hidden_dim else nn.Identity()
        self.conv2 = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        classifier_input_dim = hidden_dim * 2 + protein_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        token_features: torch.Tensor,
        attention_mask: torch.Tensor,
        protein_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = token_features.transpose(1, 2)  # (B, C, L)
        mask = attention_mask.unsqueeze(1).to(x.dtype)

        # Multi-scale feature extraction
        branch_outputs = [branch(x) for branch in self.branches]
        fused = torch.cat(branch_outputs, dim=1)
        fused = fused * mask

        # Second conv with residual connection
        residual = self.proj(x) * mask
        x = residual + self.conv2(fused) * mask

        # Mean + max pooling
        denom = mask.sum(dim=-1).clamp_min(1.0)
        avg_pool = x.sum(dim=-1) / denom

        max_mask = attention_mask.unsqueeze(1).expand_as(x)
        max_fill = torch.full_like(x, float("-inf"))
        max_pool = torch.where(max_mask > 0, x, max_fill).amax(dim=-1)
        max_pool = torch.where(torch.isfinite(max_pool), max_pool, torch.zeros_like(max_pool))

        features = torch.cat([avg_pool, max_pool], dim=-1)

        if protein_features is not None:
            protein_features = protein_features.to(dtype=features.dtype)
            features = torch.cat([features, protein_features], dim=-1)
        elif self.protein_feature_dim > 0:
            features = torch.cat([
                features,
                features.new_zeros((features.size(0), self.protein_feature_dim)),
            ], dim=-1)

        return self.classifier(features)


def freeze_module(module: nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False


def masked_sequence_lengths(attention_mask: torch.Tensor) -> torch.Tensor:
    return attention_mask.sum(dim=1)
