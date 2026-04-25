from __future__ import annotations

import torch
import torch.nn as nn

from training.data.data_utils import PROTEIN_FEATURE_DIM


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int = 63, output_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 2048,
        bottleneck: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.block = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
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
        x = self.input_proj(self.input_norm(inputs))
        x = x + self.block(x)
        return self.output_head(x)


class ChainMLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        protein_feature_dim: int = PROTEIN_FEATURE_DIM,
        hidden_dim: int = 1024,
        bottleneck: int = 1024,
        dropout: float = 0.3,
        pooling: str = "both",
        use_crafted_features: bool = True,
    ) -> None:
        super().__init__()
        self.pooling = pooling
        self.use_crafted_features = use_crafted_features
        self.feature_encoder = FeatureEncoder(protein_feature_dim, 256, dropout) if use_crafted_features else None
        pooling_width = 2 if pooling == "both" else 1
        feature_width = 256 if use_crafted_features else 0
        input_dim = pooling_width * embedding_dim + feature_width
        self.head = MLPHead(input_dim, num_classes, hidden_dim=hidden_dim, bottleneck=bottleneck, dropout=dropout)

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        pooled = batch_inputs["pooled_embeddings"]
        if self.feature_encoder is not None:
            feature_repr = self.feature_encoder(batch_inputs["protein_features"].to(dtype=pooled.dtype))
            pooled = torch.cat([pooled, feature_repr], dim=-1)
        return self.head(pooled)
