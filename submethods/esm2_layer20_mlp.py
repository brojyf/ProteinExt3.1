from __future__ import annotations

import torch
import torch.nn as nn

from submethods.modeling import FeatureEncoder, MLPHead, MaxPooling, MeanPooling
from training.data.data_utils import PROTEIN_FEATURE_DIM


class ChainMLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embedding_dim: int,
        protein_feature_dim: int = PROTEIN_FEATURE_DIM,
        hidden_dim: int = 1024,
        bottleneck: int = 1024,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.mean_pooler = MeanPooling()
        self.max_pooler = MaxPooling()
        self.feature_encoder = FeatureEncoder(protein_feature_dim, 256, dropout)
        input_dim = 2 * embedding_dim + 256
        self.head = MLPHead(input_dim, num_classes, hidden_dim=hidden_dim, bottleneck=bottleneck, dropout=dropout)

    def _mean_max(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.mean_pooler(values, mask), self.max_pooler(values, mask)

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        mean_pooled, max_pooled = self._mean_max(batch_inputs["token_embeddings"], batch_inputs["attention_mask"])
        feature_repr = self.feature_encoder(batch_inputs["protein_features"].to(dtype=mean_pooled.dtype))
        fused = torch.cat([mean_pooled, max_pooled, feature_repr], dim=-1)
        return self.head(fused)


class ESM2Layer20MLPClassifier(ChainMLPClassifier):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes=num_classes, embedding_dim=1280, **kwargs)


def build_model(args, num_classes: int) -> nn.Module:
    return ESM2Layer20MLPClassifier(
        num_classes=num_classes,
        protein_feature_dim=getattr(args, "protein_feature_dim", PROTEIN_FEATURE_DIM),
        hidden_dim=args.hidden_dim,
        bottleneck=args.bottleneck,
        dropout=args.dropout,
    )
