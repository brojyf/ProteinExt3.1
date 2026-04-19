from __future__ import annotations

import torch
import torch.nn as nn

from submethods.modeling import MLPHead, MaxPooling, MeanPooling
from training.data.data_utils import PROTEIN_FEATURE_DIM


class ProtT5MLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        embedding_dim: int = 1024,
        protein_feature_dim: int = PROTEIN_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.protein_feature_dim = protein_feature_dim
        self.mean_pooler = MeanPooling()
        self.max_pooler = MaxPooling()
        self.head = MLPHead(
            input_dim=embedding_dim * 2 + protein_feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        mean_pooled = self.mean_pooler(batch_inputs["token_embeddings"], batch_inputs["attention_mask"])
        max_pooled = self.max_pooler(batch_inputs["token_embeddings"], batch_inputs["attention_mask"])
        protein_features = batch_inputs.get("protein_features")
        if protein_features is None:
            protein_features = mean_pooled.new_zeros((mean_pooled.size(0), self.protein_feature_dim))
        else:
            protein_features = protein_features.to(dtype=mean_pooled.dtype)
        fused = torch.cat([mean_pooled, max_pooled, protein_features], dim=-1)
        return self.head(fused)


def build_model(args, num_classes: int) -> nn.Module:
    return ProtT5MLPClassifier(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.t5_embedding_dim,
        protein_feature_dim=getattr(args, "protein_feature_dim", PROTEIN_FEATURE_DIM),
    )
