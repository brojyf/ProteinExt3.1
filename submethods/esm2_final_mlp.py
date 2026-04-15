from __future__ import annotations

import torch.nn as nn

from submethods.modeling import AttentionPooling, MLPHead


class ESM2FinalLayerClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        embedding_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.pooler = AttentionPooling(embedding_dim)
        self.head = MLPHead(
            input_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        pooled = self.pooler(batch_inputs["token_embeddings"], batch_inputs["attention_mask"])
        return self.head(pooled)


def build_model(args, num_classes: int) -> nn.Module:
    return ESM2FinalLayerClassifier(
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.esm2_embedding_dim,
    )
