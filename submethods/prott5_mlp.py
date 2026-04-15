from __future__ import annotations

import torch
import torch.nn as nn

from submethods.modeling import AttentionPooling, MLPHead, MeanPooling


class ProtT5MLPClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        pooling: str = "mean",
        hidden_dim: int = 1024,
        dropout: float = 0.3,
        embedding_dim: int = 1024,
    ) -> None:
        super().__init__()
        if pooling == "mean":
            self.pooler = MeanPooling()
        elif pooling == "attention":
            self.pooler = AttentionPooling(embedding_dim)
        else:
            raise ValueError(f"Unsupported ProtT5 pooling: {pooling}")

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
    return ProtT5MLPClassifier(
        num_classes=num_classes,
        pooling=args.pooling,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.t5_embedding_dim,
    )
