from __future__ import annotations

import torch
import torch.nn as nn

from submethods.modeling import ConvClassifierHead


class ESM2Layer20CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_hidden_dim: int = 512,
        dropout: float = 0.3,
        embedding_dim: int = 1280,
    ) -> None:
        super().__init__()
        self.head = ConvClassifierHead(
            input_dim=embedding_dim + 2,
            num_classes=num_classes,
            hidden_dim=cnn_hidden_dim,
            dropout=dropout,
        )

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        fused = torch.cat([batch_inputs["token_embeddings"], batch_inputs["hydro_features"]], dim=-1)
        return self.head(fused, batch_inputs["attention_mask"])


def build_model(args, num_classes: int) -> nn.Module:
    return ESM2Layer20CNNClassifier(
        num_classes=num_classes,
        cnn_hidden_dim=args.cnn_hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.esm2_embedding_dim,
    )
