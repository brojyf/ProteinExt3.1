from __future__ import annotations

import torch
import torch.nn as nn

from submethods.modeling import ConvClassifierHead
from training.data.data_utils import PROTEIN_FEATURE_DIM


class ESM2Layer20CNNClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_hidden_dim: int = 512,
        dropout: float = 0.3,
        embedding_dim: int = 1280,
        protein_feature_dim: int = PROTEIN_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.protein_feature_dim = protein_feature_dim
        # ConvClassifierHead outputs (hidden_dim * 2) from mean+max pooling,
        # then we concat protein features before the final classifier.
        # To enable this, we use ConvClassifierHead for conv encoding only
        # and build a separate classifier that accepts the extra features.
        self.head = ConvClassifierHead(
            input_dim=embedding_dim + 2,
            num_classes=num_classes,
            hidden_dim=cnn_hidden_dim,
            dropout=dropout,
            protein_feature_dim=protein_feature_dim,
        )

    def forward(self, batch_inputs: dict) -> torch.Tensor:
        fused = torch.cat([batch_inputs["token_embeddings"], batch_inputs["hydro_features"]], dim=-1)
        return self.head(fused, batch_inputs["attention_mask"], batch_inputs.get("protein_features"))


def build_model(args, num_classes: int) -> nn.Module:
    return ESM2Layer20CNNClassifier(
        num_classes=num_classes,
        cnn_hidden_dim=args.cnn_hidden_dim,
        dropout=args.dropout,
        embedding_dim=args.esm2_embedding_dim,
        protein_feature_dim=getattr(args, "protein_feature_dim", PROTEIN_FEATURE_DIM),
    )
