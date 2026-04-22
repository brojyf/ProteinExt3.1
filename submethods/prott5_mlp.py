#
from __future__ import annotations

import torch.nn as nn

from submethods.esm2_layer20_mlp import ChainMLPClassifier
from training.data.data_utils import PROTEIN_FEATURE_DIM


class ProtT5MLPClassifier(ChainMLPClassifier):
    def __init__(self, num_classes: int, **kwargs) -> None:
        super().__init__(num_classes=num_classes, embedding_dim=1024, **kwargs)


def build_model(args, num_classes: int) -> nn.Module:
    return ProtT5MLPClassifier(
        num_classes=num_classes,
        protein_feature_dim=getattr(args, "protein_feature_dim", PROTEIN_FEATURE_DIM),
        hidden_dim=args.hidden_dim,
        bottleneck=args.bottleneck,
        dropout=args.dropout,
    )
