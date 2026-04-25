from submethods.modeling import ChainMLPClassifier
from training.data.data_utils import PROTEIN_FEATURE_DIM

EMBEDDING_DIMS = {
    "esm2": 1280,
    "prott5": 1024,
}


def build_model(args, num_classes: int, embedding_dim: int) -> ChainMLPClassifier:
    return ChainMLPClassifier(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        protein_feature_dim=getattr(args, "protein_feature_dim", PROTEIN_FEATURE_DIM),
        hidden_dim=args.hidden_dim,
        bottleneck=args.bottleneck,
        dropout=args.dropout,
        pooling=args.pooling,
        use_crafted_features=args.use_crafted_features,
    )
