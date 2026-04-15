from submethods.bp_blast_transfer import maybe_run_bp_blast_enhancement
from submethods.esm2_final_mlp import build_model as build_model2
from submethods.esm2_layer20_cnn import build_model as build_model3
from submethods.prott5_mlp import build_model as build_model1


MODEL_BUILDERS = {
    "t5": build_model1,
    "esm2": build_model2,
    "cnn": build_model3,
    "model1": build_model1,
    "model2": build_model2,
    "model3": build_model3,
}


__all__ = ["MODEL_BUILDERS", "maybe_run_bp_blast_enhancement"]
