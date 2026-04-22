from submethods.esm2_final_mlp import build_model as build_esm2_model
from submethods.esm2_layer20_mlp import build_model as build_esm2_l20_model
from submethods.prott5_mlp import build_model as build_t5_model

MODEL_BUILDERS = {
    "esm2_last": build_esm2_model,
    "esm2_l20": build_esm2_l20_model,
    "prott5": build_t5_model,
    "esm2": build_esm2_model,
    "t5": build_t5_model,
}
