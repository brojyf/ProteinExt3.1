from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.data.data_utils import collect_unique_sequences_from_folds

DEFAULT_EMBEDDING_DIR = Path(__file__).resolve().parent / "embedding"
DEFAULT_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_MAX_LENGTH = 1024
DEFAULT_T5_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
DEFAULT_ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM2_INTERMEDIATE_LAYER = 20
DEFAULT_POOLING = "mean"


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plm", required=True, choices=["esm2", "t5"])
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument("--pooling", required=True, choices=["max", "mean", "both"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProteinExt3 embedding extraction")
    add_embedding_args(parser)
    return parser.parse_args()


def is_mps_available() -> bool:
    return bool(
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and torch.backends.mps.is_built()
    )


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested device=cuda, but CUDA is not available.")
        return torch.device("cuda")

    if requested_device == "mps":
        if not is_mps_available():
            raise RuntimeError("Requested device=mps, but Apple Metal (MPS) is not available.")
        return torch.device("mps")

    return torch.device("cpu")


def _require_transformers():
    try:
        from transformers import AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer
        from transformers.masking_utils import create_bidirectional_mask
    except ImportError as exc:
        raise RuntimeError(
            "Embedding extraction requires transformers. Install it in your environment."
        ) from exc
    return AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer, create_bidirectional_mask


def _ensure_prott5_dependencies() -> None:
    missing = []
    if importlib.util.find_spec("sentencepiece") is None:
        missing.append("sentencepiece")
    if importlib.util.find_spec("google.protobuf") is None:
        missing.append("protobuf")
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "ProtT5 embedding extraction is missing required dependencies: "
            f"{missing_str}. Install them, for example: "
            "`pip install sentencepiece protobuf`."
        )


def normalize_prott5_sequence(sequence: str) -> str:
    sequence = re.sub(r"[UZOB]", "X", sequence.upper())
    return " ".join(sequence)


def save_embedding_tensor(path: Path, tensor: torch.Tensor) -> None:
    torch.save(tensor, path)


def esm2_layer_dir(layer_index: int) -> str:
    return f"layer{layer_index}"


def hidden_state_layer_dir(layer_index: int) -> str:
    return str(layer_index)


def resolve_layer_indices(layer_indices: Sequence[int]) -> List[int]:
    resolved: List[int] = []
    seen = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            raise ValueError(f"Layer index must be non-negative, got {layer_index}.")
        if layer_index in seen:
            continue
        seen.add(layer_index)
        resolved.append(layer_index)
    if not resolved:
        raise ValueError("At least one layer index is required.")
    return resolved


def resolve_pooling_names(pooling: str) -> List[str]:
    if pooling == "both":
        return ["mean", "max"]
    return [pooling]


def _validate_hidden_state_layers(layer_indices: Sequence[int], max_layer_index: int, plm: str) -> None:
    invalid = [layer_index for layer_index in layer_indices if layer_index > max_layer_index]
    if invalid:
        raise ValueError(
            f"Requested {plm} layers {invalid}, but available hidden state indices are 0..{max_layer_index}."
        )


def pool_hidden_state(hidden_state: torch.Tensor, attention_mask: torch.Tensor, pooling_name: str) -> torch.Tensor:
    mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
    if pooling_name == "mean":
        masked_hidden = hidden_state * mask.to(dtype=hidden_state.dtype)
        counts = attention_mask.sum(dim=1, keepdim=True).clamp_min(1).to(dtype=hidden_state.dtype)
        return masked_hidden.sum(dim=1) / counts
    if pooling_name == "max":
        fill_value = torch.finfo(hidden_state.dtype).min
        return hidden_state.masked_fill(~mask, fill_value).max(dim=1).values
    raise ValueError(f"Unsupported pooling mode: {pooling_name}")


def save_pooled_batch(
    *,
    output_dir: Path,
    plm: str,
    layer_index: int,
    pooling_names: Sequence[str],
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_pids: Sequence[str],
) -> None:
    for pooling_name in pooling_names:
        pooled_batch = pool_hidden_state(hidden_state, attention_mask, pooling_name).detach().cpu().to(torch.float16)
        for index, pid in enumerate(batch_pids):
            output_path = output_dir / plm / pooling_name / hidden_state_layer_dir(layer_index) / f"{pid}.pt"
            if output_path.exists():
                continue
            save_embedding_tensor(output_path, pooled_batch[index].clone())


def print_embedding_device_summary(device: torch.device) -> None:
    print(f"Embedding extraction device: {device.type}")


def embedding_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


@torch.inference_mode()
def extract_esm2_embeddings(
    *,
    sequences_by_pid: Dict[str, str],
    output_dir: Path,
    pretrained_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    layer_indices: Sequence[int] | None = None,
    pooling: str = DEFAULT_POOLING,
) -> None:
    AutoTokenizer, EsmModel, _, _, _ = _require_transformers()
    resolved_layers = resolve_layer_indices(
        layer_indices if layer_indices is not None else [DEFAULT_ESM2_INTERMEDIATE_LAYER]
    )
    pooling_names = resolve_pooling_names(pooling)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = EsmModel.from_pretrained(pretrained_name, add_pooling_layer=False).to(device)
    model.eval()
    _validate_hidden_state_layers(resolved_layers, model.config.num_hidden_layers, "esm2")
    max_layer_index = max(resolved_layers)

    for pooling_name in pooling_names:
        for layer_index in resolved_layers:
            (output_dir / "esm2" / pooling_name / hidden_state_layer_dir(layer_index)).mkdir(
                parents=True, exist_ok=True
            )

    pids = []
    for pid in sorted(sequences_by_pid):
        if any(
            not (output_dir / "esm2" / pooling_name / hidden_state_layer_dir(layer_index) / f"{pid}.pt").exists()
            for pooling_name in pooling_names
            for layer_index in resolved_layers
        ):
            pids.append(pid)

    if not pids:
        print("All ESM2 embeddings already exist. Skipping.")
        return
    progress = tqdm(range(0, len(pids), batch_size), desc="Extracting ESM2 embeddings", dynamic_ncols=True)
    for start in progress:
        batch_pids = pids[start : start + batch_size]
        batch_sequences = [sequences_by_pid[pid] for pid in batch_pids]
        tokens = tokenizer(
            batch_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with embedding_autocast(device):
            attention_mask = tokens["attention_mask"]
            hidden_states = model.embeddings(
                input_ids=tokens["input_ids"],
                attention_mask=attention_mask,
                position_ids=tokens.get("position_ids"),
            )
            encoder_attention_mask = None
            layer_attention_mask, encoder_attention_mask = model._create_attention_masks(
                attention_mask=attention_mask,
                encoder_attention_mask=encoder_attention_mask,
                embedding_output=hidden_states,
                encoder_hidden_states=None,
                cache_position=torch.arange(hidden_states.shape[1], device=hidden_states.device),
                past_key_values=None,
            )

            if 0 in resolved_layers:
                save_pooled_batch(
                    output_dir=output_dir,
                    plm="esm2",
                    layer_index=0,
                    pooling_names=pooling_names,
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                    batch_pids=batch_pids,
                )

            if max_layer_index == 0:
                continue

            for layer_number, layer_module in enumerate(model.encoder.layer, start=1):
                hidden_states = layer_module(
                    hidden_states,
                    attention_mask=layer_attention_mask,
                    encoder_hidden_states=None,
                    encoder_attention_mask=encoder_attention_mask,
                )
                if layer_number == model.config.num_hidden_layers and model.encoder.emb_layer_norm_after is not None:
                    hidden_states = model.encoder.emb_layer_norm_after(hidden_states)
                if layer_number in resolved_layers:
                    save_pooled_batch(
                        output_dir=output_dir,
                        plm="esm2",
                        layer_index=layer_number,
                        pooling_names=pooling_names,
                        hidden_state=hidden_states,
                        attention_mask=attention_mask,
                        batch_pids=batch_pids,
                    )
                if layer_number >= max_layer_index:
                    break


@torch.inference_mode()
def extract_t5_embeddings(
    *,
    sequences_by_pid: Dict[str, str],
    output_dir: Path,
    pretrained_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    layer_indices: Sequence[int] | None = None,
    pooling: str = DEFAULT_POOLING,
) -> None:
    _ensure_prott5_dependencies()
    _, _, T5EncoderModel, T5Tokenizer, create_bidirectional_mask = _require_transformers()
    resolved_layers = resolve_layer_indices(layer_indices if layer_indices is not None else [0])
    pooling_names = resolve_pooling_names(pooling)

    tokenizer = T5Tokenizer.from_pretrained(pretrained_name, do_lower_case=False)
    encoder_kwargs = {}
    if device.type == "mps":
        encoder_kwargs["torch_dtype"] = torch.float32
    model = T5EncoderModel.from_pretrained(pretrained_name, **encoder_kwargs).to(device)
    if device.type == "mps":
        model = model.float()
    model.eval()
    _validate_hidden_state_layers(resolved_layers, model.config.num_layers, "t5")
    max_layer_index = max(resolved_layers)

    for pooling_name in pooling_names:
        for layer_index in resolved_layers:
            (output_dir / "t5" / pooling_name / hidden_state_layer_dir(layer_index)).mkdir(
                parents=True, exist_ok=True
            )

    pids = []
    for pid in sorted(sequences_by_pid):
        if any(
            not (output_dir / "t5" / pooling_name / hidden_state_layer_dir(layer_index) / f"{pid}.pt").exists()
            for pooling_name in pooling_names
            for layer_index in resolved_layers
        ):
            pids.append(pid)

    if not pids:
        print("All ProtT5 embeddings already exist. Skipping.")
        return
    progress = tqdm(range(0, len(pids), batch_size), desc="Extracting ProtT5 embeddings", dynamic_ncols=True)
    for start in progress:
        batch_pids = pids[start : start + batch_size]
        processed_sequences = [normalize_prott5_sequence(sequences_by_pid[pid]) for pid in batch_pids]
        tokens = tokenizer(
            processed_sequences,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with embedding_autocast(device):
            attention_mask = tokens["attention_mask"]
            hidden_states = model.encoder.embed_tokens(tokens["input_ids"])
            hidden_states = model.encoder.dropout(hidden_states)
            layer_attention_mask = create_bidirectional_mask(
                config=model.encoder.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
            )
            position_bias = None

            if 0 in resolved_layers:
                save_pooled_batch(
                    output_dir=output_dir,
                    plm="t5",
                    layer_index=0,
                    pooling_names=pooling_names,
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                    batch_pids=batch_pids,
                )

            if max_layer_index == 0:
                continue

            for layer_number, layer_module in enumerate(model.encoder.block, start=1):
                layer_outputs = layer_module(
                    hidden_states,
                    layer_attention_mask,
                    position_bias,
                    None,
                    None,
                    None,
                    past_key_values=None,
                    use_cache=False,
                    output_attentions=False,
                    return_dict=True,
                    cache_position=None,
                )
                hidden_states = layer_outputs[0]
                position_bias = layer_outputs[1]
                if layer_number == model.config.num_layers:
                    hidden_states = model.encoder.final_layer_norm(hidden_states)
                    hidden_states = model.encoder.dropout(hidden_states)
                if layer_number in resolved_layers:
                    save_pooled_batch(
                        output_dir=output_dir,
                        plm="t5",
                        layer_index=layer_number,
                        pooling_names=pooling_names,
                        hidden_state=hidden_states,
                        attention_mask=attention_mask,
                        batch_pids=batch_pids,
                    )
                if layer_number >= max_layer_index:
                    break


def run_embedding_extraction(args) -> None:
    if args.plm is None:
        raise RuntimeError("Embedding extraction requires `--plm esm2` or `--plm t5`.")

    sequences_by_pid = collect_unique_sequences_from_folds(args.fold)
    print(f"Embedding extraction | plm={args.plm} | proteins={len(sequences_by_pid)}")
    print_embedding_device_summary(args.device)

    if args.plm == "esm2":
        extract_esm2_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=args.embedding_dir,
            pretrained_name=DEFAULT_ESM2_NAME,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            layer_indices=args.layers,
            pooling=args.pooling,
        )
    elif args.plm == "t5":
        extract_t5_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=args.embedding_dir,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            layer_indices=args.layers,
            pooling=args.pooling,
        )
    else:
        raise ValueError(f"Unsupported PLM for embedding extraction: {args.plm}")


def main() -> None:
    args = parse_args()
    args.fold = list(DEFAULT_FOLDS)
    args.embedding_dir = DEFAULT_EMBEDDING_DIR
    args.max_length = DEFAULT_MAX_LENGTH
    args.device = resolve_device("auto")
    run_embedding_extraction(args)


if __name__ == "__main__":
    main()
