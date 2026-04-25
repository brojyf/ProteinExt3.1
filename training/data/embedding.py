from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from tqdm.auto import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from training.data.data_utils import load_fasta_sequences

DEFAULT_EMBEDDING_DIR = Path(__file__).resolve().parent / "embedding"
DEFAULT_RAW_FASTA = Path(__file__).resolve().parent / "raw" / "training.fasta"
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 1024
DEFAULT_T5_NAME = "Rostlab/prot_t5_xl_half_uniref50-enc"
DEFAULT_ESM2_NAME = "facebook/esm2_t33_650M_UR50D"
DEFAULT_ESM2_INTERMEDIATE_LAYER = 20
DEFAULT_POOLING = "both"
DEFAULT_SHARD_SIZE = 4096


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plm", required=True, choices=["esm2", "prott5"])
    parser.add_argument("--fasta", type=Path, default=DEFAULT_RAW_FASTA)
    parser.add_argument("--out-dir", dest="embedding_dir", type=Path, default=DEFAULT_EMBEDDING_DIR)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--layers", required=True, type=int, nargs="+")
    parser.add_argument("--pooling", choices=["max", "mean", "both"], default=DEFAULT_POOLING)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)


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


def hidden_state_layer_dir(layer_index: int) -> str:
    return str(layer_index)


def resolve_layer_indices(layer_indices: Sequence[int]) -> List[int]:
    resolved: List[int] = []
    seen = set()
    for layer_index in layer_indices:
        if layer_index < 0:
            raise ValueError(f"Layer index must be non-negative, got {layer_index}.")
        if resolved and layer_index < resolved[-1]:
            raise ValueError("Layer indices must be in ascending order because inference stops at the last requested layer.")
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


def shard_index_path(layer_dir: Path) -> Path:
    return layer_dir / "index.json"


def load_shard_index(layer_dir: Path) -> Dict[str, str]:
    path = shard_index_path(layer_dir)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise TypeError(f"Expected shard index object at {path}")
    return {str(pid): str(shard_name) for pid, shard_name in loaded.items()}


def save_shard_index(layer_dir: Path, index: Dict[str, str]) -> None:
    with shard_index_path(layer_dir).open("w", encoding="utf-8") as handle:
        json.dump(index, handle, indent=2, sort_keys=True)


def pooled_embedding_exists(
    output_dir: Path,
    plm: str,
    pooling_name: str,
    layer_index: int,
    pid: str,
    shard_index: Dict[str, str] | None = None,
) -> bool:
    layer_dir = output_dir / plm / pooling_name / hidden_state_layer_dir(layer_index)
    if (layer_dir / f"{pid}.pt").exists():
        return True
    index = shard_index if shard_index is not None else load_shard_index(layer_dir)
    shard_name = index.get(pid)
    return shard_name is not None and (layer_dir / shard_name).exists()


class ShardEmbeddingWriter:
    def __init__(self, output_dir: Path, shard_size: int) -> None:
        if shard_size <= 0:
            raise ValueError(f"shard_size must be positive, got {shard_size}")
        self.output_dir = output_dir
        self.shard_size = shard_size
        self.indices: Dict[Path, Dict[str, str]] = {}
        self.buffers: Dict[Path, Dict[str, torch.Tensor]] = {}
        self.next_shard_ids: Dict[Path, int] = {}

    def _layer_dir(self, plm: str, pooling_name: str, layer_index: int) -> Path:
        return self.output_dir / plm / pooling_name / hidden_state_layer_dir(layer_index)

    def _ensure_layer(self, layer_dir: Path) -> None:
        if layer_dir in self.indices:
            return
        layer_dir.mkdir(parents=True, exist_ok=True)
        self.indices[layer_dir] = load_shard_index(layer_dir)
        shard_ids = []
        for path in layer_dir.glob("shard_*.pt"):
            try:
                shard_ids.append(int(path.stem.removeprefix("shard_")))
            except ValueError:
                continue
        self.next_shard_ids[layer_dir] = max(shard_ids, default=-1) + 1
        self.buffers[layer_dir] = {}

    def add_batch(
        self,
        *,
        plm: str,
        pooling_name: str,
        layer_index: int,
        batch_pids: Sequence[str],
        pooled_batch: torch.Tensor,
    ) -> None:
        layer_dir = self._layer_dir(plm, pooling_name, layer_index)
        self._ensure_layer(layer_dir)
        index = self.indices[layer_dir]
        buffer = self.buffers[layer_dir]
        for tensor_index, pid in enumerate(batch_pids):
            if pid in index or pid in buffer:
                continue
            buffer[pid] = pooled_batch[tensor_index].clone()
            if len(buffer) >= self.shard_size:
                self.flush_layer(layer_dir)

    def flush_layer(self, layer_dir: Path) -> None:
        buffer = self.buffers[layer_dir]
        if not buffer:
            return
        shard_name = f"shard_{self.next_shard_ids[layer_dir]:05d}.pt"
        torch.save(dict(buffer), layer_dir / shard_name)
        for pid in buffer:
            self.indices[layer_dir][pid] = shard_name
        buffer.clear()
        self.next_shard_ids[layer_dir] += 1
        save_shard_index(layer_dir, self.indices[layer_dir])

    def close(self) -> None:
        for layer_dir in list(self.buffers):
            self.flush_layer(layer_dir)


def save_pooled_batch(
    *,
    output_dir: Path,
    plm: str,
    layer_index: int,
    pooling_names: Sequence[str],
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    batch_pids: Sequence[str],
    shard_writer: ShardEmbeddingWriter,
) -> None:
    for pooling_name in pooling_names:
        pooled_batch = pool_hidden_state(hidden_state, attention_mask, pooling_name).detach().cpu().to(torch.float16)
        shard_writer.add_batch(
            plm=plm,
            pooling_name=pooling_name,
            layer_index=layer_index,
            batch_pids=batch_pids,
            pooled_batch=pooled_batch,
        )


def print_embedding_device_summary(device: torch.device) -> None:
    print(f"Embedding extraction device: {device.type}")


def embedding_autocast(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast("cuda")
    return nullcontext()


def create_esm2_attention_masks(model, attention_mask: torch.Tensor, hidden_states: torch.Tensor):
    kwargs = {
        "attention_mask": attention_mask,
        "encoder_attention_mask": None,
        "embedding_output": hidden_states,
        "encoder_hidden_states": None,
        "past_key_values": None,
    }
    if "cache_position" in inspect.signature(model._create_attention_masks).parameters:
        kwargs["cache_position"] = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    return model._create_attention_masks(**kwargs)


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
    shard_size: int = DEFAULT_SHARD_SIZE,
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
    max_layer_index = resolved_layers[-1]
    shard_writer = ShardEmbeddingWriter(output_dir, shard_size)

    for pooling_name in pooling_names:
        for layer_index in resolved_layers:
            (output_dir / "esm2" / pooling_name / hidden_state_layer_dir(layer_index)).mkdir(
                parents=True, exist_ok=True
            )
    shard_indices = {
        (pooling_name, layer_index): load_shard_index(
            output_dir / "esm2" / pooling_name / hidden_state_layer_dir(layer_index)
        )
        for pooling_name in pooling_names
        for layer_index in resolved_layers
    }

    pids = []
    for pid in sorted(sequences_by_pid):
        if any(
            not pooled_embedding_exists(
                output_dir,
                "esm2",
                pooling_name,
                layer_index,
                pid,
                shard_indices.get((pooling_name, layer_index)),
            )
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
            layer_attention_mask, encoder_attention_mask = create_esm2_attention_masks(model, attention_mask, hidden_states)

            if 0 in resolved_layers:
                save_pooled_batch(
                    output_dir=output_dir,
                    plm="esm2",
                    layer_index=0,
                    pooling_names=pooling_names,
                    hidden_state=hidden_states,
                    attention_mask=attention_mask,
                    batch_pids=batch_pids,
                    shard_writer=shard_writer,
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
                        shard_writer=shard_writer,
                    )
                if layer_number >= max_layer_index:
                    break
    shard_writer.close()


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
    shard_size: int = DEFAULT_SHARD_SIZE,
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
    max_layer_index = resolved_layers[-1]
    shard_writer = ShardEmbeddingWriter(output_dir, shard_size)

    for pooling_name in pooling_names:
        for layer_index in resolved_layers:
            (output_dir / "t5" / pooling_name / hidden_state_layer_dir(layer_index)).mkdir(
                parents=True, exist_ok=True
            )
    shard_indices = {
        (pooling_name, layer_index): load_shard_index(
            output_dir / "t5" / pooling_name / hidden_state_layer_dir(layer_index)
        )
        for pooling_name in pooling_names
        for layer_index in resolved_layers
    }

    pids = []
    for pid in sorted(sequences_by_pid):
        if any(
            not pooled_embedding_exists(
                output_dir,
                "t5",
                pooling_name,
                layer_index,
                pid,
                shard_indices.get((pooling_name, layer_index)),
            )
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
                    shard_writer=shard_writer,
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
                        shard_writer=shard_writer,
                    )
                if layer_number >= max_layer_index:
                    break
    shard_writer.close()


def run_embedding_extraction(args) -> None:
    if args.plm is None:
        raise RuntimeError("Embedding extraction requires `--plm esm2` or `--plm prott5`.")

    sequences_by_pid = load_fasta_sequences(args.fasta)
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
            shard_size=args.shard_size,
        )
    elif args.plm == "prott5":
        extract_t5_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=args.embedding_dir,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
            layer_indices=args.layers,
            pooling=args.pooling,
            shard_size=args.shard_size,
        )
    else:
        raise ValueError(f"Unsupported PLM for embedding extraction: {args.plm}")


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    run_embedding_extraction(args)


if __name__ == "__main__":
    main()
