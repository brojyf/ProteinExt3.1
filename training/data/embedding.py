from __future__ import annotations

import argparse
import importlib.util
import re
import sys
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


def add_embedding_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plm", required=True, choices=["esm2", "t5"])
    parser.add_argument("--batch-size", required=True, type=int)


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
    except ImportError as exc:
        raise RuntimeError(
            "Embedding extraction requires transformers. Install it in your environment."
        ) from exc
    return AutoTokenizer, EsmModel, T5EncoderModel, T5Tokenizer


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
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor.detach().cpu().to(torch.float16), path)


def esm2_layer_dir(layer_index: int) -> str:
    return f"layer{layer_index}"


def print_embedding_device_summary(device: torch.device) -> None:
    print(f"Embedding extraction device: {device.type}")


@torch.no_grad()
def extract_esm2_embeddings(
    *,
    sequences_by_pid: Dict[str, str],
    output_dir: Path,
    pretrained_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
    layer_index: int,
) -> None:
    AutoTokenizer, EsmModel, _, _ = _require_transformers()

    tokenizer = AutoTokenizer.from_pretrained(pretrained_name)
    model = EsmModel.from_pretrained(pretrained_name, add_pooling_layer=False).to(device)
    model.eval()

    pids = sorted(sequences_by_pid)
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
        outputs = model(**tokens, output_hidden_states=True)

        last_hidden = outputs.last_hidden_state
        layer_hidden = outputs.hidden_states[layer_index]
        attention_mask = tokens["attention_mask"]

        for index, pid in enumerate(batch_pids):
            valid_length = int(attention_mask[index].sum().item())
            save_embedding_tensor(output_dir / "esm2" / "last" / f"{pid}.pt", last_hidden[index, :valid_length])
            save_embedding_tensor(
                output_dir / "esm2" / esm2_layer_dir(layer_index) / f"{pid}.pt",
                layer_hidden[index, :valid_length],
            )


@torch.no_grad()
def extract_t5_embeddings(
    *,
    sequences_by_pid: Dict[str, str],
    output_dir: Path,
    pretrained_name: str,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> None:
    _ensure_prott5_dependencies()
    _, _, T5EncoderModel, T5Tokenizer = _require_transformers()

    tokenizer = T5Tokenizer.from_pretrained(pretrained_name, do_lower_case=False)
    encoder_kwargs = {}
    if device.type == "mps":
        encoder_kwargs["torch_dtype"] = torch.float32
    model = T5EncoderModel.from_pretrained(pretrained_name, **encoder_kwargs).to(device)
    if device.type == "mps":
        model = model.float()
    model.eval()

    pids = sorted(sequences_by_pid)
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
        outputs = model(**tokens)
        hidden = outputs.last_hidden_state
        attention_mask = tokens["attention_mask"]

        for index, pid in enumerate(batch_pids):
            valid_length = int(attention_mask[index].sum().item())
            save_embedding_tensor(output_dir / "t5" / "last" / f"{pid}.pt", hidden[index, :valid_length])


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
            layer_index=DEFAULT_ESM2_INTERMEDIATE_LAYER,
        )
    elif args.plm == "t5":
        extract_t5_embeddings(
            sequences_by_pid=sequences_by_pid,
            output_dir=args.embedding_dir,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=args.device,
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
