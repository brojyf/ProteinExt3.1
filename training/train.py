from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Sequence

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

from submethods import EMBEDDING_DIMS, build_model
from submethods.bp_blast_transfer import _build_database, _parse_blast_hits, _require_blast, _run_blast, _transfer_scores
from training.data.data_utils import (
    EMBEDDING_DIR,
    PROTEIN_FEATURES_DIR,
    MultiEmbeddingDataset,
    build_and_save_protein_features,
    build_sequence_protein_features,
    collect_unique_sequences_from_raw,
    collect_unique_sequences_from_folds,
    collate_multi_embedding_batch,
    load_final_training_data,
    load_fold_data,
    load_or_build_global_label_space,
    load_or_build_raw_label_space,
    load_protein_features_cache,
)
from training.data.embedding import (
    DEFAULT_ESM2_NAME,
    DEFAULT_MAX_LENGTH,
    DEFAULT_T5_NAME,
    extract_esm2_embeddings,
    extract_t5_embeddings,
    load_shard_index,
    pooled_embedding_exists,
)
from training.data.go_utils import parse_go_obo
from training.trainer import compute_multilabel_metrics, predict, train_one_epoch

DEFAULT_OUTPUT_DIR = ROOT_DIR / "models_raw"
DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"
DEFAULT_OBO_PATH = ROOT_DIR / "training" / "data" / "go-basic.obo"
DEFAULT_IC_PATH = ROOT_DIR / "training" / "data" / "ic.pkl"
DEFAULT_ESM2_FINAL_LAYER = 33
DEFAULT_PROTT5_LAYER = 24


def normalize_method(method: str) -> str:
    aliases = {
        "esm2": f"esm2-{DEFAULT_ESM2_FINAL_LAYER}",
    }
    normalized = aliases.get(method, method)
    if normalized.startswith("esm2-"):
        layer = normalized.removeprefix("esm2-")
        if not layer.isdigit():
            raise ValueError(f"ESM2 method must be esm2-<layer>, got {method}")
        return normalized
    if normalized in {"prott5", "blast"}:
        return normalized
    raise ValueError(f"Unsupported method: {method}")


def esm2_method_layer(method: str) -> int | None:
    if not method.startswith("esm2-"):
        return None
    return int(method.removeprefix("esm2-"))


def run_name(args: SimpleNamespace, fold_name: str) -> str:
    crafted = "crafted" if bool(args.use_crafted_features) else "no-crafted"
    scheduler = "cos" if args.lr_scheduler == "cosine" else "plateau"
    return f"{args.method}_{args.aspect}_{args.pooling}_{crafted}_{scheduler}_{fold_name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train neural and BLAST GO predictors")
    parser.add_argument("--method", default=None, help="Training method, e.g. esm2-33, esm2-20, prott5, or blast")
    parser.add_argument("--aspect", choices=["P", "F", "C"], default=None)
    parser.add_argument("--fold", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--pooling", choices=["both", "mean", "max"], default=None)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--oof-dir", type=Path, default=None)
    parser.add_argument("--obo", type=Path, default=DEFAULT_OBO_PATH)
    parser.add_argument("--ic-pkl", type=Path, default=DEFAULT_IC_PATH)
    parser.add_argument("--final", action="store_true", help="Train final model on propagated training.fasta/training.tsv without OOF")
    parser.add_argument("--no-crafted", action="store_true", help="Disable handcrafted protein features")
    parser.add_argument("--lr-scheduler", choices=["cosine", "plateau"], default="cosine")
    args = parser.parse_args()
    if args.method is not None and normalize_method(args.method) == "blast" and args.aspect is not None:
        parser.error("--method blast does not accept --aspect; it always generates P/F/C OOF")
    return args


def is_mps_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built())


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested cuda, but CUDA is not available")
    if requested == "mps" and not is_mps_available():
        raise RuntimeError("Requested mps, but MPS is not available")
    return torch.device(requested)


def apply_cli_overrides(config: Dict[str, object], args: argparse.Namespace) -> Dict[str, object]:
    resolved = dict(config)
    for key in ("method", "aspect", "fold", "epochs", "device", "pooling"):
        value = getattr(args, key, None)
        if value is not None:
            resolved[key] = value
    if args.batch_size is not None:
        resolved["batch_size"] = args.batch_size
    if args.threads is not None:
        resolved["blast_threads"] = args.threads
    if args.model_dir is not None:
        resolved["output_dir"] = args.model_dir
    if args.oof_dir is not None:
        resolved["oof_dir"] = args.oof_dir
    if args.ic_pkl is not None:
        resolved["ic_pkl"] = args.ic_pkl
    if args.final:
        resolved["final"] = True
    if args.no_crafted:
        resolved["use_crafted_features"] = False
    if args.lr_scheduler is not None:
        resolved["lr_scheduler"] = args.lr_scheduler
    resolved["method"] = normalize_method(str(resolved["method"]))
    return resolved


def namespace_from_config(config: Dict[str, object]) -> SimpleNamespace:
    args = SimpleNamespace(**config)
    args.device = resolve_device(str(args.device))
    args.output_dir = Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR))
    args.oof_dir = Path(getattr(args, "oof_dir", DEFAULT_OOF_DIR))
    args.ic_pkl = Path(getattr(args, "ic_pkl", DEFAULT_IC_PATH))
    return args


def load_information_content(ic_path: Path, aspect: str, classes: np.ndarray) -> np.ndarray:
    if not ic_path.exists():
        raise FileNotFoundError(f"IC pickle not found: {ic_path}")
    with ic_path.open("rb") as handle:
        payload = pickle.load(handle)
    ic_by_aspect = payload[-1] if isinstance(payload, tuple) else payload
    if aspect not in ic_by_aspect:
        raise ValueError(f"IC pickle missing aspect={aspect}: {ic_path}")
    aspect_ic = ic_by_aspect[aspect]
    missing = [str(term) for term in classes if str(term) not in aspect_ic]
    if missing:
        print(f"warning: IC pickle missing {len(missing)} {aspect} terms; using IC=0 for missing terms")
    return np.asarray([float(aspect_ic.get(str(term), 0.0)) for term in classes], dtype=np.float64)


def _pooling_names(pooling: str) -> list[str]:
    return ["mean", "max"] if pooling == "both" else [pooling]


def _embedding_exists(
    pid: str,
    plm: str,
    layer: str,
    pooling: str,
    shard_indices: Dict[str, Dict[str, str]] | None = None,
) -> bool:
    indices = shard_indices or {
        pooling_name: load_shard_index(EMBEDDING_DIR / plm / pooling_name / layer)
        for pooling_name in _pooling_names(pooling)
    }
    return all(
        pooled_embedding_exists(EMBEDDING_DIR, plm, pooling_name, int(layer), pid, indices[pooling_name])
        for pooling_name in _pooling_names(pooling)
    )


def ensure_embeddings(sequences_by_pid: Dict[str, str], batch_size: int, device: torch.device, method: str, pooling: str) -> None:
    needs_t5 = method == "prott5"
    esm2_layer = esm2_method_layer(method)
    esm2_indices = {
        pooling_name: load_shard_index(EMBEDDING_DIR / "esm2" / pooling_name / str(esm2_layer))
        for pooling_name in _pooling_names(pooling)
    } if esm2_layer is not None else None
    t5_indices = {
        pooling_name: load_shard_index(EMBEDDING_DIR / "prott5" / pooling_name / str(DEFAULT_PROTT5_LAYER))
        for pooling_name in _pooling_names(pooling)
    } if needs_t5 else None
    missing_esm2 = [
        pid for pid in sequences_by_pid
        if esm2_layer is not None and not _embedding_exists(pid, "esm2", str(esm2_layer), pooling, esm2_indices)
    ]
    missing_t5 = [
        pid for pid in sequences_by_pid if not _embedding_exists(pid, "prott5", str(DEFAULT_PROTT5_LAYER), pooling, t5_indices)
    ] if needs_t5 else []
    if missing_esm2:
        extract_esm2_embeddings(
            sequences_by_pid={pid: sequences_by_pid[pid] for pid in missing_esm2},
            output_dir=EMBEDDING_DIR,
            pretrained_name=DEFAULT_ESM2_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
            layer_indices=[esm2_layer] if esm2_layer is not None else None,
            pooling=pooling,
        )
    if missing_t5:
        extract_t5_embeddings(
            sequences_by_pid={pid: sequences_by_pid[pid] for pid in missing_t5},
            output_dir=EMBEDDING_DIR,
            pretrained_name=DEFAULT_T5_NAME,
            batch_size=batch_size,
            max_length=DEFAULT_MAX_LENGTH,
            device=device,
            pooling=pooling,
            layer_indices=[DEFAULT_PROTT5_LAYER],
        )


def load_or_build_features(folds: Sequence[int]) -> Dict[str, torch.Tensor]:
    return load_or_build_features_for_sequences(collect_unique_sequences_from_folds(folds))


def load_or_build_features_for_sequences(sequences: Dict[str, str]) -> Dict[str, torch.Tensor]:
    path = PROTEIN_FEATURES_DIR / "protein_features.pt"
    if path.exists():
        cache = load_protein_features_cache(path)
        missing = {pid: seq for pid, seq in sequences.items() if pid not in cache}
        if not missing:
            return cache
        cache.update({pid: build_sequence_protein_features(seq) for pid, seq in missing.items()})
        torch.save(cache, path)
        return cache
    return build_and_save_protein_features(sequences, path)


def save_oof(prefix: Path, pids: np.ndarray, labels: np.ndarray, probs: np.ndarray, classes: np.ndarray, metrics: dict) -> None:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    path = prefix.with_suffix(".npz")
    np.savez(
        path,
        pids=pids,
        labels=labels,
        probs=probs,
        classes=classes,
        metrics_json=np.array(json.dumps(metrics)),
    )


def run_neural_fold(args: SimpleNamespace, fold_data, protein_features_cache: Dict[str, torch.Tensor]) -> dict:
    information_content = load_information_content(args.ic_pkl, args.aspect, fold_data.classes)
    train_dataset = MultiEmbeddingDataset(
        fold_data.train_pids, fold_data.train_matrix, EMBEDDING_DIR, protein_features_cache,
        chain=args.method, pooling=args.pooling, use_crafted_features=args.use_crafted_features,
    )
    val_dataset = MultiEmbeddingDataset(
        fold_data.val_pids, fold_data.val_matrix, EMBEDDING_DIR, protein_features_cache,
        chain=args.method, pooling=args.pooling, use_crafted_features=args.use_crafted_features,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_multi_embedding_batch, pin_memory=args.device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size * 2, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate_multi_embedding_batch, pin_memory=args.device.type == "cuda",
    )
    plm_key = "esm2" if args.method.startswith("esm2-") else args.method
    model = build_model(args, num_classes=len(fold_data.classes), embedding_dim=EMBEDDING_DIMS[plm_key]).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        fused=(args.device.type == "cuda")
    )
    use_amp = args.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler}")
    print(f"fold={fold_data.fold_dir.name} lr_scheduler={args.lr_scheduler}")
    best_state = None
    best_fmax = -1.0
    best_metrics = {}
    best_epoch = 0
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        result = train_one_epoch(
            model, train_loader, optimizer, args.device, f"fold {fold_data.fold_dir.name} epoch {epoch}",
            scaler=scaler, use_amp=use_amp,
        )
        predictions = predict(model, val_loader, args.device, "validation", use_amp=use_amp)
        metrics = compute_multilabel_metrics(predictions["labels"], predictions["probs"], args.threshold)
        fmax = float(metrics["fmax"])
        print(
            f"fold={fold_data.fold_dir.name} epoch={epoch} lr={epoch_lr:.2e} "
            f"loss={result.loss:.4f} fmax={fmax:.4f} "
            f"fmax_threshold={metrics['fmax_threshold']:.2f}"
        )
        if fmax > best_fmax + args.early_stop_min_delta:
            best_epoch = epoch
            best_fmax = fmax
            best_metrics = dict(metrics)
            epochs_without_improvement = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
        if args.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(fmax)
        next_lr = float(optimizer.param_groups[0]["lr"])
        if args.lr_scheduler == "plateau" and next_lr < epoch_lr:
            print(f"fold={fold_data.fold_dir.name} epoch={epoch} reduce_lr={next_lr:.2e}")
        if epochs_without_improvement >= args.early_stop_patience:
            print(
                f"fold={fold_data.fold_dir.name} early_stop epoch={epoch} "
                f"best_epoch={best_epoch} best_fmax={best_fmax:.4f}"
            )
            break
    if best_state is None:
        raise RuntimeError("No checkpoint was produced; check epochs")
    model.load_state_dict(best_state)
    predictions = predict(model, val_loader, args.device, "final validation", use_amp=use_amp)
    metrics = compute_multilabel_metrics(
        predictions["labels"], predictions["probs"], args.threshold, information_content=information_content
    )
    metrics = dict(metrics)
    metrics["best_epoch"] = best_epoch
    checkpoint = {
        "model_state_dict": best_state,
        "classes": fold_data.classes,
        "args": vars(args).copy() | {"device": str(args.device), "output_dir": str(args.output_dir), "oof_dir": str(args.oof_dir)},
        "metrics": metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "aspect": args.aspect,
        "method": args.method,
    }
    name = run_name(args, fold_data.fold_dir.name)
    checkpoint["run_name"] = name
    model_path = args.output_dir / args.method / f"{name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    save_oof(
        args.oof_dir / args.method / name,
        predictions["pids"], predictions["labels"], predictions["probs"], fold_data.classes, metrics,
    )
    return metrics


def run_neural_final(args: SimpleNamespace, final_data, protein_features_cache: Dict[str, torch.Tensor]) -> dict:
    train_dataset = MultiEmbeddingDataset(
        final_data.train_pids, final_data.train_matrix, EMBEDDING_DIR, protein_features_cache,
        chain=args.method, pooling=args.pooling, use_crafted_features=args.use_crafted_features,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        collate_fn=collate_multi_embedding_batch, pin_memory=args.device.type == "cuda",
    )
    plm_key = "esm2" if args.method.startswith("esm2-") else args.method
    model = build_model(args, num_classes=len(final_data.classes), embedding_dim=EMBEDDING_DIMS[plm_key]).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=(args.device.type == "cuda"),
    )
    use_amp = args.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.lr_factor,
            patience=args.lr_patience,
            min_lr=args.min_lr,
        )
    else:
        raise ValueError(f"Unsupported lr_scheduler={args.lr_scheduler}")
    print(f"final lr_scheduler={args.lr_scheduler}")
    final_state = None
    final_loss = 0.0
    for epoch in range(1, args.epochs + 1):
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        result = train_one_epoch(
            model, train_loader, optimizer, args.device, f"final epoch {epoch}",
            scaler=scaler, use_amp=use_amp,
        )
        final_loss = float(result.loss)
        print(f"final epoch={epoch} lr={epoch_lr:.2e} loss={final_loss:.4f}")
        final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if args.lr_scheduler == "cosine":
            scheduler.step()
        else:
            scheduler.step(final_loss)
    if final_state is None:
        raise RuntimeError("No checkpoint was produced; check epochs")
    metrics = {"train_loss": final_loss, "epochs": int(args.epochs)}
    checkpoint = {
        "model_state_dict": final_state,
        "classes": final_data.classes,
        "args": vars(args).copy() | {"device": str(args.device), "output_dir": str(args.output_dir), "oof_dir": str(args.oof_dir)},
        "metrics": metrics,
        "best_epoch": int(args.epochs),
        "best_metrics": metrics,
        "aspect": args.aspect,
        "method": args.method,
    }
    name = run_name(args, "final")
    checkpoint["run_name"] = name
    model_path = args.output_dir / args.method / f"{name}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, model_path)
    return metrics


def run_blast_fold(args: SimpleNamespace, fold_data) -> dict:
    _require_blast()
    cache_dir = fold_data.fold_dir / "blast_cache"
    db_prefix = _build_database(fold_data.fold_dir / "train.fasta", cache_dir)
    output_path = cache_dir / "val_vs_train.tsv"
    if not output_path.exists():
        _run_blast(
            fold_data.fold_dir / "val.fasta",
            db_prefix,
            output_path,
            max_hits=args.blast_top_k,
            evalue=1e-3,
            num_threads=getattr(args, "blast_threads", 8),
        )
    hits = _parse_blast_hits(output_path)
    pids = np.asarray(fold_data.val_pids)
    probs = _transfer_scores(pids, hits, fold_data.train_labels_df, fold_data.classes)
    labels = fold_data.val_matrix.toarray().astype(np.float32)
    information_content = load_information_content(args.ic_pkl, args.aspect, fold_data.classes)
    metrics = compute_multilabel_metrics(labels, probs, args.threshold, information_content=information_content)
    save_oof(args.oof_dir / "blast" / f"blast_{args.aspect}_{fold_data.fold_dir.name}", pids, labels, probs, fold_data.classes, metrics)
    print(
        f"fold={fold_data.fold_dir.name} blast fmax={metrics['fmax']:.4f} "
        f"fmax_threshold={metrics['fmax_threshold']:.2f}"
    )
    return metrics


def run_training_job(config: Dict[str, object], obo_path: Path) -> dict:
    args = namespace_from_config(config)
    parents = parse_go_obo(obo_path)
    if bool(getattr(args, "final", False)):
        if args.method == "blast":
            raise ValueError("--final is only supported for neural methods; BLAST does not produce a model checkpoint")
        classes = load_or_build_raw_label_space(
            aspect=args.aspect, parents=parents, min_count=int(args.min_count)
        )
    else:
        classes = load_or_build_global_label_space(
            folds=args.fold, aspect=args.aspect, parents=parents, min_count=int(args.min_count)
        )
    if len(classes) == 0:
        raise RuntimeError(f"Empty label space for aspect={args.aspect}, min_count={args.min_count}")
    if bool(getattr(args, "final", False)):
        sequences = collect_unique_sequences_from_raw()
        ensure_embeddings(sequences, args.batch_size, args.device, args.method, args.pooling)
        protein_features_cache = load_or_build_features_for_sequences(sequences) if args.use_crafted_features else {}
        metrics = run_neural_final(args, load_final_training_data(aspect=args.aspect, parents=parents, classes=classes), protein_features_cache)
        print(json.dumps(metrics, indent=2, sort_keys=True))
        return metrics
    if args.method != "blast":
        sequences = collect_unique_sequences_from_folds(args.fold)
        ensure_embeddings(sequences, args.batch_size, args.device, args.method, args.pooling)
        protein_features_cache = load_or_build_features(args.fold) if args.use_crafted_features else {}
    else:
        protein_features_cache = {}
    metrics_by_fold = []
    for fold in args.fold:
        fold_data = load_fold_data(fold=fold, aspect=args.aspect, parents=parents, classes=classes)
        if args.method == "blast":
            metrics_by_fold.append(run_blast_fold(args, fold_data))
        else:
            metrics_by_fold.append(run_neural_fold(args, fold_data, protein_features_cache))
    summary = {
        "avg_fmax": float(np.mean([item["fmax"] for item in metrics_by_fold])),
        "avg_smin": float(np.mean([item["smin"] for item in metrics_by_fold])),
        "avg_aupr": float(np.mean([item["aupr"] for item in metrics_by_fold])),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def run_blast_all_aspects(cli_args: argparse.Namespace, obo_path: Path) -> dict:
    from training.hparams import resolve_matching_training_run

    summaries = {}
    for aspect in ("P", "F", "C"):
        base = resolve_matching_training_run("blast", aspect, cli_args.lr_scheduler)
        config = apply_cli_overrides(base, cli_args)
        config["aspect"] = aspect
        summaries[aspect] = run_training_job(config, obo_path)
    return summaries


def main() -> None:
    from training.hparams import get_training_runs, resolve_matching_training_run

    cli_args = parse_args()
    cli_method = normalize_method(cli_args.method) if cli_args.method is not None else None
    if cli_method == "blast":
        run_blast_all_aspects(cli_args, cli_args.obo)
        return
    if cli_args.method or cli_args.aspect or cli_args.final:
        base = resolve_matching_training_run(
            cli_args.method or f"esm2-{DEFAULT_ESM2_FINAL_LAYER}", cli_args.aspect or "P", cli_args.lr_scheduler
        )
        run_training_job(apply_cli_overrides(base, cli_args), cli_args.obo)
        return
    for config in get_training_runs(cli_args.lr_scheduler):
        run_training_job(apply_cli_overrides(config, cli_args), cli_args.obo)


if __name__ == "__main__":
    main()
