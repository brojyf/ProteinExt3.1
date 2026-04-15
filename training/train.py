from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch
from torch.utils.data import DataLoader

DEFAULT_OUTPUT_DIR = ROOT_DIR / "models"
DEFAULT_OOF_DIR = ROOT_DIR / "training" / "oof"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ProteinExt3 training pipeline. CLI args override hparams.py defaults."
    )
    parser.add_argument("--method", choices=["cnn", "esm2", "t5", "blast"])
    parser.add_argument("--aspect", choices=["P", "F", "C"])
    parser.add_argument("--batch-size", dest="batch_size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--fold", type=int, nargs="+")
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


def print_device_summary(device: torch.device) -> None:
    cuda_available = torch.cuda.is_available()
    mps_available = is_mps_available()
    print(
        "Device Check | "
        f"cuda_available={cuda_available} | "
        f"mps_available={mps_available} | "
        "cpu_available=True"
    )
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
        print(f"Using device: cuda ({device_name})")
    elif device.type == "mps":
        print("Using device: mps (Apple Metal)")
    else:
        print("Using device: cpu")


def namespace_from_config(config: dict) -> SimpleNamespace:
    resolved = dict(config)
    resolved["output_dir"] = DEFAULT_OUTPUT_DIR
    resolved["oof_dir"] = DEFAULT_OOF_DIR
    resolved["device"] = resolve_device(str(config.get("device", "auto")))
    return SimpleNamespace(**resolved)


def apply_cli_overrides(run_config: dict, cli_args: argparse.Namespace) -> dict:
    resolved = dict(run_config)
    for key in ("method", "aspect", "batch_size", "epochs", "fold"):
        value = getattr(cli_args, key, None)
        if value is not None:
            resolved[key] = value
    return resolved


def build_datasets(args: SimpleNamespace, fold_data):
    from training.data.data_utils import EMBEDDING_DIR, ProteinTokenEmbeddingDataset
    from training.data.embedding import DEFAULT_ESM2_INTERMEDIATE_LAYER, esm2_layer_dir

    if args.method == "t5":
        dataset_kwargs = {"plm": "t5", "layer": "last", "hydro_window_size": None}
    elif args.method == "esm2":
        dataset_kwargs = {"plm": "esm2", "layer": "last", "hydro_window_size": None}
    elif args.method == "cnn":
        dataset_kwargs = {
            "plm": "esm2",
            "layer": esm2_layer_dir(DEFAULT_ESM2_INTERMEDIATE_LAYER),
            "hydro_window_size": args.window_size,
        }
    else:
        raise ValueError(f"Unsupported training method: {args.method}")

    train_dataset = ProteinTokenEmbeddingDataset(
        pids=fold_data.train_pids,
        sequences=fold_data.train_sequences,
        labels=fold_data.train_matrix,
        embedding_dir=EMBEDDING_DIR,
        **dataset_kwargs,
    )
    val_dataset = ProteinTokenEmbeddingDataset(
        pids=fold_data.val_pids,
        sequences=fold_data.val_sequences,
        labels=fold_data.val_matrix,
        embedding_dir=EMBEDDING_DIR,
        **dataset_kwargs,
    )
    return train_dataset, val_dataset


def build_loaders(args: SimpleNamespace, fold_data):
    from training.data.data_utils import collate_token_embedding_batch

    train_dataset, val_dataset = build_datasets(args, fold_data)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_token_embedding_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_token_embedding_batch,
    )
    return train_loader, val_loader


def serialize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {key: serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_value(item) for item in value]
    return value


def save_outputs(
    output_dir: Path,
    run_prefix: str,
    payload: dict,
    classes: np.ndarray,
    checkpoint: dict | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"{run_prefix}_pids.npy", payload["pids"])
    np.save(output_dir / f"{run_prefix}_labels.npy", payload["labels"])
    np.save(output_dir / f"{run_prefix}_probs.npy", payload["probs"])
    np.save(output_dir / f"{run_prefix}_classes.npy", classes)

    metrics_path = output_dir / f"{run_prefix}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload["metrics"], handle, indent=2, sort_keys=True)

    if checkpoint is not None:
        torch.save(checkpoint, output_dir / f"{run_prefix}.pt")


def build_optimizer(model: torch.nn.Module, args: SimpleNamespace) -> torch.optim.Optimizer:
    optimizer_config = dict(args.optimizer)
    trainable_params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_params:
        raise RuntimeError(f"No trainable parameters found for method={args.method}")

    if args.method in {"esm2", "t5"}:
        tracked_ids = set()
        param_groups = []

        if args.pooling == "attention" and hasattr(model, "pooler"):
            attention_params = [parameter for parameter in model.pooler.parameters() if parameter.requires_grad]
            if attention_params:
                param_groups.append(
                    {
                        "params": attention_params,
                        "lr": float(optimizer_config["attention_lr"]),
                        "name": "attention",
                    }
                )
                tracked_ids.update(id(parameter) for parameter in attention_params)

        if hasattr(model, "head"):
            classifier_params = [parameter for parameter in model.head.parameters() if parameter.requires_grad]
            if classifier_params:
                param_groups.append(
                    {
                        "params": classifier_params,
                        "lr": float(optimizer_config["classifier_lr"]),
                        "name": "classifier",
                    }
                )
                tracked_ids.update(id(parameter) for parameter in classifier_params)

        remaining_params = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in tracked_ids
        ]
        if remaining_params:
            fallback_lr = float(
                optimizer_config.get(
                    "classifier_lr",
                    optimizer_config.get("attention_lr", optimizer_config.get("lr", 1e-3)),
                )
            )
            param_groups.append(
                {
                    "params": remaining_params,
                    "lr": fallback_lr,
                    "name": "remaining",
                }
            )
    else:
        param_groups = [
            {
                "params": trainable_params,
                "lr": float(optimizer_config["lr"]),
                "name": "model",
            }
        ]

    return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    args: SimpleNamespace,
    num_training_steps: int,
) -> tuple[torch.optim.lr_scheduler.LambdaLR | None, int]:
    if num_training_steps <= 0:
        return None, 0

    scheduler_config = dict(args.scheduler)
    warmup_ratio = float(scheduler_config.get("warmup_ratio", 0.05))
    min_lr_ratio = float(scheduler_config.get("min_lr_ratio", 0.0))
    warmup_steps = int(round(num_training_steps * warmup_ratio))

    def lr_lambda(current_step: int) -> float:
        if warmup_steps > 0 and current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_scale

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda), warmup_steps


def current_lrs(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    lrs = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        lrs[name] = float(group["lr"])
    return lrs


def format_lrs(optimizer: torch.optim.Optimizer) -> str:
    return ", ".join(f"{name}={lr:.2e}" for name, lr in current_lrs(optimizer).items())


def run_fold(args: SimpleNamespace, fold: int) -> dict:
    from training.data.data_utils import load_fold_data
    from submethods import MODEL_BUILDERS
    from training.trainer import compute_multilabel_metrics, predict, train_one_epoch

    fold_data = load_fold_data(fold=fold, aspect=args.aspect)
    device = args.device

    # AMP + gradient accumulation setup
    use_amp = getattr(args, "use_amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    grad_accum = max(getattr(args, "gradient_accumulation_steps", 1), 1)
    focal_gamma = float(getattr(args, "focal_gamma", 0.0))
    label_smoothing = float(getattr(args, "label_smoothing", 0.0))
    max_grad_norm = float(getattr(args, "max_grad_norm", 0.0))

    model = MODEL_BUILDERS[args.method](args, num_classes=len(fold_data.classes)).to(device)
    train_loader, val_loader = build_loaders(args, fold_data)
    optimizer = build_optimizer(model, args)

    # Scheduler counts optimizer steps, not raw batch steps
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    total_steps = args.epochs * steps_per_epoch
    scheduler, warmup_steps = build_scheduler(optimizer, args, total_steps)

    print("=" * 80)
    print(
        f"Fold {fold} | method={args.method} | aspect={args.aspect} | "
        f"train={len(fold_data.train_pids)} | val={len(fold_data.val_pids)} | classes={len(fold_data.classes)}"
    )
    print(
        f"  optimizer_lrs: {format_lrs(optimizer)} | "
        f"scheduler=warmup+cosine | warmup_steps={warmup_steps} | total_steps={total_steps}"
    )
    print(
        f"  amp={use_amp} | grad_accum={grad_accum} | max_grad_norm={max_grad_norm} | "
        f"focal_gamma={focal_gamma} | label_smoothing={label_smoothing}"
    )

    best_fmax = -1.0
    best_epoch = 0
    best_state_dict: dict | None = None

    history: List[dict] = []
    for epoch in range(1, args.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            go_term_loss_weight=float(args.go_term_loss_weight),
            progress_desc=f"Fold {fold} Epoch {epoch}/{args.epochs}",
            scaler=scaler,
            gradient_accumulation_steps=grad_accum,
            focal_gamma=focal_gamma,
            label_smoothing=label_smoothing,
            max_grad_norm=max_grad_norm,
        )
        epoch_predictions = predict(
            model,
            val_loader,
            device,
            progress_desc=f"Fold {fold} Epoch {epoch}/{args.epochs} Validation",
            use_amp=use_amp,
        )
        epoch_metrics = compute_multilabel_metrics(
            y_true=epoch_predictions["labels"],
            y_prob=epoch_predictions["probs"],
            threshold=args.threshold,
            progress_desc=f"Fold {fold} Epoch {epoch}/{args.epochs} Fmax",
        )
        epoch_lrs = current_lrs(optimizer)

        is_best = epoch_metrics["fmax"] > best_fmax
        if is_best:
            best_fmax = epoch_metrics["fmax"]
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_result.loss,
                "train_bce_loss": train_result.bce_loss,
                "train_go_term_loss": train_result.go_term_loss,
                "train_go_term_soft_f1": train_result.go_term_soft_f1,
                "val_metrics": epoch_metrics,
                "lr": epoch_lrs,
                "is_best": is_best,
            }
        )
        print(
            f"  fold {fold} | epoch {epoch:02d}{' *' if is_best else '  '} | train_loss={train_result.loss:.4f} | "
            f"train_bce={train_result.bce_loss:.4f} | "
            f"train_go_loss={train_result.go_term_loss:.4f} | "
            f"train_go_f1={train_result.go_term_soft_f1:.4f} | "
            f"val_micro_f1={epoch_metrics['micro_f1']:.4f} | "
            f"val_precision={epoch_metrics['micro_precision']:.4f} | "
            f"val_recall={epoch_metrics['micro_recall']:.4f} | "
            f"fmax={epoch_metrics['fmax']:.4f} @ {epoch_metrics['fmax_threshold']:.2f} | "
            f"lrs={', '.join(f'{name}={lr:.2e}' for name, lr in epoch_lrs.items())}"
        )

    # Restore best epoch weights before final evaluation
    print(f"  Restoring best weights from epoch {best_epoch} (fmax={best_fmax:.4f})")
    model.load_state_dict(best_state_dict)

    predictions = predict(model, val_loader, device, progress_desc=f"Fold {fold} Final Validation", use_amp=use_amp)
    model_probs = predictions["probs"]

    metrics = compute_multilabel_metrics(
        y_true=predictions["labels"],
        y_prob=model_probs,
        threshold=args.threshold,
        progress_desc=f"Fold {fold} Final Fmax",
    )
    metrics["num_classes"] = int(len(fold_data.classes))
    metrics["num_val_proteins"] = int(len(fold_data.val_pids))

    serialized_args = serialize_value(vars(args))
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "classes": fold_data.classes,
        "args": serialized_args,
        "config": serialized_args,
        "history": history,
        "fold": fold,
        "aspect": args.aspect,
        "method": args.method,
        "metrics": metrics,
    }

    payload = {
        "pids": predictions["pids"],
        "labels": predictions["labels"],
        "probs": model_probs,
        "metrics": metrics,
        "history": history,
    }
    save_outputs(
        output_dir=args.oof_dir,
        run_prefix=f"{args.method}_{args.aspect}_fold{fold}",
        payload=payload,
        classes=fold_data.classes,
    )
    if args.save_fold_artifacts:
        save_outputs(
            output_dir=args.output_dir,
            run_prefix=f"{args.method}_{args.aspect}_fold{fold}",
            payload=payload,
            classes=fold_data.classes,
        )
    print(
        f"  val micro_f1={metrics['micro_f1']:.4f} | "
        f"precision={metrics['micro_precision']:.4f} | recall={metrics['micro_recall']:.4f}"
    )
    print("=" * 80)
    return {
        "fold": fold,
        "metrics": metrics,
        "payload": payload,
        "classes": fold_data.classes,
        "checkpoint": checkpoint,
    }


def run_blast_fold(args: SimpleNamespace, fold: int) -> dict:
    from training.data.data_utils import load_fold_data
    from submethods.bp_blast_transfer import (
        _build_database,
        _parse_blast_hits,
        _require_blast,
        _run_blast,
        _transfer_scores,
    )
    from training.trainer import compute_multilabel_metrics

    _require_blast()
    fold_data = load_fold_data(fold=fold, aspect=args.aspect)

    blast_cache = fold_data.fold_dir / "blast_cache"
    db_prefix = _build_database(fold_data.fold_dir / "train.fasta", blast_cache)
    output_path = blast_cache / "val_vs_train.tsv"
    _run_blast(fold_data.fold_dir / "val.fasta", db_prefix, output_path, max_hits=10, evalue=1e-3)
    hits = _parse_blast_hits(output_path)

    val_pids = np.asarray(fold_data.val_pids)
    blast_scores = _transfer_scores(val_pids, hits, fold_data.train_labels_df, fold_data.classes)
    val_labels = fold_data.val_matrix.toarray().astype(np.float32)

    metrics = compute_multilabel_metrics(
        y_true=val_labels,
        y_prob=blast_scores,
        threshold=args.threshold,
        progress_desc=f"Fold {fold} BLAST Fmax",
    )
    metrics["num_classes"] = int(len(fold_data.classes))
    metrics["num_val_proteins"] = int(len(fold_data.val_pids))

    print("=" * 80)
    print(
        f"Fold {fold} | method=blast | aspect={args.aspect} | "
        f"val={len(fold_data.val_pids)} | classes={len(fold_data.classes)}"
    )

    payload = {
        "pids": val_pids,
        "labels": val_labels,
        "probs": blast_scores,
        "metrics": metrics,
    }
    save_outputs(
        output_dir=args.oof_dir,
        run_prefix=f"blast_{args.aspect}_fold{fold}",
        payload=payload,
        classes=fold_data.classes,
    )

    print(
        f"  val micro_f1={metrics['micro_f1']:.4f} | "
        f"fmax={metrics['fmax']:.4f} @ {metrics['fmax_threshold']:.2f}"
    )
    print("=" * 80)
    return {
        "fold": fold,
        "metrics": metrics,
        "payload": payload,
        "classes": fold_data.classes,
        "checkpoint": None,
    }


def summarize_metrics(fold_results: List[dict]) -> dict:
    summary = {}
    for key in ("micro_f1", "micro_precision", "micro_recall"):
        values = [result["metrics"][key] for result in fold_results]
        summary[f"avg_{key}"] = float(np.mean(values)) if values else 0.0
    summary["best_fold"] = max(fold_results, key=lambda result: result["metrics"]["micro_f1"])["fold"]
    return summary


def save_best_result(args: SimpleNamespace, fold_results: List[dict], summary: dict) -> None:
    best_result = max(fold_results, key=lambda result: result["metrics"]["micro_f1"])
    best_prefix = f"best_{args.method}_{args.aspect}"
    best_output = {
        **best_result["payload"],
        "metrics": {
            **best_result["metrics"],
            "selected_as_best_by": "micro_f1",
            "cv_summary": summary,
        },
    }
    save_outputs(
        output_dir=args.output_dir,
        run_prefix=best_prefix,
        payload=best_output,
        classes=best_result["classes"],
        checkpoint=best_result["checkpoint"],
    )

    summary_path = args.output_dir / f"{best_prefix}_cv_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


def run_training_job(config: dict) -> dict:
    args = namespace_from_config(config)
    print()
    print(f"Training Job | method={args.method} | aspect={args.aspect}")
    if args.method != "blast":
        print_device_summary(args.device)

    fold_results = []
    for fold in args.fold:
        if args.method == "blast":
            fold_results.append(run_blast_fold(args, fold))
        else:
            fold_results.append(run_fold(args, fold))

    summary = summarize_metrics(fold_results)
    if args.method != "blast":
        save_best_result(args, fold_results, summary)
    print("Summary:", json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    from training.hparams import get_training_runs, resolve_training_run

    cli_args = parse_args()
    training_runs = get_training_runs()
    if not training_runs:
        raise RuntimeError("No training runs configured in ProteinExt3/train/hparams.py.")

    if cli_args.method or cli_args.aspect or cli_args.batch_size or cli_args.epochs or cli_args.fold:
        base_method = cli_args.method or training_runs[0].get("method", "esm2")
        base_aspect = cli_args.aspect or training_runs[0].get("aspect", "P")
        base_config = resolve_training_run({"method": base_method, "aspect": base_aspect})
        
        run_training_job(apply_cli_overrides(base_config, cli_args))
        return

    for run_config in training_runs:
        run_training_job(run_config)


if __name__ == "__main__":
    main()
