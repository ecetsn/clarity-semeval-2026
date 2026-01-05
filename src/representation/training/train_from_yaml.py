# src/representation/training/train_from_yaml.py
# ------------------------------------------------------------
# YAML-driven training entry point for QEvasion representation-fusion models.
# Trains on the QEvasion train split with an internal validation split, selects best checkpoint by macro-F1,
# and saves per-epoch validation JSONs plus a final summary under an experiment output directory.
# ------------------------------------------------------------

from __future__ import annotations  # Enables forward references in type hints (useful in older Python versions).

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

# Repo components: data module, model factories, evaluation, logging, and fusion model.
from src.representation.data.qev_datamodule import QEvasionDataModule
from src.representation.decoders.decoder_factory import build_decoder
from src.representation.encoders.encoder_factory import build_encoder
from src.representation.evaluation.evaluator import evaluate_predictions
from src.representation.evaluation.loss_factory import build_loss
from src.representation.logging.logger_setup import init_logger
from src.representation.models.fusion_model import RepresentationFusionModel
from src.representation.projections.projection_factory import build_projection



# GLOBALS
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Project root inferred from script location.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # Primary device selection.
AMP_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"  # autocast() expects a device_type string.



# DATASET
class TextLabelDataset(Dataset):
    """
    Minimal dataset that yields (text, label_id) pairs for classification.
    Text is kept as raw string; encoder/decoder do tokenization internally.
    """
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        # Enforce alignment between examples and labels to avoid silent metric corruption.
        if len(texts) != len(labels):
            raise ValueError(f"texts/labels length mismatch: {len(texts)} vs {len(labels)}")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        # Required by PyTorch Dataset.
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        # Return label as torch.long because CrossEntropyLoss expects integer class indices.
        return self.texts[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)


# UTILS
def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducible training and validation splits."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML config file and assert the root is a mapping."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return cfg


def build_label_mapping(labels: List[str]) -> Dict[str, int]:
    """Create a stable label->id mapping from the (train) label set."""
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}


def to_json_safe(obj: Any) -> Any:
    """Convert common NumPy/PyTorch types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write pretty JSON to disk (and create parent folders if needed)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, ensure_ascii=False)


def maybe_freeze(module: nn.Module) -> None:
    """
    Freeze a module so it does not receive gradients.
    If the module defines a custom freeze() method, prefer it for correctness.
    """
    if hasattr(module, "freeze") and callable(getattr(module, "freeze")):
        module.freeze()
        return
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def has_trainable_params(module: nn.Module) -> bool:
    """Return True if at least one parameter requires gradients."""
    return any(p.requires_grad for p in module.parameters())


def infer_hidden_dim(x: torch.Tensor) -> int:
    """
    Infer embedding dimension from a pooled embedding tensor.
    Expected input shape is (B, H).
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (B,H) embedding tensor, got shape {tuple(x.shape)}")
    return int(x.shape[-1])


def get_required(cfg: Dict[str, Any], key_path: str) -> Any:
    """
    Fetch a nested YAML key and fail fast if it is missing.
    Example key_path: "experiment.name"
    """
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {key_path}")
        cur = cur[part]
    return cur


def collect_trainable_params(*modules: nn.Module) -> List[nn.Parameter]:
    """Collect all parameters with requires_grad=True across provided modules."""
    params: List[nn.Parameter] = []
    for m in modules:
        params.extend([p for p in m.parameters() if p.requires_grad])
    return params


# TRAIN / EVAL
def train_one_epoch(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    fusion_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch: int,
    cfg: Dict[str, Any],
    logger,
    train_encoder: bool,
    train_decoder: bool,
    clip_params: List[nn.Parameter],
) -> float:
    """
    Single training epoch with gradient accumulation, AMP, and optional grad clipping.
    Encoder/decoder can be frozen or trainable depending on YAML flags and LoRA settings.
    """
    # Fusion head always trains; encoder/decoder training depends on train_encoder/train_decoder flags.
    fusion_model.train()
    if train_encoder:
        encoder.train()
    else:
        encoder.eval()
    if train_decoder:
        decoder.train()
    else:
        decoder.eval()

    # Start each epoch with cleared gradients.
    optimizer.zero_grad(set_to_none=True)

    # Training knobs are read from YAML to keep the script generic.
    use_amp = bool(cfg["training"].get("use_amp", True))
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 0.0))

    total_loss = 0.0
    num_batches = 0

    # Each batch yields raw texts and integer labels.
    for step, (texts, labels) in enumerate(loader, start=1):
        y = labels.to(device=DEVICE, dtype=torch.long)
        texts_list = list(texts)

        # Compute encoder/decoder embeddings; if frozen, run under no_grad for speed/memory.
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            if train_encoder:
                enc_vec = encoder(texts_list)
            else:
                with torch.no_grad():
                    enc_vec = encoder(texts_list)

            if train_decoder:
                dec_vec = decoder(texts_list)
            else:
                with torch.no_grad():
                    dec_vec = decoder(texts_list)

        # Forward through fusion head and compute classification loss.
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            logits = fusion_model(enc_vec, dec_vec)  # Shape: (B, num_classes)
            loss = criterion(logits, y) / max(1, grad_accum)  # Scale for gradient accumulation.

        # Backprop with AMP scaling.
        scaler.scale(loss).backward()

        # Perform an optimizer step every grad_accum micro-batches.
        if step % grad_accum == 0:
            if max_grad_norm and max_grad_norm > 0:
                # Unscale grads before clipping so clipping threshold is in true scale.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * max(1, grad_accum)
        num_batches += 1

        # Periodic logging for long epochs.
        if step % int(cfg["training"].get("log_every_steps", 50)) == 0:
            logger.info(
                f"[Epoch {epoch}] step={step}/{len(loader)} "
                f"train_loss={total_loss / max(1, num_batches):.4f}"
            )

    # If the epoch ends mid-accumulation, flush the leftover gradients with one final step.
    if (len(loader) % grad_accum) != 0:
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    # Return average loss for reporting.
    return total_loss / max(1, num_batches)


@torch.no_grad()
def run_validation(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    fusion_model: nn.Module,
    loader: DataLoader,
    id_to_label: Dict[int, str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run deterministic validation: set all modules to eval(), compute predictions, and report metrics.
    Mode is restored afterwards so training can continue normally.
    """
    # Snapshot current train/eval state so we can restore it after validation.
    enc_was_train = encoder.training
    dec_was_train = decoder.training
    fusion_was_train = fusion_model.training

    # Force eval to disable dropout and other training-time randomness.
    encoder.eval()
    decoder.eval()
    fusion_model.eval()

    use_amp = bool(cfg["training"].get("use_amp", True))

    y_true_ids: List[int] = []
    y_pred_ids: List[int] = []

    # Iterate over the validation set and collect predicted class ids.
    for texts, labels in loader:
        y_true_ids.extend(labels.detach().cpu().tolist())
        texts_list = list(texts)

        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            enc_vec = encoder(texts_list)
            dec_vec = decoder(texts_list)
            logits = fusion_model(enc_vec, dec_vec)

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        y_pred_ids.extend([int(p) for p in preds])

    # Convert integer ids back to label strings for the evaluator.
    label_list = [id_to_label[i] for i in sorted(id_to_label.keys())]
    y_true = [id_to_label[i] for i in y_true_ids]
    y_pred = [id_to_label[i] for i in y_pred_ids]

    # Compute metrics and normalized confusion matrix via repo evaluator.
    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    # Store a small head sample for quick debugging.
    result["y_true_ids_head"] = y_true_ids[:20]
    result["y_pred_ids_head"] = y_pred_ids[:20]

    # Restore original module modes to avoid interfering with training.
    if enc_was_train:
        encoder.train()
    if dec_was_train:
        decoder.train()
    if fusion_was_train:
        fusion_model.train()

    return result


# ============================================================
#                        MAIN
# ============================================================

def main() -> None:
    # CLI expects a single positional argument: the YAML config path.
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()

    # Load YAML into a Python dict and validate required experiment metadata.
    cfg = load_yaml(Path(args.config))

    # Required keys ensure naming and task selection are explicit.
    exp_name = get_required(cfg, "experiment.name")
    task = get_required(cfg, "experiment.task")

    # Initialize repo logger (writes to file + console) under the configured base_dir/run_group.
    logger, log_path = init_logger(
        script_name=cfg["experiment"]["name"],
        run_group=cfg.get("logging", {}).get("run_group", "training"),
        base_dir=Path(cfg.get("logging", {}).get("base_dir", "experiments/logs/representation")),
    )

    # Basic run metadata for traceability.
    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Task: {task}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Log path: {log_path}")

    # Seed all RNGs to make the train/val split and training deterministic where possible.
    seed = int(cfg.get("training", {}).get("seed", 13))
    seed_everything(seed)

    # Build the experiment output directory under PROJECT_ROOT / outputs.base_dir / experiment.name.
    exp_out_base = Path(cfg.get("outputs", {}).get("base_dir", "experiments/representation"))
    out_dir = PROJECT_ROOT / exp_out_base / str(exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config and minimal runtime metadata for reproducibility.
    save_json(out_dir / "config.json", cfg)
    save_json(out_dir / "run_meta.json", {"log_path": str(log_path), "device": DEVICE})

    # Create and prepare the data module (downloads/loads dataset, creates train/val split).
    dm = QEvasionDataModule(
        dataset_name=cfg.get("data", {}).get("dataset_name", "ailsntua/QEvasion"),
        validation_size=float(cfg.get("data", {}).get("validation_size", 0.2)),
        seed=seed,
        max_train_samples=cfg.get("data", {}).get("max_train_samples", None),
        max_val_samples=cfg.get("data", {}).get("max_val_samples", None),
        max_test_samples=cfg.get("data", {}).get("max_test_samples", None),
    ).prepare()

    # Retrieve prepared splits from the data module.
    train_split = dm.get_split("train")
    val_split = dm.get_split("validation")

    # Select the label field based on task; clarity is 3-way and evasion is multi-class.
    if task == "clarity":
        train_labels_raw = list(train_split.clarity_labels)
        val_labels_raw = list(val_split.clarity_labels)
    elif task == "evasion":
        train_labels_raw = list(train_split.evasion_labels)
        val_labels_raw = list(val_split.evasion_labels)
    else:
        raise ValueError("experiment.task must be 'clarity' or 'evasion'")

    # Build label mapping from train labels only to avoid leakage.
    label_to_id = build_label_mapping(train_labels_raw)
    id_to_label = {v: k for k, v in label_to_id.items()}

    # Fail fast if validation has labels unseen in training.
    unseen = sorted(set(val_labels_raw) - set(label_to_id.keys()))
    if unseen:
        raise ValueError(f"Validation contains unseen labels not present in train: {unseen}")

    # Convert raw labels to integer ids for training and validation.
    train_labels = [label_to_id[lbl] for lbl in train_labels_raw]
    val_labels = [label_to_id[lbl] for lbl in val_labels_raw]
    num_classes = len(label_to_id)

    # Save mapping so results can be decoded later.
    save_json(out_dir / "label_to_id.json", label_to_id)

    # Wrap the prepared splits with a simple Dataset that emits (text, label_id).
    train_ds = TextLabelDataset(train_split.texts, train_labels)
    val_ds = TextLabelDataset(val_split.texts, val_labels)

    # Create DataLoaders; texts are kept as strings, so num_workers is usually safe at 0.
    batch_size = int(cfg.get("training", {}).get("batch_size", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(cfg.get("data", {}).get("num_workers", 0)),
        pin_memory=(DEVICE == "cuda"),
    )

    # Build encoder/decoder from YAML blocks using repo factories.
    if "encoder" not in cfg or "decoder" not in cfg:
        raise KeyError("Config must include top-level `encoder` and `decoder` blocks")

    encoder = build_encoder(cfg["encoder"], device=DEVICE)
    decoder = build_decoder(cfg["decoder"], device=DEVICE)

    # Read LoRA enablement from YAML to decide whether freezing decoder should be overridden.
    decoder_lora_enabled = bool(cfg.get("decoder", {}).get("lora", {}).get("enabled", False))

    # Apply freeze flags while allowing LoRA adapters (if enabled) to remain trainable.
    if bool(cfg.get("encoder", {}).get("freeze", True)):
        maybe_freeze(encoder)

    if bool(cfg.get("decoder", {}).get("freeze", True)) and not decoder_lora_enabled:
        maybe_freeze(decoder)

    # Determine whether encoder/decoder will actually be trained based on requires_grad flags.
    train_encoder = has_trainable_params(encoder)
    train_decoder = has_trainable_params(decoder)

    # Log trainable parameter counts for quick sanity checking.
    logger.info(f"Trainable encoder params: {sum(p.numel() for p in encoder.parameters() if p.requires_grad)}")
    logger.info(f"Trainable decoder params: {sum(p.numel() for p in decoder.parameters() if p.requires_grad)}")
    logger.info(f"Decoder LoRA enabled (YAML): {decoder_lora_enabled}")

    # Infer encoder/decoder embedding dims by running a single probe example through both.
    probe_texts = [train_split.texts[0]]
    use_amp = bool(cfg.get("training", {}).get("use_amp", True))

    # Temporarily set eval for probing to avoid stochastic layers affecting shapes.
    enc_was_train = encoder.training
    dec_was_train = decoder.training
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            enc_probe = encoder(probe_texts)
            dec_probe = decoder(probe_texts)

    # Restore original training modes after probing.
    if enc_was_train:
        encoder.train()
    if dec_was_train:
        decoder.train()

    encoder_dim = infer_hidden_dim(enc_probe)
    decoder_dim = infer_hidden_dim(dec_probe)
    logger.info(f"Inferred encoder_dim={encoder_dim}, decoder_dim={decoder_dim}")

    # Load projection configs and inject inferred input_dim values if missing.
    enc_proj_cfg = cfg.get("projection", {}).get("encoder", None)
    dec_proj_cfg = cfg.get("projection", {}).get("decoder", None)
    if enc_proj_cfg is None or dec_proj_cfg is None:
        raise KeyError("Config must include projection.encoder and projection.decoder blocks")

    # Copy configs so we can safely modify params without mutating the original dict.
    enc_proj_cfg = dict(enc_proj_cfg)
    dec_proj_cfg = dict(dec_proj_cfg)

    enc_params = dict(enc_proj_cfg.get("params", {}))
    dec_params = dict(dec_proj_cfg.get("params", {}))
    enc_params.setdefault("input_dim", encoder_dim)
    dec_params.setdefault("input_dim", decoder_dim)
    enc_proj_cfg["params"] = enc_params
    dec_proj_cfg["params"] = dec_params

    # Build projection MLPs that map each branch into the shared fusion space.
    encoder_proj = build_projection(enc_proj_cfg).to(DEVICE)
    decoder_proj = build_projection(dec_proj_cfg).to(DEVICE)

    # Build fusion model (projection + fusion operator + classifier) from YAML.
    fusion_cfg = cfg.get("fusion", {})
    fusion_mode = str(fusion_cfg.get("mode", "concat"))

    classifier_cfg = cfg.get("classifier", None)

    fusion_model = RepresentationFusionModel(
        encoder_proj=encoder_proj,
        decoder_proj=decoder_proj,
        num_classes=num_classes,
        mode=fusion_mode,
        classifier_cfg=classifier_cfg,
    ).to(DEVICE)

    # Build loss function, optionally using class weights from the training label distribution.
    loss_cfg = cfg.get("loss", {})
    criterion = build_loss(
        labels=train_labels,
        num_classes=num_classes,
        device=DEVICE,
        class_weighted=bool(loss_cfg.get("class_weighted", False)),
        scheme=str(loss_cfg.get("weighting_scheme", "inverse_frequency")),
    )

    # Optimizer trains the fusion head plus any trainable encoder/decoder parameters (e.g., LoRA).
    trainable_params = collect_trainable_params(fusion_model, encoder, decoder)
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found. Check freeze/LoRA settings.")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg.get("training", {}).get("lr", 2e-4)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.01)),
    )

    # AMP gradient scaler is enabled only on CUDA.
    scaler = GradScaler(enabled=(use_amp and DEVICE == "cuda"))

    # Gradient clipping applies only to parameters that are expected to have gradients.
    clip_params = trainable_params

    # Training loop tracks best checkpoint using validation macro-F1.
    epochs = int(cfg.get("training", {}).get("epochs", 5))
    best_macro_f1 = -1.0
    best_ckpt_path = out_dir / "best.pt"

    logger.info("Starting training loop")
    logger.info(
        f"Epochs: {epochs} | Batch size: {batch_size} | AMP: {use_amp} | "
        f"Train encoder: {train_encoder} | Train decoder: {train_decoder}"
    )

    # Iterate epochs: train, validate, save epoch metrics, and checkpoint best model.
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch=epoch,
            cfg=cfg,
            logger=logger,
            train_encoder=train_encoder,
            train_decoder=train_decoder,
            clip_params=clip_params,
        )

        val_result = run_validation(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=val_loader,
            id_to_label=id_to_label,
            cfg=cfg,
        )

        # Extract macro-F1 and accuracy for compact logging.
        metrics = val_result.get("metrics", {})
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        acc = float(metrics.get("accuracy", 0.0))

        logger.info(
            f"[Epoch {epoch}] TrainLoss={train_loss:.4f} "
            f"ValAcc={acc:.4f} ValMacroF1={macro_f1:.4f}"
        )

        # Persist per-epoch validation results for later analysis/plots.
        save_json(out_dir / f"val_epoch_{epoch}.json", val_result)

        # Update best checkpoint when macro-F1 improves.
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            torch.save(
                {
                    "epoch": epoch,
                    "experiment": cfg.get("experiment", {}),
                    "fusion_model_state_dict": fusion_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "label_to_id": label_to_id,
                    "encoder_dim": encoder_dim,
                    "decoder_dim": decoder_dim,
                    "config": cfg,
                    "log_path": str(log_path),
                    "train_encoder": train_encoder,
                    "train_decoder": train_decoder,
                },
                best_ckpt_path,
            )

            logger.info(
                f"New best checkpoint saved | Epoch={epoch} "
                f"| MacroF1={best_macro_f1:.4f} | Path={best_ckpt_path}"
            )

    # Save a small final summary to make experiment tracking easier.
    summary = {
        "experiment_name": exp_name,
        "task": task,
        "device": DEVICE,
        "epochs": epochs,
        "best_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_ckpt_path),
        "log_path": str(log_path),
        "num_classes": num_classes,
        "label_to_id_path": str(out_dir / "label_to_id.json"),
        "train_encoder": train_encoder,
        "train_decoder": train_decoder,
    }
    save_json(out_dir / "summary.json", summary)

    # Final log lines point to the best score and saved artifacts.
    logger.info("Training completed")
    logger.info(f"Best Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Artifacts saved to: {out_dir}")
    logger.info(f"Log saved to: {log_path}")


# Standard Python entry point guard.
if __name__ == "__main__":
    main()
