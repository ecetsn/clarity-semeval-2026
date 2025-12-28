# src/representation/training/train_from_yaml.py
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

from src.representation.data.qev_datamodule import QEvasionDataModule
from src.representation.decoders.decoder_factory import build_decoder
from src.representation.encoders.encoder_factory import build_encoder
from src.representation.evaluation.evaluator import evaluate_predictions
from src.representation.evaluation.loss_factory import build_loss
from src.representation.logging.logger_setup import init_logger
from src.representation.models.fusion_model import RepresentationFusionModel
from src.representation.projections.projection_factory import build_projection


# ---------------------------------------------------------
# GLOBALS
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AMP_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------
class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts/labels length mismatch: {len(texts)} vs {len(labels)}")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        return self.texts[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)


# ---------------------------------------------------------
# UTILS
# ---------------------------------------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("YAML root must be a mapping/dict.")
    return cfg


def build_label_mapping(labels: List[str]) -> Dict[str, int]:
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}


def to_json_safe(obj: Any) -> Any:
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2, ensure_ascii=False)


def maybe_freeze(module: nn.Module) -> None:
    """
    Prefer module.freeze() if it exists; otherwise generic.
    """
    if hasattr(module, "freeze") and callable(getattr(module, "freeze")):
        module.freeze()
        return
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def infer_hidden_dim(x: torch.Tensor) -> int:
    """
    x: (B, H) expected
    """
    if x.ndim != 2:
        raise ValueError(f"Expected (B,H) embedding tensor, got shape {tuple(x.shape)}")
    return int(x.shape[-1])


def get_required(cfg: Dict[str, Any], key_path: str) -> Any:
    """
    Basic helper to avoid silent None’s for required YAML keys.
    key_path example: "experiment.name"
    """
    cur: Any = cfg
    for part in key_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(f"Missing required config key: {key_path}")
        cur = cur[part]
    return cur


# ---------------------------------------------------------
# TRAIN / EVAL
# ---------------------------------------------------------
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
) -> float:
    fusion_model.train()
    optimizer.zero_grad(set_to_none=True)

    use_amp = bool(cfg["training"].get("use_amp", True))
    grad_accum = int(cfg["training"].get("grad_accum_steps", 1))
    max_grad_norm = float(cfg["training"].get("max_grad_norm", 0.0))

    total_loss = 0.0
    num_batches = 0

    for step, (texts, labels) in enumerate(loader, start=1):
        # labels from DataLoader are already a stacked tensor
        y = labels.to(device=DEVICE, dtype=torch.long)

        # frozen encoder/decoder forward
        with torch.no_grad():
            with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
                enc_vec = encoder(list(texts))  # (B, Henc)
                dec_vec = decoder(list(texts))  # (B, Hdec)

        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            logits = fusion_model(enc_vec, dec_vec)  # (B, C)
            loss = criterion(logits, y) / max(1, grad_accum)

        scaler.scale(loss).backward()

        if step % grad_accum == 0:
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # Track true loss (undo /grad_accum)
        total_loss += float(loss.item()) * max(1, grad_accum)
        num_batches += 1

        if step % int(cfg["training"].get("log_every_steps", 50)) == 0:
            logger.info(
                f"[Epoch {epoch}] step={step}/{len(loader)} "
                f"train_loss={total_loss / max(1, num_batches):.4f}"
            )

    # Handle leftover grads if len(loader) not divisible by grad_accum
    if (len(loader) % grad_accum) != 0:
        max_grad_norm = float(cfg["training"].get("max_grad_norm", 0.0))
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_grad_norm)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

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
    fusion_model.eval()
    use_amp = bool(cfg["training"].get("use_amp", True))

    y_true_ids: List[int] = []
    y_pred_ids: List[int] = []

    for texts, labels in loader:
        y_true_ids.extend(labels.detach().cpu().tolist())

        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            enc_vec = encoder(list(texts))
            dec_vec = decoder(list(texts))
            logits = fusion_model(enc_vec, dec_vec)

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        y_pred_ids.extend([int(p) for p in preds])

    # stable label_list order: id 0..C-1
    label_list = [id_to_label[i] for i in sorted(id_to_label.keys())]
    y_true = [id_to_label[i] for i in y_true_ids]
    y_pred = [id_to_label[i] for i in y_pred_ids]

    result = evaluate_predictions(
        y_true=y_true,
        y_pred=y_pred,
        label_list=label_list,
    )
    result["y_true_ids_head"] = y_true_ids[:20]
    result["y_pred_ids_head"] = y_pred_ids[:20]
    return result


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_yaml(Path(args.config))

    # Required keys (fail fast)
    exp_name = get_required(cfg, "experiment.name")
    task = get_required(cfg, "experiment.task")

    # Paths
    # logs: PROJECT_BASE/experiments/logs/representation/<run_group>/<exp_name>/<timestamp>.log
    logs_base_dir = Path(cfg.get("logging", {}).get("base_dir", "experiments/logs/representation"))
    run_group = cfg.get("logging", {}).get("run_group", "training")

    logger, log_path = init_logger(
        script_name=cfg["experiment"]["name"],
        run_group=cfg["logging"]["run_group"],
        base_dir=Path(cfg["logging"]["base_dir"]),
    )

    logger.info(f"Loaded config: {args.config}")
    logger.info(f"Experiment name: {exp_name}")
    logger.info(f"Task: {task}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Log path: {log_path}")

    seed = int(cfg.get("training", {}).get("seed", 13))
    seed_everything(seed)

    # Experiment output directory
    exp_out_base = Path(cfg.get("outputs", {}).get("base_dir", "experiments/representation"))
    out_dir = PROJECT_ROOT / exp_out_base / str(exp_name)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist config + log path for traceability
    save_json(out_dir / "config.json", cfg)
    save_json(out_dir / "run_meta.json", {"log_path": str(log_path), "device": DEVICE})

    # --------------------
    # Data
    # --------------------
    dm = QEvasionDataModule(
        dataset_name=cfg.get("data", {}).get("dataset_name", "ailsntua/QEvasion"),
        validation_size=float(cfg.get("data", {}).get("validation_size", 0.2)),
        seed=seed,
        max_train_samples=cfg.get("data", {}).get("max_train_samples", None),
        max_val_samples=cfg.get("data", {}).get("max_val_samples", None),
        max_test_samples=cfg.get("data", {}).get("max_test_samples", None),
    ).prepare()

    train_split = dm.get_split("train")
    val_split = dm.get_split("validation")

    if task == "clarity":
        train_labels_raw = list(train_split.clarity_labels)
        val_labels_raw = list(val_split.clarity_labels)
    elif task == "evasion":
        train_labels_raw = list(train_split.evasion_labels)
        val_labels_raw = list(val_split.evasion_labels)
    else:
        raise ValueError("experiment.task must be 'clarity' or 'evasion'")

    label_to_id = build_label_mapping(train_labels_raw)
    id_to_label = {v: k for k, v in label_to_id.items()}

    unseen = sorted(set(val_labels_raw) - set(label_to_id.keys()))
    if unseen:
        raise ValueError(f"Validation contains unseen labels not present in train: {unseen}")

    train_labels = [label_to_id[lbl] for lbl in train_labels_raw]
    val_labels = [label_to_id[lbl] for lbl in val_labels_raw]
    num_classes = len(label_to_id)

    save_json(out_dir / "label_to_id.json", label_to_id)

    train_ds = TextLabelDataset(train_split.texts, train_labels)
    val_ds = TextLabelDataset(val_split.texts, val_labels)

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

    # --------------------
    # Models (YAML-driven, no hardcoding)
    # --------------------
    encoder = build_encoder(cfg["encoder"], device=DEVICE)
    decoder = build_decoder(cfg["decoder"], device=DEVICE)


    # Respect YAML freeze flags (factories usually handle it, but we enforce too)
    if bool(cfg.get("encoder", {}).get("freeze", True)):
        maybe_freeze(encoder)
    if bool(cfg.get("decoder", {}).get("freeze", True)):
        maybe_freeze(decoder)

    # Infer embedding dims via a single cheap forward on a tiny batch
    # (this eliminates fragile hardcoding like 768)
    probe_texts = [train_split.texts[0]]
    with torch.no_grad():
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(bool(cfg["training"].get("use_amp", True)) and DEVICE == "cuda")):
            enc_probe = encoder(probe_texts)
            dec_probe = decoder(probe_texts)
    encoder_dim = infer_hidden_dim(enc_probe)
    decoder_dim = infer_hidden_dim(dec_probe)

    logger.info(f"Inferred encoder_dim={encoder_dim}, decoder_dim={decoder_dim}")

    # Projections via factory (baseline or ablations)
    enc_proj_cfg = cfg.get("projection", {}).get("encoder", None)
    dec_proj_cfg = cfg.get("projection", {}).get("decoder", None)

    if enc_proj_cfg is None or dec_proj_cfg is None:
        raise KeyError("Config must include projection.encoder and projection.decoder blocks")

    # Ensure input_dim exists; if user omitted, fill from inferred dims.
    enc_proj_cfg = dict(enc_proj_cfg)
    dec_proj_cfg = dict(dec_proj_cfg)

    enc_params = dict(enc_proj_cfg.get("params", {}))
    dec_params = dict(dec_proj_cfg.get("params", {}))
    enc_params.setdefault("input_dim", encoder_dim)
    dec_params.setdefault("input_dim", decoder_dim)
    enc_proj_cfg["params"] = enc_params
    dec_proj_cfg["params"] = dec_params

    encoder_proj = build_projection(enc_proj_cfg).to(DEVICE)
    decoder_proj = build_projection(dec_proj_cfg).to(DEVICE)

    fusion_model = RepresentationFusionModel(
        encoder_proj=encoder_proj,
        decoder_proj=decoder_proj,
        num_classes=num_classes,
    ).to(DEVICE)

    # --------------------
    # Loss / Optimizer
    # --------------------
    loss_cfg = cfg.get("loss", {})
    criterion = build_loss(
        labels=train_labels,
        num_classes=num_classes,
        device=DEVICE,
        class_weighted=bool(loss_cfg.get("class_weighted", False)),
        scheme=str(loss_cfg.get("weighting_scheme", "inverse_frequency")),
    )

    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=float(cfg.get("training", {}).get("lr", 2e-4)),
        weight_decay=float(cfg.get("training", {}).get("weight_decay", 0.01)),
    )

    use_amp = bool(cfg.get("training", {}).get("use_amp", True))
    scaler = GradScaler(enabled=(use_amp and DEVICE == "cuda"))

    # --------------------
    # Train loop
    # --------------------
    epochs = int(cfg.get("training", {}).get("epochs", 5))
    best_macro_f1 = -1.0
    best_ckpt_path = out_dir / "best.pt"

    logger.info("Starting training loop")
    logger.info(f"Epochs: {epochs} | Batch size: {batch_size} | AMP: {use_amp}")

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
        )

        val_result = run_validation(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=val_loader,
            id_to_label=id_to_label,
            cfg=cfg,
        )

        metrics = val_result.get("metrics", {})
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        acc = float(metrics.get("accuracy", 0.0))

        logger.info(
            f"[Epoch {epoch}] TrainLoss={train_loss:.4f} "
            f"ValAcc={acc:.4f} ValMacroF1={macro_f1:.4f}"
        )

        save_json(out_dir / f"val_epoch_{epoch}.json", val_result)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            torch.save(
                {
                    "epoch": epoch,
                    "experiment": cfg.get("experiment", {}),
                    "model_state_dict": fusion_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "label_to_id": label_to_id,
                    "encoder_dim": encoder_dim,
                    "decoder_dim": decoder_dim,
                    "config": cfg,
                    "log_path": str(log_path),
                },
                best_ckpt_path,
            )

            logger.info(
                f"New best checkpoint saved | Epoch={epoch} "
                f"| MacroF1={best_macro_f1:.4f} | Path={best_ckpt_path}"
            )

    # --------------------
    # Final summary
    # --------------------
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
    }
    save_json(out_dir / "summary.json", summary)

    logger.info("Training completed")
    logger.info(f"Best Macro-F1: {best_macro_f1:.4f}")
    logger.info(f"Artifacts saved to: {out_dir}")
    logger.info(f"Log saved to: {log_path}")


if __name__ == "__main__":
    main()
