from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from torch.amp import GradScaler, autocast

from src.representation.data.qev_datamodule import QEvasionDataModule
from src.representation.decoders.gpt2_decoder import GPT2Decoder
from src.representation.encoders.distilbert_encoder import DistilBERTEncoder
from src.representation.evaluation.evaluator import evaluate_predictions
from src.representation.models.fusion_model import RepresentationFusionModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# =========================================================
# CONFIG
# =========================================================

@dataclass
class TrainConfig:
    task: str = "clarity"  # "clarity" or "evasion"

    # Data
    dataset_name: str = "ailsntua/QEvasion"
    validation_size: float = 0.2
    seed: int = 13
    max_train_samples: int | None = None
    max_val_samples: int | None = None

    # Training
    epochs: int = 5
    batch_size: int = 4
    lr: float = 2e-4
    weight_decay: float = 0.01
    grad_accum_steps: int = 4
    max_grad_norm: float = 1.0

    # Mixed precision
    use_amp: bool = True

    # Model dims
    encoder_dim: int = 768
    decoder_dim: int = 768
    projection_dim: int = 256

    # Outputs / logging
    output_dir: str = "experiments/representation/baseline"
    use_wandb: bool = False
    wandb_project: str = "clarity-semeval-2026"
    wandb_run_name: str | None = None


CFG = TrainConfig()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# DATASET WRAPPER
class TextLabelDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts and labels length mismatch: {len(texts)} vs {len(labels)}")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.texts[idx], torch.tensor(self.labels[idx], dtype=torch.long)



# UTILITIES
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_labels(task: str, split: Any) -> List[str]:
    """
    Returns RAW string labels from the split.
    """
    if task == "clarity":
        return list(split.clarity_labels)
    if task == "evasion":
        return list(split.evasion_labels)
    raise ValueError(f"Unknown task: {task}. Expected 'clarity' or 'evasion'.")


def build_label_mapping(labels: List[str]) -> Dict[str, int]:
    """
    Deterministic mapping based on sorted unique labels.
    """
    uniq = sorted(set(labels))
    return {lbl: i for i, lbl in enumerate(uniq)}


def maybe_init_wandb(cfg: TrainConfig) -> None:
    if not cfg.use_wandb:
        return
    import wandb
    wandb.init(
        project=cfg.wandb_project,
        name=cfg.wandb_run_name,
        config=asdict(cfg),
    )


def maybe_wandb_log(cfg: TrainConfig, payload: Dict[str, float]) -> None:
    if not cfg.use_wandb:
        return
    import wandb
    wandb.log(payload)


def maybe_freeze(m: torch.nn.Module) -> None:
    """
    Prefer the module's own freeze() if it exists; otherwise do generic freezing.
    """
    if hasattr(m, "freeze") and callable(getattr(m, "freeze")):
        m.freeze()
        return
    m.eval()
    for p in m.parameters():
        p.requires_grad = False


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_json_safe(payload), f, indent=2)


def to_json_safe(obj):
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
    return obj


# TRAIN / EVAL
def train_one_epoch(
    *,
    encoder: DistilBERTEncoder,
    decoder: GPT2Decoder,
    fusion_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    epoch_idx: int,
    cfg: TrainConfig,
) -> float:
    fusion_model.train()
    total_loss = 0.0
    num_steps = 0

    optimizer.zero_grad(set_to_none=True)

    for step_idx, (texts, labels) in enumerate(loader, start=1):
        # labels are already ints (encoded), coming from TextLabelDataset
        if isinstance(labels, torch.Tensor):
            y = labels.to(device=DEVICE, dtype=torch.long)
        else:
            y = torch.tensor(labels, dtype=torch.long, device=DEVICE)

        # Frozen encoder/decoder forward (no grad)
        with torch.no_grad():
            with autocast(device_type="cuda", enabled=(cfg.use_amp and DEVICE == "cuda")):
                enc_vec = encoder(list(texts))  # expected (B, 768)
                dec_vec = decoder(list(texts))  # expected (B, 768)

        with autocast(device_type="cuda", enabled=(cfg.use_amp and DEVICE == "cuda")):
            logits = fusion_model(enc_vec, dec_vec)  # (B, C)
            loss = criterion(logits, y) / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if step_idx % cfg.grad_accum_steps == 0:
            if cfg.max_grad_norm and cfg.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * cfg.grad_accum_steps
        num_steps += 1

        if step_idx % 50 == 0:
            print(
                f"[Epoch {epoch_idx}] step={step_idx}/{len(loader)} "
                f"loss={total_loss / max(1, num_steps):.4f}"
            )

    return total_loss / max(1, num_steps)


@torch.no_grad()
def run_validation(
    *,
    encoder: DistilBERTEncoder,
    decoder: GPT2Decoder,
    fusion_model: nn.Module,
    loader: DataLoader,
    id_to_label: Dict[int, str],
    cfg: TrainConfig,
) -> Dict[str, Any]:
    fusion_model.eval()

    y_true_ids: List[int] = []
    y_pred_ids: List[int] = []

    for texts, labels in loader:
        y_true_ids.extend([int(x) for x in labels])

        with autocast(device_type="cuda", enabled=(cfg.use_amp and DEVICE == "cuda")):
            enc_vec = encoder(list(texts))
            dec_vec = decoder(list(texts))
            logits = fusion_model(enc_vec, dec_vec)

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        y_pred_ids.extend([int(p) for p in preds])

    # Evaluate in label space (strings) so evaluator can do per-class metrics nicely
    y_true = [id_to_label[i] for i in y_true_ids]
    y_pred = [id_to_label[i] for i in y_pred_ids]
    label_list = [id_to_label[i] for i in sorted(id_to_label.keys())]

    result = evaluate_predictions(y_true=y_true, y_pred=y_pred, label_list=label_list)
    # Include ids too (useful for debugging)
    result["y_true_ids_head"] = y_true_ids[:20]
    result["y_pred_ids_head"] = y_pred_ids[:20]
    return result


# MAIN
def main() -> None:
    seed_everything(CFG.seed)

    out_dir = PROJECT_ROOT / CFG.output_dir / CFG.task
    out_dir.mkdir(parents=True, exist_ok=True)

    maybe_init_wandb(CFG)
    save_json(out_dir / "config.json", asdict(CFG))

    # 1) Load data
    data_module = QEvasionDataModule(
        dataset_name=CFG.dataset_name,
        validation_size=CFG.validation_size,
        seed=CFG.seed,
        max_train_samples=CFG.max_train_samples,
        max_val_samples=CFG.max_val_samples,
    ).prepare()

    train_split = data_module.get_split("train")
    val_split = data_module.get_split("validation")

    train_labels_raw = select_labels(CFG.task, train_split)  # strings
    val_labels_raw = select_labels(CFG.task, val_split)      # strings

    # Build mapping from TRAIN only (standard practice)
    label_to_id = build_label_mapping(train_labels_raw)

    # Safety check: val shouldn't contain unseen labels
    unseen = sorted(set(val_labels_raw) - set(label_to_id.keys()))
    if unseen:
        raise ValueError(
            f"Validation contains unseen labels not in train: {unseen}. "
            "This indicates a split/label issue."
        )

    id_to_label = {v: k for k, v in label_to_id.items()}

    train_labels = [label_to_id[lbl] for lbl in train_labels_raw]
    val_labels = [label_to_id[lbl] for lbl in val_labels_raw]

    num_classes = len(label_to_id)

    # Persist mapping (critical for reproducibility)
    save_json(out_dir / "label_to_id.json", label_to_id)

    train_ds = TextLabelDataset(train_split.texts, train_labels)
    val_ds = TextLabelDataset(val_split.texts, val_labels)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    # 2) Models
    encoder = DistilBERTEncoder().to(DEVICE)
    decoder = GPT2Decoder().to(DEVICE)

    maybe_freeze(encoder)
    maybe_freeze(decoder)

    fusion_model = RepresentationFusionModel(
        encoder_dim=CFG.encoder_dim,
        decoder_dim=CFG.decoder_dim,
        projection_dim=CFG.projection_dim,
        num_classes=num_classes,
    ).to(DEVICE)

    # 3) Optimizer / loss
    optimizer = torch.optim.AdamW(
        fusion_model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    scaler = GradScaler(enabled=(CFG.use_amp and DEVICE == "cuda"))

    # 4) Train
    best_macro_f1 = -1.0
    best_path = out_dir / "best.pt"

    for epoch in range(1, CFG.epochs + 1):
        train_loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            epoch_idx=epoch,
            cfg=CFG,
        )

        val_result = run_validation(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=val_loader,
            id_to_label=id_to_label,
            cfg=CFG,
        )

        metrics = val_result.get("metrics", {})
        macro_f1 = float(metrics.get("macro_f1", 0.0))
        acc = float(metrics.get("accuracy", 0.0))

        print(
            f"[Epoch {epoch}] "
            f"TrainLoss={train_loss:.4f} "
            f"ValAcc={acc:.4f} "
            f"ValMacroF1={macro_f1:.4f}"
        )

        # Save per-epoch outputs
        save_json(out_dir / f"val_epoch_{epoch}.json", val_result)

        # W&B
        maybe_wandb_log(
            CFG,
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_accuracy": float(acc),
                "val_macro_f1": float(macro_f1),
            },
        )

        # Best checkpoint
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(
                {
                    "epoch": epoch,
                    "task": CFG.task,
                    "model_state_dict": fusion_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_macro_f1": best_macro_f1,
                    "config": asdict(CFG),
                    "label_to_id": label_to_id,
                },
                best_path,
            )
            print(f"  -> Saved new best checkpoint to {best_path} (macro_f1={best_macro_f1:.4f})")

    # Final summary
    summary = {
        "task": CFG.task,
        "best_macro_f1": best_macro_f1,
        "device": DEVICE,
        "config": asdict(CFG),
        "num_classes": num_classes,
    }
    save_json(out_dir / "summary.json", summary)
    print(f"Done. Summary saved to {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
