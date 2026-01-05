# src/representation/training/final_train_test.py
# ------------------------------------------------------------
# FINAL RUN (NO YAML):
# - Train on FULL QEvasion train split (3448)
# - Evaluate on OFFICIAL QEvasion test split (308)
#   * Clarity: single ground-truth clarity_label
#   * Evasion: 3 annotators (annotator1/2/3) + any-match accuracy
# - Saves everything under: PROJECT_ROOT/experiments/representation/FINAL_RUN/<timestamp>/
# ------------------------------------------------------------

# This script is the single entry-point for a reproducible "train-on-train, eval-on-official-test" run.


from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.representation.encoders.encoder_factory import build_encoder
from src.representation.projections.projection_factory import build_projection
from src.representation.models.fusion_model import RepresentationFusionModel
from src.representation.evaluation.evaluator import evaluate_predictions
from src.representation.evaluation.loss_factory import build_loss


# CONFIG
@dataclass
class FinalRunConfig:
    dataset_name: str = "ailsntua/QEvasion"

    # text composition
    question_col: str = "question"
    answer_col: str = "interview_answer"

    # labels
    clarity_col: str = "clarity_label"
    evasion_train_col: str = "evasion_label"
    evasion_annotator_cols: Tuple[str, str, str] = ("annotator1", "annotator2", "annotator3")

    # encoder
    encoder_type: str = "roberta"
    encoder_model: str = "roberta-base"
    encoder_pool: str = "masked_mean"
    encoder_freeze: bool = True

    # decoder (Qwen2-1.5B + LoRA)
    decoder_model: str = "Qwen/Qwen2-1.5B"
    decoder_pool: str = "last_non_pad"
    decoder_freeze: bool = True
    decoder_max_length: int = 128

    # LoRA
    lora_enabled: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # projection (both branches)
    proj_type: str = "mlp"
    proj_hidden_dim: int = 256
    proj_output_dim: int = 256
    proj_n_layers: int = 3
    proj_dropout: float = 0.1
    proj_activation: str = "relu"

    # fusion
    fusion_mode: str = "concat"

    # classifier head
    clf_type: str = "mlp"
    clf_hidden_dims: Tuple[int, ...] = (256, 256, 256)
    clf_activation: str = "relu"
    clf_dropout: float = 0.1

    # training
    seed: int = 13
    epochs: int = 5
    batch_size: int = 2
    grad_accum_steps: int = 2
    lr: float = 2e-4
    weight_decay: float = 1e-2
    max_grad_norm: float = 1.0
    use_amp: bool = True

    # loss
    class_weighted: bool = True
    weighting_scheme: str = "inverse_frequency"


CFG = FinalRunConfig()


# GLOBALS / PATHS / LOGGING
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# We default to CUDA if available, otherwise run on CPU.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# AMP device type must be one of PyTorch's supported strings.
AMP_DEVICE_TYPE = "cuda" if DEVICE == "cuda" else "cpu"

FINAL_RUN_ROOT = PROJECT_ROOT / "experiments" / "representation" / "FINAL_RUN"


def seed_everything(seed: int) -> None:
    """Seed Python/NumPy/PyTorch RNGs for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_safe(obj: Any) -> Any:
    """Convert common tensor/NumPy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]
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
    """Write a pretty-printed JSON file and create parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def setup_logger(out_dir: Path) -> logging.Logger:
    """Log to both console and run_dir/run.log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("FINAL_RUN")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# DATA
def compose_text(question: Optional[str], answer: str) -> str:
    """Standardize input formatting so encoder/decoder see identical text."""
    q = (question or "").strip()
    a = (answer or "").strip()
    if q:
        return f"Question: {q}\nAnswer: {a}"
    return a


class TextLabelDataset(Dataset):
    """Minimal Dataset that returns (text, label_id) pairs."""
    def __init__(self, texts: List[str], labels: List[int]) -> None:
        if len(texts) != len(labels):
            raise ValueError(f"texts/labels mismatch: {len(texts)} vs {len(labels)}")
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[str, torch.Tensor]:
        # Labels are stored as torch.long for CrossEntropyLoss.
        return self.texts[idx], torch.tensor(int(self.labels[idx]), dtype=torch.long)


def build_label_mapping_from_train(labels: Sequence[str]) -> Dict[str, int]:
    """Create a stable label->id mapping from train labels only."""
    uniq = sorted(set([str(x) for x in labels if x is not None and str(x).strip() != ""]))
    return {lbl: i for i, lbl in enumerate(uniq)}



# DECODER (self-contained Qwen + LoRA; avoids repo fragility)
class QwenLoRADecoder(nn.Module):
    """
    Produces a pooled sentence embedding from Qwen2 CausalLM hidden states,
    with optional LoRA adaptation.
    """
    def __init__(
        self,
        model_name: str,
        pooling: str,
        max_length: int,
        device: str,
        lora_enabled: bool,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        lora_target_modules: Sequence[str],
        freeze_base: bool = True,
    ):
        super().__init__()
        self.device = device
        self.pooling = pooling
        self.max_length = int(max_length)

        # Tokenizer config must define a pad token for batched padding.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Pick a dtype that is memory-friendly on GPU and safe on CPU.
        if self.device == "cuda":
            # bf16 is not universally supported; prefer fp16 unless bf16 is supported.
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            dtype = torch.float32

        # We load a causal LM but use it as a feature extractor via hidden states.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(self.device)

        # Disable KV cache to reduce memory during training.
        if hasattr(self.model, "config"):
            self.model.config.use_cache = False
            self.model.config.return_dict = True

        # Freeze base weights so only LoRA adapters and fusion head can train.
        if freeze_base:
            for p in self.model.parameters():
                p.requires_grad = False

        # LoRA inserts small trainable matrices into attention projections.
        if lora_enabled:
            from peft import LoraConfig, get_peft_model

            peft_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_target_modules),
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_cfg).to(self.device)
            self.model.train()

        # Hidden size defines the embedding dimension from the decoder branch.
        self.output_dim = int(self.model.config.hidden_size)

    def pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Pool token states into a single vector per example."""
        if self.pooling == "masked_mean":
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1)
            return summed / denom

        if self.pooling == "last_non_pad":
            lengths = attention_mask.sum(dim=1) - 1
            batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
            return hidden_states[batch_idx, lengths]

        if self.pooling == "max":
            mask = attention_mask.unsqueeze(-1)
            masked = hidden_states.masked_fill(mask == 0, -1e9)
            return masked.max(dim=1).values

        raise ValueError(f"Unknown pooling method: {self.pooling}")

    def forward(self, texts: List[str]) -> torch.Tensor:
        """Tokenize texts, run the LM, and pool last-layer hidden states."""
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        # We explicitly request hidden states so we can pool representations.
        outputs = self.model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hs = getattr(outputs, "hidden_states", None)
        if hs is None:
            raise RuntimeError("Decoder forward did not return hidden_states.")
        last = hs[-1]  # (B, T, D)
        return self.pool(last, batch["attention_mask"])


# TRAINING / EVAL LOOPS
def infer_hidden_dim(x: torch.Tensor) -> int:
    """Sanity-check embedding shape and return its last dimension."""
    if x.ndim != 2:
        raise ValueError(f"Expected (B,H) embedding tensor, got {tuple(x.shape)}")
    return int(x.shape[-1])


def collect_trainable_params(*modules: nn.Module) -> List[nn.Parameter]:
    """Collect parameters that require gradients across multiple modules."""
    params: List[nn.Parameter] = []
    for m in modules:
        params.extend([p for p in m.parameters() if p.requires_grad])
    return params


def train_one_epoch(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    fusion_model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: GradScaler,
    use_amp: bool,
    grad_accum_steps: int,
    max_grad_norm: float,
    logger: logging.Logger,
    epoch: int,
) -> float:
    """Run one epoch of training with gradient accumulation and optional AMP."""
    fusion_model.train()
    decoder.train()
    encoder.eval()  # always frozen

    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    num_batches = 0

    for step, (texts, labels) in enumerate(loader, start=1):
        y = labels.to(device=DEVICE, dtype=torch.long)
        texts_list = list(texts)

        # Encoder stays frozen, so we compute it under no_grad to save memory.
        with torch.no_grad():
            with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
                enc_vec = encoder(texts_list)

        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            dec_vec = decoder(texts_list)
            logits = fusion_model(enc_vec, dec_vec)
            # Loss is scaled by accumulation steps so the effective gradient matches a larger batch.
            loss = criterion(logits, y) / max(1, grad_accum_steps)

        scaler.scale(loss).backward()

        if step % grad_accum_steps == 0:
            if max_grad_norm and max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += float(loss.item()) * max(1, grad_accum_steps)
        num_batches += 1

    # Step once more if the final micro-batch did not align with grad_accum_steps.
    if (len(loader) % grad_accum_steps) != 0:
        if max_grad_norm and max_grad_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(fusion_model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    avg = total_loss / max(1, num_batches)
    logger.info(f"[Epoch {epoch}] TrainLoss={avg:.4f}")
    return avg


@torch.no_grad()
def predict_labels(
    *,
    encoder: nn.Module,
    decoder: nn.Module,
    fusion_model: nn.Module,
    texts: List[str],
    use_amp: bool,
    batch_size: int,
) -> List[int]:
    """Run batched inference and return predicted class ids."""
    encoder.eval()
    decoder.eval()
    fusion_model.eval()

    preds: List[int] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(use_amp and DEVICE == "cuda")):
            enc_vec = encoder(chunk)
            dec_vec = decoder(chunk)
            logits = fusion_model(enc_vec, dec_vec)
        p = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        preds.extend([int(x) for x in p])
    return preds


def evaluate_single_gold(
    *,
    y_true_labels: List[str],
    y_pred_ids: List[int],
    id_to_label: Dict[int, str],
) -> Dict[str, Any]:
    """Evaluate predictions when each example has one gold label (clarity)."""
    label_list = [id_to_label[i] for i in sorted(id_to_label.keys())]
    y_pred_labels = [id_to_label[i] for i in y_pred_ids]
    return evaluate_predictions(
        y_true=y_true_labels,
        y_pred=y_pred_labels,
        label_list=label_list,
    )


def evaluate_evasion_multi_annotator(
    *,
    y_pred_ids: List[int],
    id_to_label: Dict[int, str],
    annotators: Dict[str, List[Optional[str]]],  # name -> list of labels
) -> Dict[str, Any]:
    """
    Returns:
      - per_annotator: evaluator.py style dict for each annotator
      - any_match_accuracy
      - coverage info (missing labels)
    """
    # Predictions are mapped back to label strings once for reuse across evaluators.
    y_pred = [id_to_label[i] for i in y_pred_ids]
    label_list = [id_to_label[i] for i in sorted(id_to_label.keys())]

    per_ann: Dict[str, Any] = {}
    any_match_hits: List[int] = []
    any_match_total = 0

    # annotate-wise evaluation
    for ann_name, ann_labels in annotators.items():
        # We skip rows where the annotator label is missing to avoid counting invalid references.
        # filter rows with valid annotator label
        y_true_f: List[str] = []
        y_pred_f: List[str] = []
        valid = 0
        missing = 0

        for t, p in zip(ann_labels, y_pred):
            if t is None or str(t).strip() == "":
                missing += 1
                continue
            valid += 1
            y_true_f.append(str(t))
            y_pred_f.append(str(p))

        per_ann[ann_name] = {
            "valid_rows": valid,
            "missing_rows": missing,
            "results": evaluate_predictions(y_true=y_true_f, y_pred=y_pred_f, label_list=label_list),
        }

    # any-match accuracy (row-level; uses all 3 annotators if present)
    # This metric counts a prediction as correct if it matches any available annotator label.
    ann_cols = list(annotators.keys())
    ann_lists = [annotators[k] for k in ann_cols]

    for i in range(len(y_pred)):
        golds = []
        for ann_list in ann_lists:
            t = ann_list[i]
            if t is not None and str(t).strip() != "":
                golds.append(str(t))
        if len(golds) == 0:
            continue  # no reference labels present
        any_match_total += 1
        any_match_hits.append(1 if (y_pred[i] in golds) else 0)

    any_match_acc = float(np.mean(any_match_hits)) if any_match_total > 0 else None

    return {
        "per_annotator": per_ann,
        "any_match_accuracy": any_match_acc,
        "any_match_total_rows": any_match_total,
    }


# TASK RUNNER (train + test eval)
def run_task(
    *,
    task: str,  # "clarity" or "evasion"
    run_dir: Path,
    logger: logging.Logger,
    train_texts: List[str],
    train_labels_raw: List[str],
    test_texts: List[str],
    test_labels_raw: Optional[List[str]],  # clarity only
    test_annotators: Optional[Dict[str, List[Optional[str]]]],  # evasion only
) -> Dict[str, Any]:
    """Train a task-specific classifier (clarity or evasion) and evaluate on the official test split."""
    assert task in {"clarity", "evasion"}

    # Label mappings are derived from train only to prevent test leakage.
    label_to_id = build_label_mapping_from_train(train_labels_raw)
    id_to_label = {v: k for k, v in label_to_id.items()}
    num_classes = len(label_to_id)

    # We fail fast if the test set contains unseen labels.
    if task == "clarity":
        assert test_labels_raw is not None
        unseen = sorted(set([str(x) for x in test_labels_raw]) - set(label_to_id.keys()))
        if unseen:
            raise ValueError(f"[{task}] Test split contains unseen labels: {unseen}")
    else:
        assert test_annotators is not None
        all_test = []
        for v in test_annotators.values():
            all_test.extend([str(x) for x in v if x is not None and str(x).strip() != ""])
        unseen = sorted(set(all_test) - set(label_to_id.keys()))
        if unseen:
            raise ValueError(f"[{task}] Test split contains unseen annotator labels: {unseen}")

    # Convert string labels to integer ids once for training.
    train_labels = [label_to_id[str(x)] for x in train_labels_raw]

    # DataLoader returns raw text so the encoder/decoder can tokenize internally.
    train_ds = TextLabelDataset(train_texts, train_labels)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(CFG.batch_size),
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE == "cuda"),
    )

    # Build modules

    # Encoder branch is built via repo factory for consistency with earlier runs.
    encoder_cfg = {
        "type": CFG.encoder_type,
        "model_name": CFG.encoder_model,
        "pooling": CFG.encoder_pool,
        "freeze": CFG.encoder_freeze,
    }
    encoder = build_encoder(encoder_cfg, device=DEVICE)

    # Decoder branch is self-contained here to avoid dependency drift across repo modules.
    decoder = QwenLoRADecoder(
        model_name=CFG.decoder_model,
        pooling=CFG.decoder_pool,
        max_length=CFG.decoder_max_length,
        device=DEVICE,
        lora_enabled=CFG.lora_enabled,
        lora_r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        lora_target_modules=CFG.lora_target_modules,
        freeze_base=CFG.decoder_freeze,
    )

    # We probe one example to determine encoder/decoder embedding dimensions.
    probe = [train_texts[0]]
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        with autocast(device_type=AMP_DEVICE_TYPE, enabled=(CFG.use_amp and DEVICE == "cuda")):
            enc_probe = encoder(probe)
            dec_probe = decoder(probe)
    enc_dim = infer_hidden_dim(enc_probe)
    dec_dim = infer_hidden_dim(dec_probe)
    logger.info(f"[{task}] Inferred encoder_dim={enc_dim}, decoder_dim={dec_dim}")

    # Projections map both branches into the same latent dimension before fusion.
    enc_proj_cfg = {
        "type": CFG.proj_type,
        "params": {
            "input_dim": enc_dim,
            "hidden_dim": CFG.proj_hidden_dim,
            "output_dim": CFG.proj_output_dim,
            "n_layers": CFG.proj_n_layers,
            "dropout": CFG.proj_dropout,
            "activation": CFG.proj_activation,
        },
    }
    dec_proj_cfg = {
        "type": CFG.proj_type,
        "params": {
            "input_dim": dec_dim,
            "hidden_dim": CFG.proj_hidden_dim,
            "output_dim": CFG.proj_output_dim,
            "n_layers": CFG.proj_n_layers,
            "dropout": CFG.proj_dropout,
            "activation": CFG.proj_activation,
        },
    }
    encoder_proj = build_projection(enc_proj_cfg).to(DEVICE)
    decoder_proj = build_projection(dec_proj_cfg).to(DEVICE)

    # Classifier head consumes fused embeddings and outputs logits over classes.
    classifier_cfg = {
        "type": CFG.clf_type,
        "params": {
            "hidden_dims": list(CFG.clf_hidden_dims),
            "activation": CFG.clf_activation,
            "dropout": CFG.clf_dropout,
            # input_dim and num_classes are auto-filled by RepresentationFusionModel
        },
    }

    # Fusion model encapsulates projections, fusion operator, and the classifier head.
    fusion_model = RepresentationFusionModel(
        encoder_proj=encoder_proj,
        decoder_proj=decoder_proj,
        num_classes=num_classes,
        mode=CFG.fusion_mode,
        classifier_cfg=classifier_cfg,
    ).to(DEVICE)

    # Loss can be class-weighted to mitigate label imbalance
    criterion = build_loss(
        labels=train_labels,
        num_classes=num_classes,
        device=DEVICE,
        class_weighted=CFG.class_weighted,
        scheme=CFG.weighting_scheme,
    )

    # We train the fusion head and any LoRA adapter parameters in the decoder.
    trainable_params = collect_trainable_params(fusion_model, decoder)
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters found (fusion_model + decoder).")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(CFG.lr),
        weight_decay=float(CFG.weight_decay),
    )

    # AMP scaling reduces underflow risk when training in fp16/bf16 on GPU.
    scaler = GradScaler(enabled=(CFG.use_amp and DEVICE == "cuda"))


    # Train
    # We save epoch checkpoints so you can inspect or resume later.
    ckpt_dir = run_dir / task / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_losses: List[float] = []
    for epoch in range(1, int(CFG.epochs) + 1):
        loss = train_one_epoch(
            encoder=encoder,
            decoder=decoder,
            fusion_model=fusion_model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            use_amp=bool(CFG.use_amp),
            grad_accum_steps=int(CFG.grad_accum_steps),
            max_grad_norm=float(CFG.max_grad_norm),
            logger=logger,
            epoch=epoch,
        )
        train_losses.append(loss)

        # Checkpoints include fusion weights and LoRA adapter state for the decoder.
        torch.save(
            {
                "task": task,
                "epoch": epoch,
                "cfg": asdict(CFG),
                "label_to_id": label_to_id,
                "fusion_model_state": fusion_model.state_dict(),
                "decoder_state": decoder.state_dict(),  # includes peft adapter weights
                "optimizer_state": optimizer.state_dict(),
                "train_loss": loss,
                "encoder_dim": enc_dim,
                "decoder_dim": dec_dim,
            },
            ckpt_dir / f"epoch_{epoch}.pt",
        )


    # Test eval
    # We always evaluate on the official test split at the end of training.
    y_pred_ids = predict_labels(
        encoder=encoder,
        decoder=decoder,
        fusion_model=fusion_model,
        texts=test_texts,
        use_amp=bool(CFG.use_amp),
        batch_size=int(CFG.batch_size),
    )

    if task == "clarity":
        # Clarity uses a single gold label per instance.
        assert test_labels_raw is not None
        test_result = evaluate_single_gold(
            y_true_labels=[str(x) for x in test_labels_raw],
            y_pred_ids=y_pred_ids,
            id_to_label=id_to_label,
        )
    else:
        assert test_annotators is not None
        # Evasion uses three annotator labels; we compute per-annotator and any-match scores.
        test_result = evaluate_evasion_multi_annotator(
            y_pred_ids=y_pred_ids,
            id_to_label=id_to_label,
            annotators=test_annotators,
        )

    # Persist artifacts
    # Each task gets its own subfolder with mappings, losses, metrics, and predictions.
    task_dir = run_dir / task
    task_dir.mkdir(parents=True, exist_ok=True)

    save_json(task_dir / "label_to_id.json", label_to_id)
    save_json(task_dir / "train_losses.json", {"train_losses": train_losses})
    save_json(task_dir / "test_results.json", test_result)

    # JSONL predictions make it easy to debug errors and build submissions.
    preds_path = task_dir / "test_predictions.jsonl"
    with open(preds_path, "w", encoding="utf-8") as f:
        for i, (txt, pid) in enumerate(zip(test_texts, y_pred_ids)):
            row: Dict[str, Any] = {
                "index": i,
                "pred": id_to_label[pid],
            }
            if task == "clarity":
                row["gold"] = str(test_labels_raw[i])
            else:
                # store all 3 references (may be None)
                for ann_name, ann_list in test_annotators.items():
                    row[ann_name] = ann_list[i]
            # optional: include text (large); keep truncated
            row["text_head"] = txt[:250]
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")

    logger.info(f"[{task}] Saved test predictions -> {preds_path}")
    return {
        "task": task,
        "num_classes": num_classes,
        "encoder_dim": enc_dim,
        "decoder_dim": dec_dim,
        "train_losses": train_losses,
        "test_results": test_result,
        "predictions_path": str(preds_path),
        "label_to_id_path": str(task_dir / "label_to_id.json"),
        "checkpoints_dir": str(ckpt_dir),
    }


# ============================================================
#                            MAIN
# ============================================================

def main() -> None:
    # Each run is stored under a timestamp to keep outputs isolated.
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = FINAL_RUN_ROOT / ts
    logger = setup_logger(run_dir)

    # Global seeding helps keep results stable across reruns.
    seed_everything(int(CFG.seed))

    logger.info("============================================================")
    logger.info("FINAL RUN START")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Run dir: {run_dir}")
    logger.info("============================================================")

    # Save the exact config so the run can be reproduced later.
    save_json(run_dir / "final_run_config.json", asdict(CFG))

    # Load the official dataset splits from Hugging Face.
    t0 = time.perf_counter()
    ds = load_dataset(CFG.dataset_name)
    train_ds = ds["train"]
    test_ds = ds["test"]

    # Load the official dataset splits from Hugging Face.
    train_texts = [
        compose_text(train_ds[i].get(CFG.question_col), train_ds[i].get(CFG.answer_col))
        for i in range(len(train_ds))
    ]
    test_texts = [
        compose_text(test_ds[i].get(CFG.question_col), test_ds[i].get(CFG.answer_col))
        for i in range(len(test_ds))
    ]

    # Task 1 labels are directly available as a single gold label.
    clarity_train_labels = [str(x) for x in train_ds[CFG.clarity_col]]
    clarity_test_labels = [str(x) for x in test_ds[CFG.clarity_col]]

    # Task 2 uses a single evasion label for training.
    evasion_train_labels = [str(x) for x in train_ds[CFG.evasion_train_col]]

    # Task 2 test set provides three annotator labels per example.
    a1, a2, a3 = CFG.evasion_annotator_cols
    evasion_test_annotators = {
        a1: [test_ds[i].get(a1) for i in range(len(test_ds))],
        a2: [test_ds[i].get(a2) for i in range(len(test_ds))],
        a3: [test_ds[i].get(a3) for i in range(len(test_ds))],
    }

    logger.info(f"Dataset: {CFG.dataset_name}")
    logger.info(f"Train rows: {len(train_ds)} | Test rows: {len(test_ds)}")
    logger.info("Will run TWO separate trainings (clarity then evasion) using the same architecture.")

    # -------------------------
    # Run CLARITY
    # -------------------------
    logger.info("------------------------------------------------------------")
    logger.info("TASK 1: CLARITY (train on full train split, eval on official test)")
    logger.info("------------------------------------------------------------")
    clarity_summary = run_task(
        task="clarity",
        run_dir=run_dir,
        logger=logger,
        train_texts=train_texts,
        train_labels_raw=clarity_train_labels,
        test_texts=test_texts,
        test_labels_raw=clarity_test_labels,
        test_annotators=None,
    )

    # -------------------------
    # Run EVASION
    # -------------------------
    logger.info("------------------------------------------------------------")
    logger.info("TASK 2: EVASION (train on full train split, eval on official test w/ 3 annotators)")
    logger.info("------------------------------------------------------------")
    evasion_summary = run_task(
        task="evasion",
        run_dir=run_dir,
        logger=logger,
        train_texts=train_texts,
        train_labels_raw=evasion_train_labels,
        test_texts=test_texts,
        test_labels_raw=None,
        test_annotators=evasion_test_annotators,
    )

    duration_s = float(time.perf_counter() - t0)

    # FINAL_SUMMARY.json is a compact index for downstream reporting.
    final_summary = {
        "run_dir": str(run_dir),
        "device": DEVICE,
        "duration_seconds": duration_s,
        "config_path": str(run_dir / "final_run_config.json"),
        "clarity": clarity_summary,
        "evasion": evasion_summary,
        "notes": {
            "clarity_eval": "Single-ground-truth on test clarity_label",
            "evasion_eval": "Annotator-wise (annotator1/2/3) via evaluator.py + any-match accuracy",
            "model_saving": "Per-epoch .pt checkpoints include fusion head + LoRA adapter weights in decoder_state",
            "submission_hint": "SemEval typically expects test predictions files; see clarity/test_predictions.jsonl and evasion/test_predictions.jsonl",
        },
    }
    save_json(run_dir / "FINAL_SUMMARY.json", final_summary)

    logger.info("============================================================")
    logger.info("FINAL RUN COMPLETE")
    logger.info(f"Duration: {duration_s:.2f}s")
    logger.info(f"Summary -> {run_dir / 'FINAL_SUMMARY.json'}")
    logger.info("============================================================")


if __name__ == "__main__":
    main()
