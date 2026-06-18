#!/usr/bin/env python3
"""
train_3b_full.py  —  Qwen2.5-3B Deep Fine-tune (long-run edition)

Mirrors train_3b_full.ipynb exactly, plus:
  • Resume from latest checkpoint   (--resume)
  • Time-budget graceful stop       (--max-hours 66)
  • Periodic checkpoint             (--ckpt-every 500)
  • SIGTERM / SIGINT handler        saves state, then exits cleanly
  • Dual logging: console + file    logs/train_3b_full_<RUN_ID>.log
  • Eval-only mode                  (--eval-only)
"""

import argparse
import csv
import json
import math
import os
import signal
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "Cleaned"
LOG_DIR   = BASE_DIR / "logs"
FIG_DIR   = BASE_DIR / "figures"
MODEL_DIR = BASE_DIR / "models" / "llm_3b_cls"
NB_LOG_DIR = BASE_DIR / "notebooks" / "logs_3b_full"
CKPT_DIR  = MODEL_DIR / "checkpoint_latest"

for d in [LOG_DIR, FIG_DIR, MODEL_DIR, NB_LOG_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Defaults (all overridable via CLI) ────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
NUM_LABELS  = 3
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

LORA_RANK    = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"]

MAX_LENGTH   = 256
BATCH_SIZE   = 2
GRAD_ACCUM   = 4
MAX_EPOCHS   = 20
EARLY_STOP   = 5
LR           = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOG_EVERY    = 20
CKPT_EVERY   = 500
USE_AUTOCAST = True

FOCAL_GAMMA  = 2.0
MARGIN       = 1.0
ALPHA_FOCAL  = 0.7
ALPHA_MARGIN = 0.3
LABEL_SMOOTH = 0.1

USE_RDROP    = True
RDROP_ALPHA  = 0.5
USE_LLRD     = True
LLRD_DECAY   = 0.95
USE_SUPCON   = True
SUPCON_TEMP  = 0.07
SUPCON_WEIGHT = 0.1

BIAS_SEARCH  = True
BIAS_GRID    = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

SEED = 42

# ── Graceful-stop flag (set by SIGTERM/SIGINT) ────────────────────────────────
_STOP_REQUESTED = False


def _signal_handler(signum, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print("\n[SIGNAL] Stop requested — will save checkpoint after current step.",
          flush=True)


signal.signal(signal.SIGTERM, _signal_handler)
signal.signal(signal.SIGINT,  _signal_handler)


# ── Logging ───────────────────────────────────────────────────────────────────
_log_fh = None


def setup_logger(run_id: str):
    global _log_fh
    log_path = LOG_DIR / f"train_3b_full_{run_id}.log"
    _log_fh  = open(log_path, "a", buffering=1)
    _log_fh.write(f"\n\n{'='*70}\n[START] {datetime.now().isoformat()}\n{'='*70}\n")
    print(f"[LOG] {log_path}")


def log(msg: str):
    ts  = datetime.now().strftime("%H:%M:%S")
    out = f"[{ts}] {msg}"
    print(out, flush=True)
    if _log_fh:
        _log_fh.write(out + "\n")


def fmt_eta(seconds: float) -> str:
    td = timedelta(seconds=int(seconds))
    h, rem = divmod(td.seconds, 3600)
    m = rem // 60
    if td.days:
        return f"{td.days}d {h:02d}h {m:02d}m"
    return f"{h:02d}h {m:02d}m"


# ── Checkpoint helpers ────────────────────────────────────────────────────────
CKPT_STATE_FILE = CKPT_DIR / "train_state.json"


def save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                    best_f1, no_improve, samples_seen, run_id, step_history,
                    epoch_history):
    log(f"Saving checkpoint (epoch={epoch}, step={global_step}, best_f1={best_f1:.4f}) ...")
    model.save_pretrained(str(CKPT_DIR))
    torch.save(optimizer.state_dict(), CKPT_DIR / "optimizer.pt")
    torch.save(scheduler.state_dict(), CKPT_DIR / "scheduler.pt")
    state = {
        "epoch":         epoch,
        "global_step":   global_step,
        "best_f1":       best_f1,
        "no_improve":    no_improve,
        "samples_seen":  samples_seen,
        "run_id":        run_id,
    }
    CKPT_STATE_FILE.write_text(json.dumps(state, indent=2))

    # Also persist histories so resume can continue appending
    (CKPT_DIR / "step_history.json").write_text(
        json.dumps(step_history, ensure_ascii=False))
    (CKPT_DIR / "epoch_history.json").write_text(
        json.dumps(epoch_history, ensure_ascii=False))
    log("Checkpoint saved.")


def load_checkpoint_state() -> dict | None:
    if CKPT_STATE_FILE.exists():
        return json.loads(CKPT_STATE_FILE.read_text())
    return None


# ── Dataset ───────────────────────────────────────────────────────────────────
class ViHSDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts   = df["text"].tolist()
        self.labels  = df["label"].tolist()
        self.tok     = tokenizer
        self.max_len = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(
            self.texts[idx], max_length=self.max_len,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ── Loss functions ────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            n = logits.size(-1)
            smooth = torch.full_like(logits, self.ls / n)
            smooth.scatter_(1, targets.unsqueeze(1), 1 - self.ls + self.ls / n)
            ce = -(smooth * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, targets, weight=self.weight,
                                 reduction="none")
        pt = torch.exp(
            -F.cross_entropy(logits, targets, weight=self.weight, reduction="none"))
        return (((1 - pt) ** self.gamma) * ce).mean()


def margin_loss(logits, labels, margin=1.0):
    B = logits.size(0)
    correct = logits[torch.arange(B), labels]
    mask    = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B), labels] = False
    wrong   = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return F.relu(margin - correct + wrong).mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight, gamma, margin, a_f, a_m, ls):
        super().__init__()
        self.focal = FocalLoss(gamma, weight, ls)
        self.m = margin
        self.a_f = a_f
        self.a_m = a_m

    def forward(self, logits, labels):
        fl = self.focal(logits, labels)
        ml = margin_loss(logits, labels, self.m)
        return self.a_f * fl + self.a_m * ml, fl.item(), ml.item()


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.T = temperature

    def forward(self, features, labels):
        B, device = features.size(0), features.device
        if B < 2:
            return torch.tensor(0.0, device=device)
        sim     = torch.matmul(features, features.T) / self.T
        sim     = sim - sim.max(dim=1, keepdim=True).values.detach()
        eye     = torch.eye(B, device=device)
        mask_pos = (
            (labels.unsqueeze(1) == labels.unsqueeze(0)).float() * (1 - eye)
        )
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        exp_sim  = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        return -(log_prob * mask_pos).sum() / mask_pos.sum()


# ── LLRD optimizer ────────────────────────────────────────────────────────────
def build_llrd_groups(model, base_lr, decay=0.95, wd=0.01):
    try:
        n = model.base_model.model.config.num_hidden_layers
    except AttributeError:
        n = 36

    param_lr = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in ("score", "classifier")):
            depth = 0
        else:
            depth = n + 1
            for i in range(n):
                if f".layers.{i}." in name:
                    depth = n - i
                    break
        param_lr[id(p)] = (p, base_lr * (decay ** depth))

    lr_to_params: dict = {}
    for _, (p, lr) in param_lr.items():
        lr_to_params.setdefault(lr, []).append(p)

    groups = sorted(
        [{"params": ps, "lr": lr, "weight_decay": wd}
         for lr, ps in lr_to_params.items()],
        key=lambda g: -g["lr"],
    )
    if not groups:
        raise RuntimeError(
            "LLRD: no trainable parameters found. "
            "After PeftModel.from_pretrained, LoRA params must be unfrozen "
            "before building the optimizer (requires_grad=True).")
    log(f"LLRD: {len(groups)} groups  LR [{groups[-1]['lr']:.2e} .. {groups[0]['lr']:.2e}]")
    return groups


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion, logit_bias=None,
             split_name="Val", return_logits=False):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0

    for batch in loader:
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=USE_AUTOCAST and device.type == "cuda"):
            out    = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
        loss, _, _ = criterion(logits, lbls)
        total_loss += loss.item()
        if logit_bias is not None:
            logits = logits + logit_bias.to(logits.device)
        all_logits.append(logits.float().cpu())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro",  zero_division=0)
    f1c = f1_score(all_labels, all_preds, average=None,     zero_division=0).tolist()
    log(f"  [{split_name}] loss={total_loss/len(loader):.4f}  acc={acc:.4f}  "
        f"F1={f1m:.4f}  C={f1c[0]:.4f}  O={f1c[1]:.4f}  H={f1c[2]:.4f}")

    if return_logits:
        return acc, f1m, f1c, all_preds, all_labels, torch.cat(all_logits)
    return acc, f1m, f1c, all_preds, all_labels


# ── Per-step loss with R-Drop + SupCon ───────────────────────────────────────
def compute_loss_step(model, ids, mask, lbls, criterion, supcon_crit, device):
    B = ids.size(0)

    if USE_SUPCON:
        out1    = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logits1 = out1.logits
        try:
            last_h  = out1.hidden_states[-1]
            seq_end = mask.sum(dim=1) - 1
            cls_emb = last_h[torch.arange(B, device=device), seq_end, :]
            cls_emb = F.normalize(cls_emb.float(), dim=-1)
            sc_loss = supcon_crit(cls_emb, lbls)
            sc_val  = sc_loss.item()
        except Exception:
            sc_loss = torch.tensor(0.0, device=device)
            sc_val  = 0.0

        if USE_RDROP:
            out2    = model(input_ids=ids, attention_mask=mask)
            logits2 = out2.logits
            p1 = F.softmax(logits1.float(), dim=-1).clamp(1e-7, 1)
            p2 = F.softmax(logits2.float(), dim=-1).clamp(1e-7, 1)
            kl = (F.kl_div(p1.log(), p2, reduction="batchmean") +
                  F.kl_div(p2.log(), p1, reduction="batchmean")) / 2
            t1, fl, ml = criterion(logits1, lbls)
            t2, _,  _  = criterion(logits2, lbls)
            loss = (t1 + t2) / 2 + RDROP_ALPHA * kl + SUPCON_WEIGHT * sc_loss
            return loss, fl, ml, kl.item(), sc_val, logits1
        else:
            t1, fl, ml = criterion(logits1, lbls)
            loss = t1 + SUPCON_WEIGHT * sc_loss
            return loss, fl, ml, 0.0, sc_val, logits1

    elif USE_RDROP:
        out1 = model(input_ids=ids, attention_mask=mask)
        out2 = model(input_ids=ids, attention_mask=mask)
        p1   = F.softmax(out1.logits.float(), dim=-1).clamp(1e-7, 1)
        p2   = F.softmax(out2.logits.float(), dim=-1).clamp(1e-7, 1)
        kl   = (F.kl_div(p1.log(), p2, reduction="batchmean") +
                F.kl_div(p2.log(), p1, reduction="batchmean")) / 2
        t1, fl, ml = criterion(out1.logits, lbls)
        t2, _,  _  = criterion(out2.logits, lbls)
        return (t1 + t2) / 2 + RDROP_ALPHA * kl, fl, ml, kl.item(), 0.0, out1.logits

    else:
        out = model(input_ids=ids, attention_mask=mask)
        loss, fl, ml = criterion(out.logits, lbls)
        return loss, fl, ml, 0.0, 0.0, out.logits


# ── Main train loop ───────────────────────────────────────────────────────────
def train(model, optimizer, scheduler, train_loader, val_loader,
          device, criterion, supcon_crit, tokenizer,
          run_id, log_csv, max_hours, max_epochs, early_stop, ckpt_every,
          resume_state=None):
    """
    Returns: (best_f1, total_steps, samples_seen, step_history, epoch_history)

    Stopping conditions (whichever triggers first):
      1. Early stopping: val F1 does not improve for `early_stop` epochs
      2. Time budget:    elapsed >= max_hours * 3600 - 1800 (30-min safety buffer)
      3. Max epochs:     epoch counter reaches max_epochs
      4. Signal:         SIGTERM or SIGINT received (_STOP_REQUESTED flag)
    """
    best_f1      = resume_state["best_f1"]      if resume_state else 0.0
    no_improve   = resume_state["no_improve"]   if resume_state else 0
    global_step  = resume_state["global_step"]  if resume_state else 0
    samples_seen = resume_state["samples_seen"] if resume_state else 0
    start_epoch  = resume_state["epoch"] + 1    if resume_state else 1

    step_history  = (json.loads((CKPT_DIR / "step_history.json").read_text())
                     if resume_state and (CKPT_DIR / "step_history.json").exists()
                     else [])
    epoch_history = (json.loads((CKPT_DIR / "epoch_history.json").read_text())
                     if resume_state and (CKPT_DIR / "epoch_history.json").exists()
                     else [])

    t0           = time.time()
    max_seconds  = max_hours * 3600
    stop_at      = max_seconds - 1800  # save 30 min buffer before hard limit

    log(f"Training start — epochs {start_epoch}..{max_epochs}  "
        f"early_stop={early_stop}  max_hours={max_hours:.1f}h  "
        f"ckpt_every={ckpt_every}")
    if resume_state:
        log(f"Resuming: epoch={start_epoch-1}  step={global_step}  "
            f"best_f1={best_f1:.4f}  no_improve={no_improve}")

    for epoch in range(start_epoch, max_epochs + 1):
        # ── time-budget check before epoch ────────────────────────────────────
        elapsed = time.time() - t0
        if elapsed >= stop_at:
            log(f"Time budget reached ({elapsed/3600:.2f}h >= {max_hours-0.5:.1f}h). "
                "Stopping before epoch starts.")
            save_checkpoint(model, optimizer, scheduler, epoch - 1, global_step,
                            best_f1, no_improve, samples_seen, run_id,
                            step_history, epoch_history)
            break

        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{max_epochs}", leave=False)

        for step, batch in enumerate(pbar):
            if _STOP_REQUESTED:
                log("Stop signal — saving checkpoint and exiting.")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                best_f1, no_improve, samples_seen, run_id,
                                step_history, epoch_history)
                return best_f1, global_step, samples_seen, step_history, epoch_history

            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)
            samples_seen += ids.size(0)

            try:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                    enabled=USE_AUTOCAST and device.type == "cuda"):
                    loss, fl, ml, kl_v, sc_v, _ = compute_loss_step(
                        model, ids, mask, lbls, criterion, supcon_crit, device)
                (loss / GRAD_ACCUM).backward()
                epoch_loss += loss.item()
            except Exception as _e:
                _emsg = str(_e)
                if "CUDA" in _emsg or "cuda" in _emsg or "AcceleratorError" in type(_e).__name__:
                    log(f"[CUDA ERROR] epoch={epoch} batch={step} opt_step={global_step}: {_e}")
                    log("Saving emergency checkpoint before exit...")
                    try:
                        optimizer.zero_grad()
                        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                        best_f1, no_improve, samples_seen, run_id,
                                        step_history, epoch_history)
                        log(f"Emergency checkpoint saved. Re-run with --resume to continue "
                            f"from epoch {epoch + 1} (global_step={global_step}).")
                    except Exception as _se:
                        log(f"[WARN] Emergency checkpoint failed: {_se}")
                    raise
                raise

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                cur_lr  = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0

                # ── periodic checkpoint ────────────────────────────────────────
                if global_step % ckpt_every == 0:
                    save_checkpoint(model, optimizer, scheduler, epoch,
                                    global_step, best_f1, no_improve,
                                    samples_seen, run_id,
                                    step_history, epoch_history)

                if global_step % LOG_EVERY == 0:
                    steps_per_sec = global_step / max(elapsed, 1)
                    total_steps   = (len(train_loader) // GRAD_ACCUM) * max_epochs
                    remaining     = max(total_steps - global_step, 0)
                    eta_sec       = remaining / max(steps_per_sec, 1e-6)
                    log(f"  s={global_step:5d}  loss={loss.item():.4f}  "
                        f"fl={fl:.3f} mg={ml:.3f} kl={kl_v:.3f} sc={sc_v:.3f}  "
                        f"lr={cur_lr:.1e}  n={samples_seen:,}  "
                        f"elapsed={elapsed/3600:.2f}h  ETA={fmt_eta(eta_sec)}")

                step_history.append({
                    "step": global_step, "loss": round(loss.item(), 5),
                    "focal": round(fl, 5), "margin": round(ml, 5),
                    "rdrop_kl": round(kl_v, 5), "supcon": round(sc_v, 5),
                    "lr": cur_lr, "samples_seen": samples_seen,
                })
                with open(log_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, global_step, round(loss.item(), 5),
                        round(fl, 5), round(ml, 5), round(kl_v, 5), round(sc_v, 5),
                        round(cur_lr, 8), samples_seen, round(elapsed, 1),
                    ])
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}", kl=f"{kl_v:.3f}", sc=f"{sc_v:.3f}")

            # ── time-budget check mid-epoch ────────────────────────────────────
            if (time.time() - t0) >= stop_at:
                log(f"Time budget reached mid-epoch {epoch}. Saving and stopping.")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                best_f1, no_improve, samples_seen, run_id,
                                step_history, epoch_history)
                return best_f1, global_step, samples_seen, step_history, epoch_history

        # ── end-of-epoch validation ────────────────────────────────────────────
        avg_loss = epoch_loss / len(train_loader)
        elapsed  = time.time() - t0
        log(f"\n[Epoch {epoch}] avg_loss={avg_loss:.4f}  "
            f"samples={samples_seen:,}  elapsed={elapsed/3600:.2f}h")
        val_acc, val_f1, val_f1c, _, _ = evaluate(
            model, val_loader, device, criterion, split_name="Val")

        epoch_history.append({
            "epoch": epoch, "train_loss": round(avg_loss, 5),
            "val_acc": round(val_acc, 5), "val_f1": round(val_f1, 5),
            "val_f1_cls": [round(x, 5) for x in val_f1c],
            "global_step": global_step,
            "elapsed_h": round(elapsed / 3600, 3),
        })

        if val_f1 > best_f1:
            best_f1   = val_f1
            no_improve = 0
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))
            log(f"  => New best  F1={best_f1:.4f}  step={global_step}")
        else:
            no_improve += 1
            log(f"  => No improve ({no_improve}/{early_stop})")
            if no_improve >= early_stop:
                log(f"Early stopping at epoch {epoch}")
                save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                                best_f1, no_improve, samples_seen, run_id,
                                step_history, epoch_history)
                break

        # save checkpoint after each epoch regardless
        save_checkpoint(model, optimizer, scheduler, epoch, global_step,
                        best_f1, no_improve, samples_seen, run_id,
                        step_history, epoch_history)

    total_min = (time.time() - t0) / 60
    log(f"Done: {total_min:.1f}min | Val F1={best_f1:.4f} | "
        f"steps={global_step:,} | samples={samples_seen:,}")
    return best_f1, global_step, samples_seen, step_history, epoch_history


# ── Post-training: plots, bias search, error analysis ─────────────────────────
def post_training(model, test_loader, val_loader, test_df, test_labels_raw,
                  test_preds, results, run_id, device, criterion):
    # Logit bias search
    if BIAS_SEARCH:
        log("Logit bias grid search on val set...")
        _, _, _, _, val_labels_raw, val_logits_raw = evaluate(
            model, val_loader, device, criterion,
            split_name="Val (bias search)", return_logits=True)
        val_labels_arr = np.array(val_labels_raw)

        best_f1_b, best_b = 0.0, [0.0, 0.0, 0.0]
        grid_results = []
        for b_off in BIAS_GRID:
            for b_hate in BIAS_GRID:
                preds = (val_logits_raw +
                         torch.tensor([0.0, b_off, b_hate])).argmax(-1).numpy()
                f1 = f1_score(val_labels_arr, preds, average="macro",
                              zero_division=0)
                grid_results.append((b_off, b_hate, f1))
                if f1 > best_f1_b:
                    best_f1_b = f1
                    best_b = [0.0, b_off, b_hate]

        best_bias = torch.tensor(best_b)
        log(f"Best bias: OFF={best_b[1]:.2f}  HATE={best_b[2]:.2f}  "
            f"Val F1={best_f1_b:.4f}")

        # Bias heatmap
        go = sorted(set(r[0] for r in grid_results))
        gh = sorted(set(r[1] for r in grid_results))
        heat = np.zeros((len(go), len(gh)))
        io = {v: i for i, v in enumerate(go)}
        ih = {v: i for i, v in enumerate(gh)}
        for bo, bh, f1 in grid_results:
            heat[io[bo], ih[bh]] = f1
        fig, ax = plt.subplots(figsize=(9, 7))
        sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd",
                    xticklabels=[f"{v:.1f}" for v in gh],
                    yticklabels=[f"{v:.1f}" for v in go],
                    ax=ax, cbar_kws={"label": "F1-Macro"})
        ax.set_xlabel("bias_HATE"); ax.set_ylabel("bias_OFFENSIVE")
        ax.set_title(f"Logit Bias Search  "
                     f"Best: OFF={best_b[1]:.2f} HATE={best_b[2]:.2f} "
                     f"F1={best_f1_b:.4f}", fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "12_logit_bias_search.png", dpi=150, bbox_inches="tight")
        plt.close()
    else:
        best_bias = torch.tensor([0.0, 0.0, 0.0])

    test_acc_b, test_f1_b, test_f1cls_b, test_preds_b, _ = evaluate(
        model, test_loader, device, criterion,
        logit_bias=best_bias, split_name="Test+Bias")

    results.update({
        "logit_bias":             best_b if BIAS_SEARCH else [0.0, 0.0, 0.0],
        "f1_macro_with_bias":     round(test_f1_b, 5),
        "f1_per_class_with_bias": [round(x, 5) for x in test_f1cls_b],
    })

    # Error analysis
    use_biased  = test_f1_b >= results["f1_macro"]
    preds_final = test_preds_b if use_biased else test_preds
    texts_test  = test_df["text"].tolist()
    PAIR_NAMES  = {(0,1): "CLEAN<->OFF", (0,2): "CLEAN<->HATE", (1,2): "OFF<->HATE"}
    error_pairs: dict = defaultdict(list)
    for i, (t, p) in enumerate(zip(test_labels_raw, preds_final)):
        if t != p:
            error_pairs[tuple(sorted([t, p]))].append({
                "idx": i, "true": LABEL_NAMES[t], "pred": LABEL_NAMES[p],
                "text": texts_test[i][:100],
            })

    total_err = sum(len(v) for v in error_pairs.values())
    log(f"Errors: {total_err}/{len(test_labels_raw)} "
        f"({100*total_err/len(test_labels_raw):.1f}%)")

    err_df = pd.DataFrame([{**e, "pair": PAIR_NAMES.get(k, str(k))}
                            for k, vs in error_pairs.items() for e in vs])
    if not err_df.empty:
        err_csv = NB_LOG_DIR / f"errors_{run_id}.csv"
        err_df.to_csv(err_csv, index=False, encoding="utf-8")
        log(f"Errors CSV -> {err_csv}")

    # Loss curves
    if results.get("step_history"):
        steps  = [h["step"]     for h in results["step_history"]]
        losses = [h["loss"]     for h in results["step_history"]]
        kls    = [h["rdrop_kl"] for h in results["step_history"]]
        scs    = [h["supcon"]   for h in results["step_history"]]
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        axes[0].plot(steps, losses, "#2196F3", lw=1.5)
        axes[0].set_title("Combined Loss", fontweight="bold"); axes[0].grid(alpha=0.3)
        axes[1].plot(steps, kls, "#FF5722", lw=1.2)
        axes[1].set_title("R-Drop KL Loss", fontweight="bold"); axes[1].grid(alpha=0.3)
        axes[2].plot(steps, scs, "#4CAF50", lw=1.2)
        axes[2].set_title("SupCon Loss", fontweight="bold"); axes[2].grid(alpha=0.3)
        for ax in axes: ax.set_xlabel("Step")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "14_loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # F1 curves
    if results.get("epoch_history"):
        epochs   = [h["epoch"]  for h in results["epoch_history"]]
        val_f1s  = [h["val_f1"] for h in results["epoch_history"]]
        val_accs = [h["val_acc"] for h in results["epoch_history"]]
        f1c_lst  = [[h["val_f1_cls"][i] for h in results["epoch_history"]]
                    for i in range(3)]
        best_ep  = epochs[val_f1s.index(max(val_f1s))]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(epochs, val_f1s,  "b-o", lw=2,   label="F1-Macro", ms=6)
        axes[0].plot(epochs, val_accs, "g--s", lw=1.5, label="Accuracy",  ms=5)
        axes[0].axvline(best_ep, color="red", ls="--", lw=1.2,
                        label=f"Best ep={best_ep}")
        axes[0].set_title("Val F1-Macro & Accuracy", fontweight="bold")
        axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
        for i, (lbl, col) in enumerate(
                zip(LABEL_NAMES, ["#4CAF50", "#FF9800", "#F44336"])):
            axes[1].plot(epochs, f1c_lst[i], "-o", color=col, lw=2,
                         label=f"F1-{lbl}", ms=6)
        axes[1].axvline(best_ep, color="gray", ls="--", lw=1.2)
        axes[1].set_title("Val F1 per Class", fontweight="bold")
        axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "15_f1_curve.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Confusion matrix
    cm_plot = confusion_matrix(test_labels_raw, preds_final)
    cm_pct  = cm_plot.astype(float) / cm_plot.sum(axis=1, keepdims=True) * 100
    sfx     = " (+bias)" if use_biased else ""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, fmt, title in [
        (axes[0], cm_plot, "d",   f"Count{sfx}"),
        (axes[1], cm_pct,  ".1f", f"Percent{sfx}"),
    ]:
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                    ax=ax, linewidths=0.5, annot_kws={"size": 11})
        ax.set_title(f"Confusion Matrix ({title})", fontweight="bold")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()

    return preds_final, use_biased


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Qwen2.5-3B long-run fine-tune with checkpoint resume")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from checkpoint_latest/")
    parser.add_argument("--eval-only",   action="store_true",
                        help="Skip training, run test evaluation on saved model")
    parser.add_argument("--max-hours",   type=float, default=66.0,
                        help="Maximum wall-clock hours before graceful stop (default 66)")
    parser.add_argument("--max-epochs",  type=int,   default=MAX_EPOCHS)
    parser.add_argument("--early-stop",  type=int,   default=EARLY_STOP)
    parser.add_argument("--ckpt-every",  type=int,   default=CKPT_EVERY,
                        help="Save checkpoint every N optimizer steps")
    parser.add_argument("--batch-size",  type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=LR)
    parser.add_argument("--seed",        type=int,   default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── determine run ID ──────────────────────────────────────────────────────
    resume_state = None
    if args.resume:
        resume_state = load_checkpoint_state()
        if resume_state is None:
            print("[WARN] --resume: no checkpoint found, starting fresh.")
        else:
            print(f"[INFO] Resuming run_id={resume_state['run_id']}  "
                  f"epoch={resume_state['epoch']}  step={resume_state['global_step']}")

    run_id = (resume_state["run_id"] if resume_state
              else datetime.now().strftime("%Y%m%d_%H%M%S"))
    setup_logger(run_id)

    log(f"run_id={run_id}  max_hours={args.max_hours}  "
        f"max_epochs={args.max_epochs}  early_stop={args.early_stop}  "
        f"ckpt_every={args.ckpt_every}")

    # ── GPU ───────────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        log("[WARN] No GPU detected — running on CPU (very slow)")
        device  = torch.device("cpu")
        use_8bit = False
    else:
        device = torch.device("cuda")
        total  = torch.cuda.get_device_properties(0).total_memory / 1e9
        use_8bit = total < 7.5
        log(f"GPU: {torch.cuda.get_device_name(0)}  VRAM={total:.1f}GB  "
            f"mode={'8bit' if use_8bit else 'bfloat16'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    def load_split(name: str, filename: str) -> pd.DataFrame:
        path = DATA_DIR / filename
        df   = pd.read_csv(path)
        if "label_id" in df.columns and "label" not in df.columns:
            df = df.rename(columns={"label_id": "label"})
        if "clean_text" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"clean_text": "text"})
        elif "free_text" in df.columns and "text" not in df.columns:
            df = df.rename(columns={"free_text": "text"})
        df = df.dropna(subset=["text", "label"])
        df["label"] = df["label"].astype(int)
        dist = dict(df["label"].value_counts().sort_index())
        log(f"  {name:16s}: {len(df):,}  {dist}")
        return df[["text", "label"]]

    log("Loading data...")
    train_df = load_split("train (balanced)", "train_augmented.csv")
    val_df   = load_split("val (original)",   "dev_cleaned.csv")
    test_df  = load_split("test (original)",  "test_cleaned.csv")

    class_weights = torch.ones(NUM_LABELS, dtype=torch.float32).to(device)

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bs = args.batch_size
    train_ds = ViHSDDataset(train_df, tokenizer, MAX_LENGTH)
    val_ds   = ViHSDDataset(val_df,   tokenizer, MAX_LENGTH)
    test_ds  = ViHSDDataset(test_df,  tokenizer, MAX_LENGTH)
    train_loader = DataLoader(train_ds, batch_size=bs,     shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs * 4, shuffle=False,
                              num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs * 4, shuffle=False,
                              num_workers=2, pin_memory=True)
    log(f"Train={len(train_loader)} batches  Val={len(val_loader)}  "
        f"Test={len(test_loader)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    log(f"Loading {MODEL_ID} ...")
    load_kw = dict(num_labels=NUM_LABELS, trust_remote_code=True,
                   ignore_mismatched_sizes=True)
    if use_8bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kw["device_map"] = "auto"
    else:
        load_kw["dtype"]      = torch.bfloat16
        load_kw["device_map"] = "auto"

    if args.eval_only or (args.resume and resume_state is not None):
        # Prefer checkpoint_latest (has optimizer state), fall back to best model
        if (CKPT_DIR / "adapter_config.json").exists():
            model_load_path = str(CKPT_DIR)
        elif (MODEL_DIR / "adapter_config.json").exists():
            model_load_path = str(MODEL_DIR)
        else:
            raise FileNotFoundError(
                f"No adapter_config.json found in {CKPT_DIR} or {MODEL_DIR}. "
                "Run without --resume to start fresh training.")
        log(f"Loading fine-tuned model from {model_load_path}")
        from peft import PeftModel
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, **load_kw)
        base.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(base, model_load_path)
        # PeftModel.from_pretrained freezes all params by default.
        # Re-enable requires_grad for exactly the params that were trainable
        # in the original get_peft_model call:
        #   - LoRA adapter weights (lora_A / lora_B / lora_embedding)
        #   - SEQ_CLS head saved via modules_to_save.default  (NOT original_module)
        if not args.eval_only:
            for name, param in model.named_parameters():
                if any(k in name for k in
                       ("lora_A", "lora_B", "lora_embedding", "modules_to_save")):
                    param.requires_grad_(True)
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
    else:
        base = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, **load_kw)
        base.config.pad_token_id = tokenizer.pad_token_id

        # bias init from val distribution
        with torch.no_grad():
            val_cnt  = np.array([val_df["label"].value_counts().get(i, 1)
                                 for i in range(NUM_LABELS)], float)
            log_freq = np.log(val_cnt / val_cnt.sum())
            log_freq -= log_freq.mean()
            head = getattr(base, "score", None) or getattr(base, "classifier", None)
            if head is not None and hasattr(head, "bias") and head.bias is not None:
                head.bias.data = torch.tensor(log_freq, dtype=torch.float32)
                log(f"Classifier bias init: "
                    f"{[round(x, 3) for x in log_freq.tolist()]}")

        lora_cfg = LoraConfig(
            task_type=TaskType.SEQ_CLS, r=LORA_RANK, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none",
        )
        model = get_peft_model(base, lora_cfg)
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if USE_SUPCON:
        log("[INFO] Gradient checkpointing disabled (SupCon needs hidden states)")
    else:
        model.gradient_checkpointing_enable()

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    log(f"Trainable: {tp/1e6:.1f}M / {tt/1e9:.3f}B  ({100*tp/tt:.2f}%)")

    criterion   = CombinedLoss(class_weights, FOCAL_GAMMA, MARGIN,
                               ALPHA_FOCAL, ALPHA_MARGIN, LABEL_SMOOTH).to(device)
    supcon_crit = SupConLoss(SUPCON_TEMP).to(device)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    if USE_LLRD:
        param_groups = build_llrd_groups(model, args.lr, LLRD_DECAY, WEIGHT_DECAY)
    else:
        param_groups = [{"params": [p for p in model.parameters()
                                    if p.requires_grad],
                         "lr": args.lr, "weight_decay": WEIGHT_DECAY}]

    optimizer   = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
    total_steps = (len(train_loader) // GRAD_ACCUM) * args.max_epochs
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler   = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)

    # restore optimizer/scheduler state on resume
    if resume_state and (CKPT_DIR / "optimizer.pt").exists():
        log("Restoring optimizer state...")
        try:
            optimizer.load_state_dict(
                torch.load(CKPT_DIR / "optimizer.pt", weights_only=True))
            scheduler.load_state_dict(
                torch.load(CKPT_DIR / "scheduler.pt", weights_only=True))
            log("Optimizer/scheduler state restored.")
        except (ValueError, RuntimeError) as e:
            log(f"[WARN] Cannot restore optimizer state ({e}). "
                "Continuing with reset optimizer momentum — "
                "model weights and scheduler LR are still from checkpoint.")

    log(f"Total steps={total_steps}  Warmup={warmup_steps}")

    # ── CSV log file ──────────────────────────────────────────────────────────
    log_csv = NB_LOG_DIR / f"train_log_{run_id}.csv"
    if not args.resume or not log_csv.exists():
        with open(log_csv, "w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "step", "loss", "focal", "margin", "rdrop_kl", "supcon",
                "lr", "samples_seen", "elapsed_s",
            ])

    # ── Eval-only mode ────────────────────────────────────────────────────────
    if args.eval_only:
        log("=== EVAL-ONLY mode ===")
        test_acc, test_f1, test_f1cls, test_preds, test_labels = evaluate(
            model, test_loader, device, criterion, split_name="Test")
        log("\n" + classification_report(
            test_labels, test_preds, target_names=LABEL_NAMES, digits=4))
        return

    # ── Training ──────────────────────────────────────────────────────────────
    best_val_f1, total_steps_run, samples_seen, step_history, epoch_history = train(
        model, optimizer, scheduler, train_loader, val_loader,
        device, criterion, supcon_crit, tokenizer,
        run_id, log_csv, args.max_hours, args.max_epochs, args.early_stop,
        args.ckpt_every, resume_state=resume_state,
    )

    # ── Load best checkpoint for test eval ───────────────────────────────────
    # MODEL_DIR only has adapter_config.json after first epoch completes with
    # improvement; fall back to CKPT_DIR (saved on SIGTERM / periodic ckpt).
    from peft import PeftModel
    if (MODEL_DIR / "adapter_config.json").exists():
        eval_path = MODEL_DIR
    elif (CKPT_DIR / "adapter_config.json").exists():
        eval_path = CKPT_DIR
        log("[WARN] No epoch completed — using checkpoint_latest for test eval")
    else:
        eval_path = None

    if eval_path is not None:
        log(f"Loading model from {eval_path} for test evaluation...")
        base2 = AutoModelForSequenceClassification.from_pretrained(
            MODEL_ID, **load_kw)
        base2.config.pad_token_id = tokenizer.pad_token_id
        best_model = PeftModel.from_pretrained(base2, str(eval_path))
    else:
        log("[WARN] No saved adapter found — evaluating with current in-memory model")
        best_model = model

    log("\n" + "="*65)
    log("TEST SET — best checkpoint")
    log("="*65)
    t0_test = time.time()
    test_acc, test_f1, test_f1cls, test_preds, test_labels = evaluate(
        best_model, test_loader, device, criterion, split_name="Test")
    sps = len(test_df) / (time.time() - t0_test)
    log(f"  Speed: {sps:.1f} samples/sec")
    log("\n" + classification_report(
        test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

    results = {
        "model_id":      MODEL_ID,
        "model_size":    "3b_cls_v3",
        "timestamp":     datetime.now().isoformat(),
        "training_data": "train_augmented (24k balanced 8k×3)",
        "lora_rank":     LORA_RANK,
        "quantization":  "8bit" if use_8bit else "bfloat16",
        "loss_type":     f"Focal(g={FOCAL_GAMMA})+Margin+LS({LABEL_SMOOTH})",
        "rdrop":         USE_RDROP,    "rdrop_alpha":   RDROP_ALPHA,
        "llrd":          USE_LLRD,     "llrd_decay":    LLRD_DECAY,
        "supcon":        USE_SUPCON,   "supcon_weight": SUPCON_WEIGHT,
        "accuracy":      round(test_acc, 5),
        "f1_macro":      round(test_f1, 5),
        "f1_per_class":  [round(x, 5) for x in test_f1cls],
        "cm":            confusion_matrix(test_labels, test_preds).tolist(),
        "speed_sps":     round(sps, 1),
        "total_steps":   total_steps_run,
        "samples_seen":  samples_seen,
        "best_val_f1":   round(best_val_f1, 5),
        "step_history":  step_history,
        "epoch_history": epoch_history,
        "test_samples":  len(test_df),
    }

    post_training(best_model, test_loader, val_loader, test_df,
                  test_labels, test_preds, results, run_id, device, criterion)

    log_json = NB_LOG_DIR / f"results_{run_id}.json"
    with open(log_json, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Results -> {log_json}")

    with open(LOG_DIR / "llm_3b_cls_results.json", "w") as f:
        json.dump({
            "model_id":       MODEL_ID, "model_size": "3b_cls_v3",
            "timestamp":      results["timestamp"],
            "accuracy":       results["accuracy"],
            "f1_macro":       results["f1_macro"],
            "f1_per_class":   results["f1_per_class"],
            "cm":             results["cm"],
            "speed_sps":      results["speed_sps"],
            "params_b":       3.09,
            "trainable_pct":  round(LORA_RANK / 16 * 1.64, 2),
            "lora_rank":      LORA_RANK,
            "total_steps":    total_steps_run,
            "time_limited":   False,
            "test_samples":   len(test_df),
        }, f, indent=2, ensure_ascii=False)

    log("\n" + "="*65)
    log("FINAL SUMMARY — Qwen2.5-3B CLS v3")
    log("="*65)
    log(f"  F1-Macro      : {results['f1_macro']:.4f}")
    log(f"  F1 [C/O/H]    : {results['f1_per_class'][0]:.4f} / "
        f"{results['f1_per_class'][1]:.4f} / {results['f1_per_class'][2]:.4f}")
    log(f"  Accuracy      : {results['accuracy']:.4f}")
    log(f"  Best Val F1   : {results['best_val_f1']:.4f}")
    log(f"  Total steps   : {results['total_steps']:,}")
    log(f"  Speed         : {results['speed_sps']:.1f} sps")
    log("="*65)


if __name__ == "__main__":
    main()
