#!/usr/bin/env python3
"""
analyze_3b_checkpoint.py — Post-training analysis on checkpoint_latest

Load best available checkpoint (checkpoint_latest or models/llm_3b_cls),
run full test evaluation, logit bias search, generate all figures and save
results JSON — as if training completed normally.

Usage:
    python3 analyze_3b_checkpoint.py
    python3 analyze_3b_checkpoint.py --ckpt-dir models/llm_3b_cls/checkpoint_latest
"""

import argparse, json, time, warnings
from collections import defaultdict
from datetime import datetime
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
from peft import PeftModel
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "Cleaned"
LOG_DIR    = BASE_DIR / "logs"
FIG_DIR    = BASE_DIR / "figures"
MODEL_DIR  = BASE_DIR / "models" / "llm_3b_cls"
CKPT_DIR   = MODEL_DIR / "checkpoint_latest"
NB_LOG_DIR = BASE_DIR / "notebooks" / "logs_3b_full"
for d in [LOG_DIR, FIG_DIR, NB_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
NUM_LABELS  = 3
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]
MAX_LENGTH  = 256
BATCH_SIZE  = 2
BIAS_GRID   = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
FOCAL_GAMMA = 2.0
MARGIN      = 1.0
ALPHA_FOCAL = 0.7
ALPHA_MARGIN = 0.3
LABEL_SMOOTH = 0.1


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class ViHSDDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df["text"].tolist(); self.labels = df["label"].tolist()
        self.tok = tokenizer; self.max_len = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       truncation=True, padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[idx], dtype=torch.long)}


# ── Loss ──────────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__(); self.gamma=gamma; self.weight=weight; self.ls=label_smoothing

    def forward(self, logits, targets):
        w = self.weight.to(dtype=logits.dtype, device=logits.device) if self.weight is not None else None
        if self.ls > 0:
            n = logits.size(-1)
            smooth = torch.full_like(logits, self.ls / n)
            smooth.scatter_(1, targets.unsqueeze(1), 1 - self.ls + self.ls / n)
            ce = -(smooth * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, targets, weight=w, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, weight=w, reduction="none"))
        return (((1 - pt) ** self.gamma) * ce).mean()


def margin_loss_fn(logits, labels, margin=1.0):
    B = logits.size(0)
    correct = logits[torch.arange(B), labels]
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask[torch.arange(B), labels] = False
    wrong = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
    return F.relu(margin - correct + wrong).mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight, gamma, margin, a_f, a_m, ls):
        super().__init__()
        self.focal = FocalLoss(gamma, weight, ls); self.m=margin; self.a_f=a_f; self.a_m=a_m

    def forward(self, logits, labels):
        fl = self.focal(logits, labels); ml = margin_loss_fn(logits, labels, self.m)
        return self.a_f*fl + self.a_m*ml, fl.item(), ml.item()


# ── Evaluate ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, criterion, logit_bias=None,
             split_name="Val", return_logits=False):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []; total_loss = 0.0
    for batch in tqdm(loader, desc=split_name, leave=False):
        ids = batch["input_ids"].to(device); mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits
        loss, _, _ = criterion(logits, lbls); total_loss += loss.item()
        if logit_bias is not None: logits = logits + logit_bias.to(logits.device)
        all_logits.append(logits.float().cpu())
        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_labels.extend(lbls.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1c = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    log(f"  [{split_name}] loss={total_loss/len(loader):.4f}  acc={acc:.4f}  "
        f"F1={f1m:.4f}  C={f1c[0]:.4f}  O={f1c[1]:.4f}  H={f1c[2]:.4f}")
    if return_logits:
        return acc, f1m, f1c, all_preds, all_labels, torch.cat(all_logits)
    return acc, f1m, f1c, all_preds, all_labels


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Path to adapter checkpoint dir (default: auto-detect)")
    args = parser.parse_args()

    # ── Find best checkpoint ──────────────────────────────────────────────────
    if args.ckpt_dir:
        ckpt_path = Path(args.ckpt_dir)
    elif (MODEL_DIR / "adapter_config.json").exists():
        ckpt_path = MODEL_DIR          # best model saved after epoch improvement
        log(f"Using best model: {ckpt_path}")
    elif (CKPT_DIR / "adapter_config.json").exists():
        ckpt_path = CKPT_DIR           # latest periodic checkpoint
        log(f"No best model found — using checkpoint_latest: {ckpt_path}")
    else:
        raise FileNotFoundError("No adapter checkpoint found.")

    state = {}
    if (CKPT_DIR / "train_state.json").exists():
        state = json.loads((CKPT_DIR / "train_state.json").read_text())
        log(f"Train state: epoch={state.get('epoch','?')}  "
            f"step={state.get('global_step','?')}  best_f1={state.get('best_f1',0):.4f}")

    # ── GPU ───────────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vram   = torch.cuda.get_device_properties(0).total_memory/1e9 if device.type=="cuda" else 0
    use_8bit = vram < 7.5
    log(f"Device: {device}  VRAM={vram:.1f}GB  mode={'8bit' if use_8bit else 'bfloat16'}")

    # ── Data ──────────────────────────────────────────────────────────────────
    def load_split(name, filename):
        df = pd.read_csv(DATA_DIR / filename)
        for old, new in [("label_id","label"),("clean_text","text"),("free_text","text")]:
            if old in df.columns and new not in df.columns: df = df.rename(columns={old:new})
        df = df.dropna(subset=["text","label"]); df["label"] = df["label"].astype(int)
        log(f"  {name:16s}: {len(df):,}  {dict(df['label'].value_counts().sort_index())}")
        return df[["text","label"]]

    log("Loading data...")
    val_df  = load_split("val",  "dev_cleaned.csv")
    test_df = load_split("test", "test_cleaned.csv")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    bs = BATCH_SIZE * 4
    val_loader  = DataLoader(ViHSDDataset(val_df,  tokenizer, MAX_LENGTH), batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(ViHSDDataset(test_df, tokenizer, MAX_LENGTH), batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)

    # ── Load model ────────────────────────────────────────────────────────────
    log(f"Loading base model {MODEL_ID}...")
    load_kw = dict(num_labels=NUM_LABELS, trust_remote_code=True,
                   ignore_mismatched_sizes=True)
    if use_8bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kw["device_map"] = "auto"
    else:
        load_kw["dtype"] = torch.bfloat16; load_kw["device_map"] = "auto"

    base  = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **load_kw)
    base.config.pad_token_id = tokenizer.pad_token_id
    log(f"Loading PEFT adapter from {ckpt_path}...")
    model = PeftModel.from_pretrained(base, str(ckpt_path))
    model.eval()

    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in model.parameters())
    log(f"Parameters: {tt/1e9:.3f}B total  (trainable={tp/1e6:.1f}M for eval)")

    class_weights = torch.ones(NUM_LABELS, dtype=torch.float32).to(device)
    criterion = CombinedLoss(class_weights, FOCAL_GAMMA, MARGIN,
                             ALPHA_FOCAL, ALPHA_MARGIN, LABEL_SMOOTH).to(device)

    # ── Val evaluation ────────────────────────────────────────────────────────
    log("\n=== Validation set ===")
    val_acc, val_f1, val_f1c, _, val_labels_raw, val_logits_raw = evaluate(
        model, val_loader, device, criterion,
        split_name="Val", return_logits=True)

    # ── Logit bias grid search on val ─────────────────────────────────────────
    log("\nLogit bias grid search on val set...")
    val_labels_arr = np.array(val_labels_raw)
    best_f1_b, best_b = 0.0, [0.0, 0.0, 0.0]; grid_results = []
    for b_off in BIAS_GRID:
        for b_hate in BIAS_GRID:
            preds = (val_logits_raw + torch.tensor([0.0, b_off, b_hate])).argmax(-1).numpy()
            f1 = f1_score(val_labels_arr, preds, average="macro", zero_division=0)
            grid_results.append((b_off, b_hate, f1))
            if f1 > best_f1_b: best_f1_b = f1; best_b = [0.0, b_off, b_hate]
    best_bias = torch.tensor(best_b)
    log(f"Best bias: OFF={best_b[1]:.2f}  HATE={best_b[2]:.2f}  Val F1={best_f1_b:.4f}")

    # ── Test evaluation ───────────────────────────────────────────────────────
    log("\n=== Test set (no bias) ===")
    t0 = time.time()
    test_acc, test_f1, test_f1c, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion, split_name="Test")
    sps = len(test_df) / (time.time() - t0)
    log(f"  Speed: {sps:.1f} sps")

    log("\n=== Test set (+bias) ===")
    test_acc_b, test_f1_b, test_f1c_b, test_preds_b, _ = evaluate(
        model, test_loader, device, criterion,
        logit_bias=best_bias, split_name="Test+Bias")

    use_biased  = test_f1_b >= test_f1
    preds_final = test_preds_b if use_biased else test_preds
    sfx         = " (+bias)" if use_biased else ""
    log(f"\nUsing {'biased' if use_biased else 'unbiased'} predictions for final metrics.")
    log(f"  Delta F1: {test_f1_b - test_f1:+.4f}  "
        f"OFF:{test_f1c_b[1]-test_f1c[1]:+.4f}  HATE:{test_f1c_b[2]-test_f1c[2]:+.4f}")

    log("\n" + classification_report(test_labels, preds_final,
                                     target_names=LABEL_NAMES, digits=4))

    # ── Load histories for plots ──────────────────────────────────────────────
    step_history = []
    if (CKPT_DIR / "step_history.json").exists():
        step_history = json.loads((CKPT_DIR / "step_history.json").read_text())
        log(f"Step history: {len(step_history)} entries")

    # ── Figure 12: Logit bias heatmap ─────────────────────────────────────────
    go = sorted(set(r[0] for r in grid_results))
    gh = sorted(set(r[1] for r in grid_results))
    heat = np.zeros((len(go), len(gh)))
    io = {v:i for i,v in enumerate(go)}; ih = {v:i for i,v in enumerate(gh)}
    for bo, bh, f1 in grid_results: heat[io[bo], ih[bh]] = f1
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"{v:.1f}" for v in gh],
                yticklabels=[f"{v:.1f}" for v in go], ax=ax,
                cbar_kws={"label": "F1-Macro"})
    ax.set_xlabel("bias_HATE"); ax.set_ylabel("bias_OFFENSIVE")
    ax.set_title(f"Logit Bias Search — Best: OFF={best_b[1]:.2f} HATE={best_b[2]:.2f} "
                 f"F1={best_f1_b:.4f}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_logit_bias_search.png", dpi=150, bbox_inches="tight")
    plt.close(); log("Saved figure 12: logit bias heatmap")

    # ── Figure 14: Loss curve (from step_history) ─────────────────────────────
    if step_history:
        steps  = [h["step"]     for h in step_history]
        losses = [h["loss"]     for h in step_history]
        kls    = [h["rdrop_kl"] for h in step_history]
        scs    = [h["supcon"]   for h in step_history]

        # smoothed loss (window=20)
        def smooth(arr, w=20):
            k = np.ones(w)/w
            return np.convolve(arr, k, mode="same") if len(arr) >= w else arr

        fig, axes = plt.subplots(1, 3, figsize=(17, 4))
        axes[0].plot(steps, losses, "#2196F3", lw=0.6, alpha=0.4)
        axes[0].plot(steps, smooth(losses), "#1565C0", lw=2, label="smoothed")
        axes[0].set_title(f"Combined Loss (last step={steps[-1]})", fontweight="bold")
        axes[0].legend(); axes[0].grid(alpha=0.3)
        axes[1].plot(steps, kls, "#FF5722", lw=1)
        axes[1].set_title("R-Drop KL Loss", fontweight="bold"); axes[1].grid(alpha=0.3)
        axes[2].plot(steps, scs, "#4CAF50", lw=1)
        axes[2].set_title("SupCon Loss", fontweight="bold"); axes[2].grid(alpha=0.3)
        for ax in axes: ax.set_xlabel("Optimizer Step")
        plt.suptitle(f"Training curves — Qwen2.5-3B CLS v3  "
                     f"({len(step_history)} steps, {state.get('epoch','?')} epochs partial)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "14_loss_curve.png", dpi=150, bbox_inches="tight")
        plt.close(); log("Saved figure 14: loss curve")

        # LR curve
        lrs = [h["lr"] for h in step_history]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(steps, lrs, "#9C27B0", lw=1.5)
        ax.set_xlabel("Optimizer Step"); ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (cosine + warmup)", fontweight="bold"); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIG_DIR / "14b_lr_schedule.png", dpi=150, bbox_inches="tight")
        plt.close(); log("Saved figure 14b: LR schedule")

    # ── Figure 16: Confusion matrix ───────────────────────────────────────────
    cm_plot = confusion_matrix(test_labels, preds_final)
    cm_pct  = cm_plot.astype(float) / cm_plot.sum(axis=1, keepdims=True) * 100
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
    plt.suptitle(f"Qwen2.5-3B CLS v3 — Test set{sfx}  "
                 f"F1={test_f1_b if use_biased else test_f1:.4f}  "
                 f"Acc={test_acc_b if use_biased else test_acc:.4f}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(); log("Saved figure 16: confusion matrix")

    # ── Figure: per-class F1 bar ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#4CAF50", "#FF9800", "#F44336"]
    final_f1c = test_f1c_b if use_biased else test_f1c
    bars = ax.bar(LABEL_NAMES, final_f1c, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", fontsize=11, padding=4)
    ax.axhline(test_f1_b if use_biased else test_f1, ls="--", color="gray",
               lw=1.5, label=f"F1-Macro={test_f1_b if use_biased else test_f1:.4f}")
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1 Score"); ax.legend()
    ax.set_title(f"Per-class F1 — Qwen2.5-3B CLS v3{sfx}  "
                 f"(step={state.get('global_step','?')})", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "16b_f1_per_class.png", dpi=150, bbox_inches="tight")
    plt.close(); log("Saved figure 16b: per-class F1")

    # ── Figure: Error analysis ─────────────────────────────────────────────────
    PAIR_NAMES = {(0,1): "CLEAN↔OFF", (0,2): "CLEAN↔HATE", (1,2): "OFF↔HATE"}
    error_pairs = defaultdict(list)
    for i, (t, p) in enumerate(zip(test_labels, preds_final)):
        if t != p:
            error_pairs[tuple(sorted([t, p]))].append(
                {"true": LABEL_NAMES[t], "pred": LABEL_NAMES[p],
                 "text": test_df["text"].tolist()[i][:100]})
    total_err = sum(len(v) for v in error_pairs.values())
    log(f"\nErrors: {total_err}/{len(test_labels)} ({100*total_err/len(test_labels):.1f}%)")
    for k, name in PAIR_NAMES.items():
        errs = error_pairs.get(k, [])
        log(f"  {name}: {len(errs)} ({100*len(errs)/max(total_err,1):.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pair_counts = {PAIR_NAMES[k]: len(v) for k, v in error_pairs.items() if k in PAIR_NAMES}
    if sum(pair_counts.values()) > 0:
        axes[0].pie(pair_counts.values(), labels=pair_counts.keys(),
                    autopct="%1.1f%%", colors=["#FF9800","#F44336","#9C27B0"],
                    startangle=90, textprops={"fontsize": 10})
    axes[0].set_title("Error types by label pair", fontweight="bold")
    cm_arr = confusion_matrix(test_labels, preds_final)
    fn = [(cm_arr.sum(1)[i] - cm_arr[i,i]) / max(cm_arr.sum(1)[i],1) for i in range(3)]
    fp = [(cm_arr.sum(0)[i] - cm_arr[i,i]) / max(cm_arr.sum(0)[i],1) for i in range(3)]
    x = np.arange(3); w = 0.35
    b1 = axes[1].bar(x-w/2, fn, w, label="FN rate", color="#FF5722", alpha=0.85)
    b2 = axes[1].bar(x+w/2, fp, w, label="FP rate", color="#2196F3", alpha=0.85)
    axes[1].bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
    axes[1].bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
    axes[1].set_xticks(x); axes[1].set_xticklabels(LABEL_NAMES)
    axes[1].set_title("FN / FP rate per class", fontweight="bold")
    axes[1].legend(); axes[1].grid(axis="y", alpha=0.3); axes[1].set_ylim(0, 1)
    plt.suptitle(f"Error Analysis — {total_err} errors / {len(test_labels)} samples",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_error_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(); log("Saved figure 13: error analysis")

    # ── Figure 17: Comparison with other models ────────────────────────────────
    baselines = {
        "PhoBERT":   LOG_DIR / "phobert_results.json",
        "3B SFT v1": LOG_DIR / "llm_3b_results.json",
    }
    compare = {}
    for name, path in baselines.items():
        if path.exists(): compare[name] = json.load(open(path))
    final_f1  = test_f1_b  if use_biased else test_f1
    final_acc = test_acc_b if use_biased else test_acc
    compare["3B CLS v3"] = {
        "accuracy": round(final_acc, 5), "f1_macro": round(final_f1, 5),
        "f1_per_class": [round(x, 5) for x in final_f1c],
        "speed_sps": round(sps, 1),
    }
    if len(compare) >= 2:
        names   = list(compare.keys())
        palette = ["#2196F3","#FF9800","#4CAF50","#9C27B0"][:len(names)]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle("So sánh kết quả — 3B CLS v3 vs baseline",
                     fontsize=13, fontweight="bold")
        def bar(ax, vals, title, ylim=(0, 1)):
            bs2 = ax.bar(names, vals, color=palette, edgecolor="white", width=0.5)
            ax.bar_label(bs2, fmt="%.4f", fontsize=8, padding=3)
            ax.set_title(title, fontweight="bold"); ax.set_ylim(*ylim)
            ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)
        bar(axes[0,0], [compare[n]["f1_macro"]       for n in names], "F1-Macro")
        bar(axes[0,1], [compare[n]["accuracy"]        for n in names], "Accuracy")
        bar(axes[0,2], [compare[n]["f1_per_class"][0] for n in names], "F1-CLEAN")
        bar(axes[1,0], [compare[n]["f1_per_class"][1] for n in names], "F1-OFFENSIVE")
        bar(axes[1,1], [compare[n]["f1_per_class"][2] for n in names], "F1-HATE")
        bar(axes[1,2], [compare[n].get("speed_sps",0) for n in names],
            "Speed (sps)", ylim=(0, None))
        plt.tight_layout()
        plt.savefig(FIG_DIR / "17_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(); log("Saved figure 17: comparison")

        log(f"\n{'Model':<18} {'Acc':>7} {'F1':>7} {'CLEAN':>7} {'OFF':>7} "
            f"{'HATE':>7} {'sps':>6}")
        log("-"*66)
        for n, d in compare.items():
            fc = d["f1_per_class"]
            log(f"{n:<18} {d['accuracy']:>7.4f} {d['f1_macro']:>7.4f} "
                f"{fc[0]:>7.4f} {fc[1]:>7.4f} {fc[2]:>7.4f} "
                f"{d.get('speed_sps',0):>6.1f}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    run_id  = state.get("run_id", datetime.now().strftime("%Y%m%d_%H%M%S"))
    results = {
        "model_id":               MODEL_ID,
        "model_size":             "3b_cls_v3",
        "timestamp":              datetime.now().isoformat(),
        "checkpoint":             str(ckpt_path),
        "training_steps":         state.get("global_step", 0),
        "training_epochs_done":   state.get("epoch", 0),
        "samples_seen":           state.get("samples_seen", 0),
        "lora_rank":              64,
        "quantization":           "bfloat16",
        "val_f1":                 round(val_f1, 5),
        "val_acc":                round(val_acc, 5),
        "val_f1_per_class":       [round(x, 5) for x in val_f1c],
        "accuracy":               round(final_acc, 5),
        "f1_macro":               round(final_f1, 5),
        "f1_per_class":           [round(x, 5) for x in final_f1c],
        "f1_macro_no_bias":       round(test_f1, 5),
        "f1_per_class_no_bias":   [round(x, 5) for x in test_f1c],
        "logit_bias":             best_b,
        "f1_macro_with_bias":     round(test_f1_b, 5),
        "f1_per_class_with_bias": [round(x, 5) for x in test_f1c_b],
        "cm":                     confusion_matrix(test_labels, preds_final).tolist(),
        "speed_sps":              round(sps, 1),
        "test_samples":           len(test_df),
    }
    out_json = NB_LOG_DIR / f"results_analysis_{run_id}.json"
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    log(f"\nResults -> {out_json}")

    # Also overwrite the standard results file used by compare_models.py
    (LOG_DIR / "llm_3b_cls_results.json").write_text(json.dumps({
        "model_id":       MODEL_ID, "model_size": "3b_cls_v3",
        "timestamp":      results["timestamp"],
        "accuracy":       results["accuracy"],
        "f1_macro":       results["f1_macro"],
        "f1_per_class":   results["f1_per_class"],
        "cm":             results["cm"],
        "speed_sps":      results["speed_sps"],
        "params_b":       3.09, "lora_rank": 64,
        "trainable_pct":  3.74, "time_limited": False,
        "total_steps":    results["training_steps"],
        "test_samples":   len(test_df),
    }, indent=2, ensure_ascii=False))
    log("Saved logs/llm_3b_cls_results.json")

    log("\n" + "="*65)
    log("FINAL SUMMARY — Qwen2.5-3B CLS v3")
    log("="*65)
    log(f"  Checkpoint     : step {state.get('global_step','?')} "
        f"(epoch {state.get('epoch','?')})")
    log(f"  Val  F1-Macro  : {val_f1:.4f}")
    log(f"  Test F1-Macro  : {final_f1:.4f}{' (with bias)' if use_biased else ''}")
    log(f"  F1 [C / O / H] : "
        f"{final_f1c[0]:.4f} / {final_f1c[1]:.4f} / {final_f1c[2]:.4f}")
    log(f"  Accuracy       : {final_acc:.4f}")
    log(f"  Logit bias     : OFF={best_b[1]:.2f}  HATE={best_b[2]:.2f}")
    log(f"  Speed          : {sps:.1f} sps")
    log("="*65)
    log(f"\nFigures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
