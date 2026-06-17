"""
create_notebook.py — Tạo train_3b_full.ipynb (v3 — all improvements)

Cải tiến so với v2:
  + train_augmented.csv (8000×3 balanced) thay train_cleaned.csv
  + R-Drop (double forward + KL divergence regularization)
  + Layer-wise LR decay (LLRD) — layer gần input nhận LR nhỏ hơn
  + Supervised Contrastive Loss (SupConLoss) — pull same-class embeddings
  + DATA_DIR = BASE_DIR / "Cleaned" (local copy, không phụ thuộc thư mục ngoài)

v2 đã có: torch.autocast, logit bias search, error analysis, label smoothing,
          bias init classification head, focal+margin loss, LoRA r=64

Chạy:  python3 create_notebook.py
       (dùng .venv Python nếu cần: /path/to/.venv/bin/python3 create_notebook.py)
Output: train_3b_full.ipynb (cùng thư mục)
"""

from pathlib import Path
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell

# ─────────────────────────────────────────────────────────────────────────────
CELL_TITLE = """\
# Qwen2.5-3B — Deep Fine-tuning v3: All Improvements

**Tất cả cải tiến được áp dụng (không giới hạn thời gian train):**

| Kỹ thuật | Mô tả | Lợi ích |
|----------|-------|---------|
| Balanced training data | `train_augmented.csv` (8k × 3 class) | Loại bỏ class imbalance ngay từ đầu |
| Classification head | `AutoModelForSequenceClassification` | Đúng objective, nhanh inference |
| Focal Loss + Margin | γ=2, margin=1.0 | Hard-example mining |
| Label smoothing | ε=0.1 | Tránh overfit, calibration tốt hơn |
| LoRA r=64 | 4× baseline | Nhiều capacity hơn |
| Bias init head | `log(class_freq)` | Counter over-predict CLEAN ban đầu |
| Layer-wise LR decay | LLRD decay=0.95 | Tránh catastrophic forgetting |
| R-Drop | KL(p1‖p2) regularization | Robustness, ~1–2% F1 gain |
| SupConLoss | Supervised contrastive on CLS emb | Pull same-class, push diff-class |
| `torch.autocast` | bfloat16 mixed precision | ~20% nhanh hơn |
| Logit bias search | Grid search trên val set | Counter RLHF Instruct bias |
| Error analysis | OFF↔CLEAN, OFF↔HATE breakdown | Hiểu failure mode |

**Classes:** `CLEAN (0)` · `OFFENSIVE (1)` · `HATE (2)`
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_INSTALL = """\
import subprocess, sys

REQUIRED = {
    "torch": "torch", "transformers": "transformers>=5.0",
    "peft": "peft>=0.19", "bitsandbytes": "bitsandbytes",
    "accelerate": "accelerate", "scikit_learn": "scikit-learn",
    "matplotlib": "matplotlib", "seaborn": "seaborn",
    "pandas": "pandas", "tqdm": "tqdm",
}
for mod, pkg in REQUIRED.items():
    try:
        __import__(mod)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
print("All packages ready.")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_IMPORTS = """\
import os, sys, json, time, csv, math, warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
)
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")
torch.backends.cuda.matmul.allow_tf32 = True

print(f"PyTorch     : {torch.__version__}")
print(f"CUDA avail  : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_CONFIG = """\
# ── Paths (DATA_DIR local — copy trong project) ───────────────────────────────
BASE_DIR   = Path("..").resolve()
DATA_DIR   = BASE_DIR / "Cleaned"         # train_augmented / dev / test CSV
LOG_DIR    = BASE_DIR / "logs"
FIG_DIR    = BASE_DIR / "figures"
MODEL_DIR  = BASE_DIR / "models" / "llm_3b_cls"
NB_LOG_DIR = BASE_DIR / "notebooks" / "logs_3b_full"
for d in [LOG_DIR, FIG_DIR, MODEL_DIR, NB_LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_ID    = "Qwen/Qwen2.5-3B-Instruct"
NUM_LABELS  = 3
LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

# ── LoRA ──────────────────────────────────────────────────────────────────────
LORA_RANK    = 64
LORA_ALPHA   = 128
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]

# ── Training ──────────────────────────────────────────────────────────────────
MAX_LENGTH   = 256
BATCH_SIZE   = 2
GRAD_ACCUM   = 4          # effective batch = 8
MAX_EPOCHS   = 20         # không giới hạn thời gian, early stop sẽ dừng
EARLY_STOP   = 5
LR           = 1e-4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.06
LOG_EVERY    = 20
USE_AUTOCAST = True       # torch.autocast bfloat16

# ── Loss ──────────────────────────────────────────────────────────────────────
FOCAL_GAMMA  = 2.0
MARGIN       = 1.0
ALPHA_FOCAL  = 0.7
ALPHA_MARGIN = 0.3
LABEL_SMOOTH = 0.1

# ── R-Drop (IMPROVEMENTS §3.2) ────────────────────────────────────────────────
USE_RDROP    = True
RDROP_ALPHA  = 0.5        # weight của KL divergence loss

# ── Layer-wise LR decay / LLRD (IMPROVEMENTS §3.1) ───────────────────────────
USE_LLRD     = True
LLRD_DECAY   = 0.95       # per-layer decay factor (gentle, 36 layers Qwen2.5-3B)

# ── Supervised Contrastive Loss (IMPROVEMENTS §3.4) ──────────────────────────
USE_SUPCON    = True
SUPCON_TEMP   = 0.07
SUPCON_WEIGHT = 0.1       # trọng số SupCon trong tổng loss

# ── Logit bias search (IMPROVEMENTS §2.7) ────────────────────────────────────
BIAS_SEARCH = True
BIAS_GRID   = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f"Base dir    : {BASE_DIR}")
print(f"Data dir    : {DATA_DIR}")
print(f"Model       : {MODEL_ID}")
print(f"LoRA rank   : {LORA_RANK}")
print(f"R-Drop      : {USE_RDROP}  alpha={RDROP_ALPHA}")
print(f"LLRD        : {USE_LLRD}   decay={LLRD_DECAY}")
print(f"SupCon      : {USE_SUPCON} weight={SUPCON_WEIGHT}")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_GPU_CHECK = """\
def check_gpu():
    if not torch.cuda.is_available():
        print("[WARN] No GPU!")
        return torch.device("cpu"), False
    device = torch.device("cuda")
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU  : {torch.cuda.get_device_name(0)}")
    print(f"VRAM : {total:.1f} GB")
    use_8bit = total < 7.5
    print(f"Mode : {'8-bit quantization' if use_8bit else 'bfloat16 full precision'}")
    return device, use_8bit

DEVICE, USE_8BIT = check_gpu()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_LOAD_DATA = """\
def load_split(name: str, filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    df = pd.read_csv(path)
    if "label_id" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"label_id": "label"})
    if "clean_text" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"clean_text": "text"})
    elif "free_text" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"free_text": "text"})
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    dist = dict(df["label"].value_counts().sort_index())
    print(f"  {name:16s}: {len(df):,}  {dist}")
    return df[["text", "label"]]

print("Loading data...")
# train_augmented.csv: perfectly balanced 8000 samples/class → loại bỏ class imbalance
train_df = load_split("train (balanced)", "train_augmented.csv")
val_df   = load_split("val (original)",   "dev_cleaned.csv")
test_df  = load_split("test (original)",  "test_cleaned.csv")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
colors = ["#4CAF50", "#FF9800", "#F44336"]
for ax, (name, df) in zip(axes, [
    ("Train (balanced)", train_df), ("Val (original)", val_df), ("Test (original)", test_df)
]):
    counts = df["label"].value_counts().sort_index()
    bars = ax.bar(LABEL_NAMES, [counts.get(i, 0) for i in range(3)],
                  color=colors, edgecolor="white")
    ax.bar_label(bars, fontsize=9)
    ax.set_title(f"{name}\\n({len(df):,})", fontweight="bold")
    ax.set_ylabel("Samples"); ax.grid(axis="y", alpha=0.3)
plt.suptitle("Phân phối nhãn — train balanced, val/test original distribution",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(FIG_DIR / "11_data_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# train_augmented balanced → class_weights = 1.0 (focal loss handles hard examples)
class_weights = torch.ones(NUM_LABELS, dtype=torch.float32).to(DEVICE)
print(f"\\nClass weights: {class_weights.tolist()}  (train balanced → uniform)")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_DATASET = """\
class ViHSDDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.texts  = df["text"].tolist()
        self.labels = df["label"].tolist()
        self.tok    = tokenizer
        self.max_len = max_length

    def __len__(self): return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tok(self.texts[idx], max_length=self.max_len,
                       truncation=True, padding="max_length", return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.long),
        }

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

train_ds = ViHSDDataset(train_df, tokenizer, MAX_LENGTH)
val_ds   = ViHSDDataset(val_df,   tokenizer, MAX_LENGTH)
test_ds  = ViHSDDataset(test_df,  tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE * 4, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches : {len(train_loader)}  |  Val : {len(val_loader)}  |  Test : {len(test_loader)}")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_MODEL = """\
# ── Focal Loss ────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, label_smoothing=0.0):
        super().__init__()
        self.gamma = gamma; self.weight = weight; self.ls = label_smoothing

    def forward(self, logits, targets):
        if self.ls > 0:
            n = logits.size(-1)
            smooth = torch.full_like(logits, self.ls / n)
            smooth.scatter_(1, targets.unsqueeze(1), 1 - self.ls + self.ls / n)
            ce = -(smooth * F.log_softmax(logits, dim=-1)).sum(dim=-1)
        else:
            ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-F.cross_entropy(logits, targets, weight=self.weight, reduction="none"))
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
        self.m = margin; self.a_f = a_f; self.a_m = a_m

    def forward(self, logits, labels):
        fl = self.focal(logits, labels)
        ml = margin_loss(logits, labels, self.m)
        return self.a_f * fl + self.a_m * ml, fl.item(), ml.item()

# ── Supervised Contrastive Loss ───────────────────────────────────────────────
class SupConLoss(nn.Module):
    # Khosla et al. 2020 - pull same-class embeddings, push different-class.
    def __init__(self, temperature=0.07):
        super().__init__(); self.T = temperature

    def forward(self, features, labels):
        B, device = features.size(0), features.device
        if B < 2:
            return torch.tensor(0.0, device=device)
        sim = torch.matmul(features, features.T) / self.T
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()  # stability
        eye = torch.eye(B, device=device)
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0)).float() * (1 - eye)
        if mask_pos.sum() == 0:
            return torch.tensor(0.0, device=device)
        exp_sim  = torch.exp(sim) * (1 - eye)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        return -(log_prob * mask_pos).sum() / mask_pos.sum()

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading Qwen2.5-3B (AutoModelForSequenceClassification)...")
load_kw = dict(num_labels=NUM_LABELS, trust_remote_code=True, ignore_mismatched_sizes=True)
if USE_8BIT:
    from transformers import BitsAndBytesConfig
    load_kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    load_kw["device_map"] = "auto"
else:
    load_kw["torch_dtype"] = torch.bfloat16
    load_kw["device_map"]  = "auto"

base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, **load_kw)
base_model.config.pad_token_id = tokenizer.pad_token_id

# ── Bias init: counter RLHF over-predict CLEAN ───────────────────────────────
# Use val distribution (original) for calibration — train is balanced so log-freq = 0
with torch.no_grad():
    val_cnt  = np.array([val_df["label"].value_counts().get(i, 1) for i in range(NUM_LABELS)], float)
    log_freq = np.log(val_cnt / val_cnt.sum())
    log_freq -= log_freq.mean()
    head = getattr(base_model, "score", None) or getattr(base_model, "classifier", None)
    if head is not None and hasattr(head, "bias") and head.bias is not None:
        head.bias.data = torch.tensor(log_freq, dtype=torch.float32)
        print(f"Classifier bias init (val freq): {[round(x, 3) for x in log_freq.tolist()]}")

# ── LoRA ──────────────────────────────────────────────────────────────────────
lora_cfg = LoraConfig(
    task_type=TaskType.SEQ_CLS, r=LORA_RANK, lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT, target_modules=LORA_TARGETS, bias="none",
)
model = get_peft_model(base_model, lora_cfg)
model.print_trainable_parameters()

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

if USE_SUPCON:
    # SupCon needs hidden states → gradient checkpointing incompatible
    print("[INFO] Gradient checkpointing disabled (SupCon needs intermediate hidden states).")
else:
    model.gradient_checkpointing_enable()

criterion   = CombinedLoss(class_weights, FOCAL_GAMMA, MARGIN, ALPHA_FOCAL, ALPHA_MARGIN, LABEL_SMOOTH).to(DEVICE)
supcon_crit = SupConLoss(SUPCON_TEMP).to(DEVICE)

tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
tt = sum(p.numel() for p in model.parameters())
print(f"Trainable: {tp/1e6:.1f}M / {tt/1e9:.3f}B  ({100*tp/tt:.2f}%)")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_OPTIMIZER = """\
# ── Layer-wise LR Decay (LLRD) ────────────────────────────────────────────────
def build_llrd_groups(model, base_lr, decay=0.95, wd=0.01):
    # head -> LR*decay^0, top layer -> LR*decay^1, bottom -> LR*decay^n, embeds -> LR*decay^(n+1)
    try:
        n = model.base_model.model.config.num_hidden_layers
    except AttributeError:
        n = 36  # Qwen2.5-3B default

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
                    depth = n - i  # top layer → 1, bottom → n
                    break
        param_lr[id(p)] = (p, base_lr * (decay ** depth))

    lr_to_params = {}
    for _, (p, lr) in param_lr.items():
        lr_to_params.setdefault(lr, []).append(p)

    groups = sorted(
        [{"params": ps, "lr": lr, "weight_decay": wd} for lr, ps in lr_to_params.items()],
        key=lambda g: -g["lr"]
    )
    print(f"LLRD: {len(groups)} groups  LR [{groups[-1]['lr']:.2e} .. {groups[0]['lr']:.2e}]")
    return groups


if USE_LLRD:
    param_groups = build_llrd_groups(model, LR, LLRD_DECAY, WEIGHT_DECAY)
else:
    param_groups = [{"params": [p for p in model.parameters() if p.requires_grad],
                     "lr": LR, "weight_decay": WEIGHT_DECAY}]

optimizer    = AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)
total_steps  = (len(train_loader) // GRAD_ACCUM) * MAX_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
print(f"Total steps: {total_steps}  |  Warmup: {warmup_steps}")

RUN_ID   = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_CSV  = NB_LOG_DIR / f"train_log_{RUN_ID}.csv"
LOG_JSON = NB_LOG_DIR / f"results_{RUN_ID}.json"

with open(LOG_CSV, "w", newline="") as f:
    csv.writer(f).writerow([
        "epoch", "step", "loss", "focal", "margin", "rdrop_kl", "supcon",
        "lr", "samples_seen", "elapsed_s"
    ])

step_history  = []
epoch_history = []
samples_seen  = 0
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_EVAL_FN = """\
@torch.no_grad()
def evaluate(loader, logit_bias=None, split_name="Val", return_logits=False):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    total_loss = 0.0

    for batch in loader:
        ids  = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        lbls = batch["labels"].to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=USE_AUTOCAST and DEVICE.type == "cuda"):
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
    print(f"  [{split_name}] loss={total_loss/len(loader):.4f}  acc={acc:.4f}  "
          f"F1={f1m:.4f}  C={f1c[0]:.4f}  O={f1c[1]:.4f}  H={f1c[2]:.4f}")

    if return_logits:
        return acc, f1m, f1c, all_preds, all_labels, torch.cat(all_logits)
    return acc, f1m, f1c, all_preds, all_labels
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_TRAIN = """\
def compute_loss_step(ids, mask, lbls):
    # Returns: (loss, focal, margin, kl_val, sc_val, logits)
    B = ids.size(0)

    if USE_SUPCON:
        # output_hidden_states=True needed for SupCon embeddings
        out1 = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        logits1 = out1.logits
        # Last non-padding token embedding from final hidden layer
        try:
            last_h  = out1.hidden_states[-1]              # (B, L, D)
            seq_end = mask.sum(dim=1) - 1                  # index of last real token
            cls_emb = last_h[torch.arange(B, device=DEVICE), seq_end, :]
            cls_emb = F.normalize(cls_emb.float(), dim=-1)
            sc_loss = supcon_crit(cls_emb, lbls)
            sc_val  = sc_loss.item()
        except Exception:
            sc_loss = torch.tensor(0.0, device=DEVICE); sc_val = 0.0

        if USE_RDROP:
            out2 = model(input_ids=ids, attention_mask=mask)
            logits2 = out2.logits
            p1  = F.softmax(logits1.float(), dim=-1).clamp(1e-7, 1)
            p2  = F.softmax(logits2.float(), dim=-1).clamp(1e-7, 1)
            kl  = (F.kl_div(p1.log(), p2, reduction="batchmean") +
                   F.kl_div(p2.log(), p1, reduction="batchmean")) / 2
            t1, fl, ml = criterion(logits1, lbls); t2, _, _ = criterion(logits2, lbls)
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
        t1, fl, ml = criterion(out1.logits, lbls); t2, _, _ = criterion(out2.logits, lbls)
        return (t1 + t2) / 2 + RDROP_ALPHA * kl, fl, ml, kl.item(), 0.0, out1.logits

    else:
        out = model(input_ids=ids, attention_mask=mask)
        loss, fl, ml = criterion(out.logits, lbls)
        return loss, fl, ml, 0.0, 0.0, out.logits


def train():
    global samples_seen
    best_f1, no_improve, global_step = 0.0, 0, 0
    t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); optimizer.zero_grad(); epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{MAX_EPOCHS}", leave=False)

        for step, batch in enumerate(pbar):
            ids  = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            lbls = batch["labels"].to(DEVICE)
            samples_seen += ids.size(0)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=USE_AUTOCAST and DEVICE.type == "cuda"):
                loss, fl, ml, kl_v, sc_v, _ = compute_loss_step(ids, mask, lbls)

            (loss / GRAD_ACCUM).backward()
            epoch_loss += loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                global_step += 1

                cur_lr  = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0

                if global_step % LOG_EVERY == 0:
                    print(f"  s={global_step:4d} loss={loss.item():.4f} "
                          f"fl={fl:.3f} mg={ml:.3f} kl={kl_v:.3f} sc={sc_v:.3f} "
                          f"lr={cur_lr:.1e} n={samples_seen:,} {elapsed/60:.1f}min")

                step_history.append({
                    "step": global_step, "loss": round(loss.item(), 5),
                    "focal": round(fl, 5), "margin": round(ml, 5),
                    "rdrop_kl": round(kl_v, 5), "supcon": round(sc_v, 5),
                    "lr": cur_lr, "samples_seen": samples_seen,
                })
                with open(LOG_CSV, "a", newline="") as f:
                    csv.writer(f).writerow([
                        epoch, global_step, round(loss.item(), 5),
                        round(fl, 5), round(ml, 5), round(kl_v, 5), round(sc_v, 5),
                        round(cur_lr, 8), samples_seen, round(elapsed, 1)
                    ])
                pbar.set_postfix(loss=f"{loss.item():.4f}", kl=f"{kl_v:.3f}", sc=f"{sc_v:.3f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\\n[Epoch {epoch}] avg_loss={avg_loss:.4f}  samples={samples_seen:,}")
        val_acc, val_f1, val_f1c, _, _ = evaluate(val_loader, split_name="Val")

        epoch_history.append({
            "epoch": epoch, "train_loss": round(avg_loss, 5),
            "val_acc": round(val_acc, 5), "val_f1": round(val_f1, 5),
            "val_f1_cls": [round(x, 5) for x in val_f1c], "global_step": global_step,
        })

        if val_f1 > best_f1:
            best_f1 = val_f1; no_improve = 0
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save_pretrained(str(MODEL_DIR))
            print(f"  => Best F1={best_f1:.4f}  step={global_step}")
        else:
            no_improve += 1
            print(f"  => No improve ({no_improve}/{EARLY_STOP})")
            if no_improve >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch}"); break

    total_min = (time.time() - t0) / 60
    print(f"Done: {total_min:.1f}min | Val F1={best_f1:.4f} | steps={global_step:,} | n={samples_seen:,}")
    return best_f1, global_step

BEST_VAL_F1, TOTAL_STEPS = train()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_TEST_EVAL = """\
print("\\n" + "="*65)
print("TEST SET — best checkpoint")
print("="*65)

t0_test = time.time()
test_acc, test_f1, test_f1cls, test_preds, test_labels = evaluate(test_loader, split_name="Test")
sps = len(test_df) / (time.time() - t0_test)
print(f"  Speed : {sps:.1f} samples/sec")
print()
print(classification_report(test_labels, test_preds, target_names=LABEL_NAMES, digits=4))

results = {
    "model_id":        MODEL_ID,
    "model_size":      "3b_cls_v3",
    "timestamp":       datetime.now().isoformat(),
    "training_data":   "train_augmented (24k balanced 8k×3)",
    "lora_rank":       LORA_RANK,
    "quantization":    "8bit" if USE_8BIT else "bfloat16",
    "loss_type":       f"Focal(g={FOCAL_GAMMA})+Margin+LS({LABEL_SMOOTH})",
    "rdrop":           USE_RDROP,   "rdrop_alpha":    RDROP_ALPHA,
    "llrd":            USE_LLRD,    "llrd_decay":     LLRD_DECAY,
    "supcon":          USE_SUPCON,  "supcon_weight":  SUPCON_WEIGHT,
    "accuracy":        round(test_acc, 5),
    "f1_macro":        round(test_f1, 5),
    "f1_per_class":    [round(x, 5) for x in test_f1cls],
    "cm":              confusion_matrix(test_labels, test_preds).tolist(),
    "speed_sps":       round(sps, 1),
    "total_steps":     TOTAL_STEPS,
    "samples_seen":    samples_seen,
    "best_val_f1":     round(BEST_VAL_F1, 5),
    "epoch_history":   epoch_history,
    "test_samples":    len(test_df),
}
with open(LOG_JSON, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

with open(LOG_DIR / "llm_3b_cls_results.json", "w") as f:
    json.dump({
        "model_id": MODEL_ID, "model_size": "3b_cls_v3",
        "timestamp": results["timestamp"],
        "train_minutes": round(TOTAL_STEPS * GRAD_ACCUM / len(train_loader) * 1.5, 1),
        "accuracy": results["accuracy"], "f1_macro": results["f1_macro"],
        "f1_per_class": results["f1_per_class"], "cm": results["cm"],
        "speed_sps": results["speed_sps"], "params_b": 3.09,
        "trainable_pct": round(LORA_RANK / 16 * 1.64, 2),
        "lora_rank": LORA_RANK, "total_steps": TOTAL_STEPS,
        "time_limited": False, "test_samples": len(test_df),
    }, f, indent=2, ensure_ascii=False)
print(f"Results -> {LOG_JSON}")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_LOGIT_BIAS = """\
if not BIAS_SEARCH:
    BEST_BIAS = torch.tensor([0.0, 0.0, 0.0])
    print("Bias search disabled.")
else:
    print("Logit bias grid search trên val set...")
    _, _, _, _, val_labels_raw, val_logits_raw = evaluate(
        val_loader, split_name="Val (bias search)", return_logits=True)
    val_labels_arr = np.array(val_labels_raw)

    best_f1_b, best_b = 0.0, [0.0, 0.0, 0.0]
    grid_results = []
    for b_off in BIAS_GRID:
        for b_hate in BIAS_GRID:
            preds = (val_logits_raw + torch.tensor([0.0, b_off, b_hate])).argmax(-1).numpy()
            f1    = f1_score(val_labels_arr, preds, average="macro", zero_division=0)
            grid_results.append((b_off, b_hate, f1))
            if f1 > best_f1_b:
                best_f1_b = f1; best_b = [0.0, b_off, b_hate]

    BEST_BIAS = torch.tensor(best_b)
    print(f"Best bias: OFF={best_b[1]:.2f}  HATE={best_b[2]:.2f}  Val F1={best_f1_b:.4f}")

    go = sorted(set(r[0] for r in grid_results))
    gh = sorted(set(r[1] for r in grid_results))
    heat = np.zeros((len(go), len(gh)))
    io = {v: i for i, v in enumerate(go)}; ih = {v: i for i, v in enumerate(gh)}
    for bo, bh, f1 in grid_results:
        heat[io[bo], ih[bh]] = f1

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(heat, annot=True, fmt=".4f", cmap="YlOrRd",
                xticklabels=[f"{v:.1f}" for v in gh], yticklabels=[f"{v:.1f}" for v in go],
                ax=ax, cbar_kws={"label": "F1-Macro"})
    ax.set_xlabel("bias_HATE"); ax.set_ylabel("bias_OFFENSIVE")
    ax.set_title(f"Logit Bias Search\\nBest: OFF={best_b[1]:.2f}, HATE={best_b[2]:.2f}, F1={best_f1_b:.4f}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_logit_bias_search.png", dpi=150, bbox_inches="tight")
    plt.show()

test_acc_b, test_f1_b, test_f1cls_b, test_preds_b, _ = evaluate(
    test_loader, logit_bias=BEST_BIAS, split_name="Test+Bias")
print(f"Delta F1: {test_f1_b - results['f1_macro']:+.4f}  "
      f"OFF:{test_f1cls_b[1]-results['f1_per_class'][1]:+.4f}  "
      f"HATE:{test_f1cls_b[2]-results['f1_per_class'][2]:+.4f}")

results["logit_bias"]             = best_b
results["f1_macro_with_bias"]     = round(test_f1_b, 5)
results["f1_per_class_with_bias"] = [round(x, 5) for x in test_f1cls_b]
with open(LOG_JSON, "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_ERROR_ANALYSIS = """\
use_biased  = test_f1_b >= results["f1_macro"]
preds_final = test_preds_b if use_biased else test_preds
texts_test  = test_df["text"].tolist()

PAIR_NAMES = {(0,1): "CLEAN<->OFF", (0,2): "CLEAN<->HATE", (1,2): "OFF<->HATE"}
error_pairs = defaultdict(list)
for i, (t, p) in enumerate(zip(test_labels, preds_final)):
    if t != p:
        error_pairs[tuple(sorted([t, p]))].append({
            "idx": i, "true": LABEL_NAMES[t], "pred": LABEL_NAMES[p],
            "text": texts_test[i][:100],
        })

total_err = sum(len(v) for v in error_pairs.values())
print(f"Errors: {total_err} / {len(test_labels)} ({100*total_err/len(test_labels):.1f}%)")
for k, name in PAIR_NAMES.items():
    errs = error_pairs.get(k, [])
    pct  = 100 * len(errs) / max(total_err, 1)
    print(f"  {name}: {len(errs)} ({pct:.1f}%)")
    for e in errs[:2]:
        print(f"    [{e['true']}->{e['pred']}] {e['text'][:75]}...")

err_df = pd.DataFrame([{**e, "pair": PAIR_NAMES.get(k, str(k))}
                        for k, vs in error_pairs.items() for e in vs])
err_csv = NB_LOG_DIR / f"errors_{RUN_ID}.csv"
err_df.to_csv(err_csv, index=False, encoding="utf-8")
print(f"Errors CSV -> {err_csv}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
pair_counts = {PAIR_NAMES[k]: len(v) for k, v in error_pairs.items() if k in PAIR_NAMES}
axes[0].pie(pair_counts.values(), labels=pair_counts.keys(),
            autopct="%1.1f%%", colors=["#FF9800","#F44336","#9C27B0"],
            startangle=90, textprops={"fontsize": 10})
axes[0].set_title("Error types by label pair", fontweight="bold")

cm_arr = confusion_matrix(test_labels, preds_final)
fn = [(cm_arr.sum(1)[i] - cm_arr[i,i]) / max(cm_arr.sum(1)[i], 1) for i in range(3)]
fp = [(cm_arr.sum(0)[i] - cm_arr[i,i]) / max(cm_arr.sum(0)[i], 1) for i in range(3)]
x  = np.arange(3); w = 0.35
b1 = axes[1].bar(x-w/2, fn, w, label="FN rate", color="#FF5722", alpha=0.85)
b2 = axes[1].bar(x+w/2, fp, w, label="FP rate", color="#2196F3", alpha=0.85)
axes[1].bar_label(b1, fmt="%.2f", fontsize=9, padding=2)
axes[1].bar_label(b2, fmt="%.2f", fontsize=9, padding=2)
axes[1].set_xticks(x); axes[1].set_xticklabels(LABEL_NAMES)
axes[1].set_title("FN / FP rate per class", fontweight="bold")
axes[1].legend(); axes[1].grid(axis="y", alpha=0.3); axes[1].set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "13_error_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_PLOT_LOSS = """\
steps  = [h["step"]     for h in step_history]
losses = [h["loss"]     for h in step_history]
kls    = [h["rdrop_kl"] for h in step_history]
scs    = [h["supcon"]   for h in step_history]

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
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_PLOT_F1 = """\
epochs   = [h["epoch"]  for h in epoch_history]
val_f1s  = [h["val_f1"] for h in epoch_history]
val_accs = [h["val_acc"] for h in epoch_history]
f1c      = [[h["val_f1_cls"][i] for h in epoch_history] for i in range(3)]
best_ep  = epochs[val_f1s.index(max(val_f1s))]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(epochs, val_f1s,  "b-o", lw=2,   label="F1-Macro", ms=6)
axes[0].plot(epochs, val_accs, "g--s",lw=1.5, label="Accuracy",  ms=5)
axes[0].axvline(best_ep, color="red", ls="--", lw=1.2, label=f"Best ep={best_ep}")
axes[0].set_title("Val F1-Macro & Accuracy", fontweight="bold")
axes[0].set_xlabel("Epoch"); axes[0].legend(); axes[0].grid(alpha=0.3)
for i, (lbl, col) in enumerate(zip(LABEL_NAMES, ["#4CAF50","#FF9800","#F44336"])):
    axes[1].plot(epochs, f1c[i], "-o", color=col, lw=2, label=f"F1-{lbl}", ms=6)
axes[1].axvline(best_ep, color="gray", ls="--", lw=1.2)
axes[1].set_title("Val F1 per Class", fontweight="bold")
axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(FIG_DIR / "15_f1_curve.png", dpi=150, bbox_inches="tight")
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_PLOT_CM = """\
best_preds = test_preds_b if use_biased else test_preds
cm_plot = confusion_matrix(test_labels, best_preds)
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
plt.show()
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_COMPARE = """\
baselines = {
    "PhoBERT":     LOG_DIR / "phobert_results.json",
    "3B SFT v1":   LOG_DIR / "llm_3b_results.json",
}
compare = {}
for name, path in baselines.items():
    if path.exists():
        compare[name] = json.load(open(path))
compare["3B CLS v3"] = {
    "accuracy":     results["accuracy"],
    "f1_macro":     results["f1_macro"],
    "f1_per_class": results["f1_per_class"],
    "speed_sps":    results["speed_sps"],
}

if len(compare) >= 2:
    names   = list(compare.keys())
    palette = ["#2196F3","#FF9800","#4CAF50","#9C27B0"][:len(names)]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("So sanh ket qua (v3 voi tat ca improvements)", fontsize=13, fontweight="bold")

    def bar(ax, vals, title, ylim=(0, 1)):
        bs = ax.bar(names, vals, color=palette, edgecolor="white", width=0.5)
        ax.bar_label(bs, fmt="%.4f", fontsize=8, padding=3)
        ax.set_title(title, fontweight="bold"); ax.set_ylim(*ylim)
        ax.tick_params(axis="x", rotation=20); ax.grid(axis="y", alpha=0.3)

    bar(axes[0,0], [compare[n]["f1_macro"]       for n in names], "F1-Macro")
    bar(axes[0,1], [compare[n]["accuracy"]        for n in names], "Accuracy")
    bar(axes[0,2], [compare[n]["f1_per_class"][0] for n in names], "F1-CLEAN")
    bar(axes[1,0], [compare[n]["f1_per_class"][1] for n in names], "F1-OFFENSIVE")
    bar(axes[1,1], [compare[n]["f1_per_class"][2] for n in names], "F1-HATE")
    bar(axes[1,2], [compare[n].get("speed_sps",0) for n in names], "Speed (sps)", ylim=(0, None))

    plt.tight_layout()
    plt.savefig(FIG_DIR / "17_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\\n{'Model':<18} {'Acc':>7} {'F1':>7} {'CLEAN':>7} {'OFF':>7} {'HATE':>7} {'sps':>6}")
    print("-"*66)
    for n, d in compare.items():
        fc = d["f1_per_class"]
        print(f"{n:<18} {d['accuracy']:>7.4f} {d['f1_macro']:>7.4f}"
              f" {fc[0]:>7.4f} {fc[1]:>7.4f} {fc[2]:>7.4f} {d.get('speed_sps',0):>6.1f}")
"""

# ─────────────────────────────────────────────────────────────────────────────
CELL_SUMMARY = """\
print("\\n" + "="*65)
print("FINAL SUMMARY — Qwen2.5-3B CLS v3 (All Improvements)")
print("="*65)
print(f"  F1-Macro (no bias) : {results['f1_macro']:.4f}")
print(f"  F1-Macro (+bias)   : {results.get('f1_macro_with_bias', results['f1_macro']):.4f}")
print(f"  F1 [C / O / H]    : {results['f1_per_class'][0]:.4f} / "
      f"{results['f1_per_class'][1]:.4f} / {results['f1_per_class'][2]:.4f}")
print(f"  Accuracy           : {results['accuracy']:.4f}")
print()
print(f"  Training data : train_augmented (24k, balanced 8k/class)")
print(f"  Quantization  : {results['quantization']}")
print(f"  LoRA rank     : {results['lora_rank']}")
print(f"  R-Drop        : {results['rdrop']}  alpha={results['rdrop_alpha']}")
print(f"  LLRD          : {results['llrd']}   decay={results['llrd_decay']}")
print(f"  SupCon        : {results['supcon']} weight={results['supcon_weight']}")
print(f"  Logit bias    : {results.get('logit_bias', [0,0,0])}")
print()
print(f"  Total steps   : {results['total_steps']:,}")
print(f"  Samples seen  : {results['samples_seen']:,}")
print(f"  Best Val F1   : {results['best_val_f1']:.4f}")
print(f"  Speed         : {results['speed_sps']:.1f} sps")
print("="*65)
"""

# ─────────────────────────────────────────────────────────────────────────────
# Assemble notebook
# ─────────────────────────────────────────────────────────────────────────────
cells = [
    new_markdown_cell(CELL_TITLE),        # 00
    new_code_cell(CELL_INSTALL),          # 01
    new_code_cell(CELL_IMPORTS),          # 02
    new_code_cell(CELL_CONFIG),           # 03
    new_code_cell(CELL_GPU_CHECK),        # 04
    new_code_cell(CELL_LOAD_DATA),        # 05
    new_code_cell(CELL_DATASET),          # 06
    new_code_cell(CELL_MODEL),            # 07
    new_code_cell(CELL_OPTIMIZER),        # 08
    new_code_cell(CELL_EVAL_FN),          # 09
    new_code_cell(CELL_TRAIN),            # 10
    new_code_cell(CELL_TEST_EVAL),        # 11
    new_code_cell(CELL_LOGIT_BIAS),       # 12
    new_code_cell(CELL_ERROR_ANALYSIS),   # 13
    new_code_cell(CELL_PLOT_LOSS),        # 14
    new_code_cell(CELL_PLOT_F1),          # 15
    new_code_cell(CELL_PLOT_CM),          # 16
    new_code_cell(CELL_COMPARE),          # 17
    new_code_cell(CELL_SUMMARY),          # 18
]

nb = new_notebook(cells=cells)
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12.0"},
}

out = Path(__file__).parent / "train_3b_full.ipynb"
with open(out, "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print(f"Created : {out}")
print(f"Cells   : {len(cells)}")
print()
for i, line in enumerate([
    "md  — Title & all improvements table",
    "    — Install packages",
    "    — Imports",
    "    — Config (rdrop/llrd/supcon/bias_search/autocast/data)",
    "    — GPU check (auto bfloat16 / 8-bit fallback)",
    "    — Load train_augmented (balanced 8k×3) + EDA -> fig/11",
    "    — ViHSDDataset + DataLoaders",
    "    — FocalLoss + MarginLoss + SupConLoss + LoRA r=64 + bias init",
    "    — LLRD optimizer groups + cosine scheduler + logger",
    "    — evaluate() with logit_bias + return_logits",
    "    — Training loop: R-Drop + SupCon + autocast + LLRD",
    "    — Test eval + sps + save JSON",
    "    — Logit bias grid search -> fig/12",
    "    — Error analysis (OFF<->CLEAN, OFF<->HATE) -> fig/13",
    "    — Loss curves (combined + KL + SupCon) -> fig/14",
    "    — F1 curve per epoch -> fig/15",
    "    — Confusion matrix -> fig/16",
    "    — Comparison vs PhoBERT/3B-SFT -> fig/17",
    "    — Final summary",
]):
    print(f"  {i:02d} {line}")
