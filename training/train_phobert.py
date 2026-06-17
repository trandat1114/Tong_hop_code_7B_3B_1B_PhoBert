"""
Fine-tuning PhoBERT (vinai/phobert-base-v2) cho phát hiện bình luận độc hại tiếng Việt.
Giới hạn: 85 phút training (time-based early stopping).
Full fine-tune (100% params) với weighted CrossEntropyLoss.

Dataset: ViHSD | Labels: CLEAN=0, OFFENSIVE=1, HATE=2
"""

import sys, json, time, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Line-buffered stdout để log realtime khi redirect vào file
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Cleaned"
SAVE_DIR   = BASE_DIR / "models" / "phobert"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)
SAVE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME     = "vinai/phobert-base-v2"
MAX_LEN        = 256
BATCH_SIZE     = 64
EPOCHS         = 10           # sẽ dừng sớm qua early-stop hoặc time limit
LR             = 2e-5
WEIGHT_DECAY   = 0.01
DROPOUT        = 0.3
PATIENCE       = 3
MAX_TRAIN_SECS = 85 * 60      # 85 phút

ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}
PARAMS_B = 0.135


def _norm(df):
    """Chuẩn hoá tên cột: rename label_id→label, clean_text/free_text→comment."""
    if "label_id" in df.columns:
        df = df.rename(columns={"label_id": "label"})
    if "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "comment"})
    elif "free_text" in df.columns:
        df = df.rename(columns={"free_text": "comment"})
    return df[["comment", "label"]].dropna()


def load_data(sample=None):
    train_df = _norm(pd.read_csv(DATA_DIR / "train_cleaned.csv"))
    val_df   = _norm(pd.read_csv(DATA_DIR / "dev_cleaned.csv"))
    test_df  = _norm(pd.read_csv(DATA_DIR / "test_cleaned.csv"))
    if sample:
        train_df = train_df.sample(n=min(sample, len(train_df)), random_state=42)
        val_df   = val_df.sample(n=min(sample // 5, len(val_df)), random_state=42)
        test_df  = test_df.sample(n=min(sample // 4, len(test_df)), random_state=42)
    print(f"[Data] Train:{len(train_df)} Val:{len(val_df)} Test:{len(test_df)}")
    return train_df, val_df, test_df


def build_model(dropout=0.3):
    import torch.nn as nn
    from transformers import AutoModel

    class PhoBERTClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(MODEL_NAME)
            self.dropout    = nn.Dropout(dropout)
            self.classifier = nn.Linear(self.phobert.config.hidden_size, 3)

        def forward(self, input_ids, attention_mask):
            out    = self.phobert(input_ids=input_ids, attention_mask=attention_mask)
            pooled = out.last_hidden_state[:, 0, :]   # [CLS]
            return self.classifier(self.dropout(pooled))

    return PhoBERTClassifier()


def build_dataset(df, tokenizer, max_len):
    import torch
    from torch.utils.data import Dataset

    class ToxicDS(Dataset):
        def __init__(self, df, tok, mlen):
            self.texts  = df["comment"].tolist()
            self.labels = df["label"].tolist()
            self.tok, self.mlen = tok, mlen

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            enc = self.tok(
                self.texts[i], max_length=self.mlen,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            return {
                "input_ids"     : enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label"         : torch.tensor(self.labels[i], dtype=torch.long),
            }

    return ToxicDS(df, tokenizer, max_len)


def train(args):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    train_df, val_df, test_df = load_data(args.sample)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = build_model(DROPOUT).to(device)
    total_p   = sum(p.numel() for p in model.parameters())
    print(f"[Model] {MODEL_NAME} — {total_p/1e6:.1f}M tham số (full fine-tune)")

    # Weighted loss cho class imbalance
    counts  = train_df["label"].value_counts().sort_index().values
    weights = torch.tensor(len(train_df) / (3 * counts), dtype=torch.float).to(device)
    print(f"[Loss] class weights: {[round(w, 2) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_ds     = build_dataset(train_df, tokenizer, MAX_LEN)
    val_ds       = build_dataset(val_df,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,     shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=4, pin_memory=True)

    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps  = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * 0.1)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_f1, patience_cnt = 0.0, 0
    best_path = SAVE_DIR / "best_weights.pth"
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Time limit check trước khi bắt đầu epoch
        if time.time() - t0 > MAX_TRAIN_SECS:
            print(f"[TimeLimitReached] Dừng trước epoch {epoch} ({(time.time()-t0)/60:.1f}min)")
            break

        model.train()
        ep_loss, ep_preds, ep_golds = 0.0, [], []

        for step, batch in enumerate(train_loader, 1):
            if time.time() - t0 > MAX_TRAIN_SECS:
                print(f"\n  [STOP] Giới hạn {MAX_TRAIN_SECS//60}min đạt được trong epoch {epoch}")
                break
            ids, mask, labels = (
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["label"].to(device),
            )
            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            ep_loss += loss.item()
            ep_preds.extend(logits.argmax(-1).cpu().numpy())
            ep_golds.extend(labels.cpu().numpy())

            if step % 30 == 0:
                avg_loss = ep_loss / step
                f1_now   = f1_score(ep_golds, ep_preds, average="macro")
                elapsed  = (time.time() - t0) / 60
                print(f"  Ep{epoch} [{step}/{len(train_loader)}] loss={avg_loss:.4f} f1={f1_now:.4f} {elapsed:.1f}min", end="\r")

        # Validation
        model.eval()
        vp, vg = [], []
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
                vp.extend(out.argmax(-1).cpu().numpy())
                vg.extend(batch["label"].numpy())

        val_f1  = f1_score(vg, vp, average="macro")
        val_acc = accuracy_score(vg, vp)
        elapsed = (time.time() - t0) / 60
        print(f"\nEpoch {epoch}/{EPOCHS} | Val F1={val_f1:.4f} Acc={val_acc:.4f} | {elapsed:.1f}min")

        if val_f1 > best_f1:
            best_f1, patience_cnt = val_f1, 0
            torch.save(model.state_dict(), best_path)
            print(f"  [Save] Best F1={best_f1:.4f} → {best_path}")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"[EarlyStop] {PATIENCE} epochs không cải thiện")
                break

    train_secs = time.time() - t0
    print(f"\n[Train] Tổng: {train_secs/60:.1f} phút, best Val F1={best_f1:.4f}")

    # ─── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[Eval] Load best model và đánh giá test set...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()

    test_ds     = build_dataset(test_df, tokenizer, MAX_LEN)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2, num_workers=4, pin_memory=True)

    tp, tg = [], []
    t_inf  = time.time()
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            tp.extend(out.argmax(-1).cpu().numpy())
            tg.extend(batch["label"].numpy())
    speed = len(tg) / (time.time() - t_inf)

    acc = accuracy_score(tg, tp)
    f1m = f1_score(tg, tp, average="macro")
    f1p = f1_score(tg, tp, average=None).tolist()
    cm  = confusion_matrix(tg, tp).tolist()

    print(f"[Results] Accuracy={acc:.4f} F1-Macro={f1m:.4f} Speed={speed:.1f}sps")
    print(classification_report(tg, tp, target_names=["CLEAN", "OFFENSIVE", "HATE"]))

    # Lưu weights cho inference
    torch.save(model.state_dict(), SAVE_DIR / "custom_phobert_weights.pth")

    result = {
        "model_id"     : MODEL_NAME,
        "model_size"   : "phobert",
        "timestamp"    : datetime.now().isoformat(),
        "train_minutes": round(train_secs / 60, 1),
        "accuracy"     : round(acc, 4),
        "f1_macro"     : round(f1m, 4),
        "f1_per_class" : [round(x, 4) for x in f1p],
        "cm"           : cm,
        "speed_sps"    : round(speed, 1),
        "params_b"     : PARAMS_B,
        "trainable_pct": 100.0,
        "test_samples" : len(tg),
    }

    with open(LOG_DIR / "phobert_results.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "comment"   : test_df["comment"].tolist(),
        "label_true": tg, "label_pred": tp,
        "label_name": [ID2LABEL[p] for p in tp],
    }).to_csv(LOG_DIR / "phobert_predictions.csv", index=False)

    print(f"[Save] {LOG_DIR}/phobert_results.json")
    print(f"[Save] {SAVE_DIR}/custom_phobert_weights.pth")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune PhoBERT cho phát hiện độc hại")
    parser.add_argument("--sample", type=int, default=None,
                        help="Số mẫu để debug nhanh")
    args = parser.parse_args()
    train(args)
