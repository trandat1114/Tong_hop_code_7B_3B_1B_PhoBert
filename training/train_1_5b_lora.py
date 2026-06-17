"""
Fine-tuning Qwen2.5-1.5B-Instruct với LoRA cho bài toán phát hiện bình luận độc hại tiếng Việt.

Dataset: ViHSD (Vietnamese Hate Speech Detection)
Labels: CLEAN (0), OFFENSIVE (1), HATE (2)

LƯU Ý: Model 1.5B zero-shot có F1-Macro=0.1524 do bias RLHF.
Script này fine-tune để cải thiện kết quả trên tập tiếng Việt.

Yêu cầu: GPU ≥8GB VRAM (RTX 3080 trở lên)
"""

import os
import sys
import json
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Cleaned"
OUTPUT_DIR = BASE_DIR / "models" / "llm_1_5b" / "lora_adapter"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─── Config ───────────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_SEQ_LEN  = 512
BATCH_SIZE   = 4        # 1.5B nhỏ hơn, batch lớn hơn được
GRAD_ACCUM   = 2        # effective batch = 8
EPOCHS       = 3        # nhiều epoch hơn vì model nhỏ học chậm hơn
LR           = 3e-4     # LR cao hơn một chút cho model nhỏ
LORA_RANK    = 32       # rank cao hơn để bù cho capacity thấp
LORA_ALPHA   = 64
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.05
SAVE_STEPS   = 100
LOG_STEPS    = 10

LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích nội dung mạng xã hội tiếng Việt.
Phân loại bình luận vào một trong 3 nhãn:
- CLEAN: Bình luận bình thường, không có yếu tố tiêu cực
- OFFENSIVE: Bình luận xúc phạm, thô tục, thiếu tôn trọng
- HATE: Bình luận thù ghét nhằm vào cá nhân/nhóm người, kêu gọi bạo lực

Ví dụ:
Bình luận: "Hôm nay trời đẹp quá!" → CLEAN
Bình luận: "Mày ngu vl đi chỗ khác chơi" → OFFENSIVE
Bình luận: "Cái bọn đó nên bị tiêu diệt hết đi" → HATE

Chỉ trả lời một từ: CLEAN, OFFENSIVE, hoặc HATE."""


def build_prompt(comment: str) -> str:
    return f"""<|im_start|>system
{SYSTEM_PROMPT}<|im_end|>
<|im_start|>user
Bình luận: {comment}<|im_end|>
<|im_start|>assistant
"""


def build_training_example(comment: str, label_id: int) -> str:
    label = ID2LABEL[label_id]
    return build_prompt(comment) + label + "<|im_end|>"


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    val_df   = pd.read_csv(DATA_DIR / "val_cleaned.csv")
    test_df  = pd.read_csv(DATA_DIR / "test_cleaned.csv")

    # Xử lý class imbalance — oversample HATE và OFFENSIVE
    clean_df     = train_df[train_df["label"] == 0]
    offensive_df = train_df[train_df["label"] == 1]
    hate_df      = train_df[train_df["label"] == 2]

    # Duplicate minority classes để balance
    target_n     = len(clean_df)
    offensive_os = offensive_df.sample(n=target_n, replace=True, random_state=42)
    hate_os      = hate_df.sample(n=target_n, replace=True, random_state=42)
    train_df_balanced = pd.concat([clean_df, offensive_os, hate_os]).sample(frac=1, random_state=42)

    print(f"[Data] Train (balanced): {len(train_df_balanced)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"[Data] Label dist (balanced train):\n{train_df_balanced['label'].value_counts()}")
    return train_df_balanced, val_df, test_df


def train(args):
    try:
        from unsloth import FastLanguageModel
        USE_UNSLOTH = True
        print("[Model] Dùng Unsloth để tối ưu tốc độ")
    except ImportError:
        USE_UNSLOTH = False
        print("[Model] Không có Unsloth, dùng transformers+peft")

    import torch
    from transformers import TrainingArguments, DataCollatorForSeq2Seq
    from trl import SFTTrainer
    from datasets import Dataset
    from sklearn.metrics import accuracy_score, f1_score, classification_report

    train_df, val_df, test_df = load_data()
    if args.sample:
        train_df = train_df.sample(n=min(args.sample, len(train_df)), random_state=42)
        val_df   = val_df.sample(n=min(args.sample // 5, len(val_df)), random_state=42)

    # ── Load model ──
    if USE_UNSLOTH:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=LORA_RANK,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=LORA_RANK, lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)

    total   = sum(p.numel() for p in model.parameters())
    train_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] {total/1e9:.2f}B tổng, {train_p/1e6:.2f}M trainable ({100*train_p/total:.2f}%)")

    # ── Tokenize ──
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        texts = [build_training_example(c, l)
                 for c, l in zip(examples["comment"], examples["label"])]
        return tokenizer(texts, truncation=True, max_length=MAX_SEQ_LEN, padding=False)

    train_ds = Dataset.from_pandas(train_df[["comment", "label"]].reset_index(drop=True))
    val_ds   = Dataset.from_pandas(val_df[["comment", "label"]].reset_index(drop=True))
    train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tok   = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=8,
        warmup_ratio=WARMUP_RATIO,
        learning_rate=LR,
        bf16=True,
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=True,
        report_to="none",
        optim="adamw_8bit" if USE_UNSLOTH else "adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LEN,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8,
                                             return_tensors="pt", padding=True),
        args=training_args,
    )

    print(f"\n[Train] Bắt đầu fine-tuning {MODEL_NAME} ({EPOCHS} epochs)...")
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"[Train] Hoàn thành sau {elapsed/60:.1f} phút")

    model.save_pretrained(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print(f"[Save] LoRA adapter → {OUTPUT_DIR}")

    # ── Evaluate ──
    print("\n[Eval] Đánh giá trên test set...")
    if USE_UNSLOTH:
        FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    if args.sample:
        test_df = test_df.sample(n=min(args.sample // 4, len(test_df)), random_state=42)

    preds, golds = [], []
    device = next(model.parameters()).device
    BATCH  = 32

    for i in range(0, len(test_df), BATCH):
        batch   = test_df.iloc[i:i+BATCH]
        prompts = [build_prompt(c) for c in batch["comment"]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                        max_length=MAX_SEQ_LEN).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        for tok_ids in out:
            decoded = tokenizer.decode(tok_ids[enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True).strip().upper()
            if "HATE" in decoded:
                preds.append(2)
            elif "OFFENSIVE" in decoded:
                preds.append(1)
            else:
                preds.append(0)
        golds.extend(batch["label"].tolist())

        if (i // BATCH) % 10 == 0:
            print(f"  [{i}/{len(test_df)}]", end="\r")

    acc = accuracy_score(golds, preds)
    f1  = f1_score(golds, preds, average="macro")
    print(f"\n[Results] Accuracy={acc:.4f}, F1-Macro={f1:.4f}")
    print(classification_report(golds, preds, target_names=["CLEAN", "OFFENSIVE", "HATE"]))

    results = {
        "model": "Qwen2.5-1.5B-Instruct-LoRA",
        "timestamp": datetime.now().isoformat(),
        "training_minutes": round(elapsed / 60, 1),
        "test_accuracy": round(acc, 4),
        "test_f1_macro": round(f1, 4),
        "lora_rank": LORA_RANK,
        "epochs": EPOCHS,
        "oversampled": True,
        "note": "Fine-tuned với balanced dataset (oversample minority classes)",
    }
    with open(LOG_DIR / "train_1_5b_results.json", "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Save] {LOG_DIR / 'train_1_5b_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-1.5B với LoRA")
    parser.add_argument("--sample", type=int, default=None,
                        help="Số mẫu để debug (vd: --sample 200)")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--rank", type=int, default=LORA_RANK)
    parser.add_argument("--no-oversample", action="store_true",
                        help="Tắt oversampling minority classes")
    args = parser.parse_args()

    EPOCHS    = args.epochs
    LR        = args.lr
    LORA_RANK = args.rank

    train(args)
