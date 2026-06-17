"""
Unified LoRA fine-tuning cho Qwen2.5 models: 1.5B, 3B, 7B
Dataset: ViHSD — Vietnamese Hate Speech Detection
Labels: CLEAN=0, OFFENSIVE=1, HATE=2
Giới hạn: 85 phút/model (hard stop qua callback)

Sử dụng:
  python train_llm_lora.py --model-size 1.5b
  python train_llm_lora.py --model-size 3b
  python train_llm_lora.py --model-size 7b
  python train_llm_lora.py --model-size 1.5b --sample 500   # debug
"""

import os, sys, json, time, argparse
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Line-buffered stdout để log realtime khi redirect vào file
sys.stdout.reconfigure(line_buffering=True)

# Import unsloth BEFORE trl/transformers so Unsloth can patch correctly
try:
    import unsloth  # noqa: F401
except ImportError:
    pass

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "Cleaned"
LOG_DIR   = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ─── Config per model size ────────────────────────────────────────────────────
MODEL_CONFIGS = {
    "1.5b": {
        "model_id"      : "Qwen/Qwen2.5-1.5B-Instruct",
        "batch_size"    : 8,
        "grad_accum"    : 1,
        "lora_rank"     : 32,
        "lora_alpha"    : 64,
        "default_epochs": 3,
        "lr"            : 3e-4,
        "save_subdir"   : "llm_1_5b/lora_adapter",
        "infer_batch"   : 64,
        "params_b"      : 1.54,
    },
    "3b": {
        "model_id"      : "Qwen/Qwen2.5-3B-Instruct",
        "batch_size"    : 4,
        "grad_accum"    : 2,
        "lora_rank"     : 16,
        "lora_alpha"    : 32,
        "default_epochs": 3,
        "lr"            : 2e-4,
        "save_subdir"   : "llm_3b_lora",
        "infer_batch"   : 32,
        "params_b"      : 3.09,
    },
    "7b": {
        "model_id"      : "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "batch_size"    : 2,
        "grad_accum"    : 4,
        "lora_rank"     : 16,
        "lora_alpha"    : 32,
        "default_epochs": 2,
        "lr"            : 2e-4,
        "save_subdir"   : "llm_7b_lora",
        "infer_batch"   : 16,
        "params_b"      : 7.62,
    },
}

MAX_TRAIN_SECS = 85 * 60  # 85 phút hard stop

LABEL2ID = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
ID2LABEL  = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân loại nội dung mạng xã hội tiếng Việt.\n"
    "Phân loại bình luận vào MỘT trong 3 nhãn:\n"
    "- CLEAN: Bình luận bình thường, không tiêu cực\n"
    "- OFFENSIVE: Xúc phạm, thô tục, thiếu tôn trọng cá nhân\n"
    "- HATE: Thù ghét nhóm người, kêu gọi bạo lực, kỳ thị\n\n"
    "Ví dụ:\n"
    "Bình luận: \"Hôm nay thời tiết đẹp!\" → CLEAN\n"
    "Bình luận: \"Mày ngu vl đi chỗ khác\" → OFFENSIVE\n"
    "Bình luận: \"Cái bọn đó phải bị tiêu diệt\" → HATE\n\n"
    "Chỉ trả lời đúng MỘT TỪ: CLEAN, OFFENSIVE, hoặc HATE."
)


def make_prompt(comment: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nBình luận: {comment}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def make_example(comment: str, label: int) -> str:
    return make_prompt(comment) + ID2LABEL[label] + "<|im_end|>"


def _norm(df):
    """Chuẩn hoá tên cột: rename label_id→label, clean_text/free_text→comment."""
    if "label_id" in df.columns:
        df = df.rename(columns={"label_id": "label"})
    if "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "comment"})
    elif "free_text" in df.columns:
        df = df.rename(columns={"free_text": "comment"})
    return df[["comment", "label"]].dropna()


def load_data(sample: int = None):
    train_df = _norm(pd.read_csv(DATA_DIR / "train_cleaned.csv"))
    val_df   = _norm(pd.read_csv(DATA_DIR / "dev_cleaned.csv"))
    test_df  = _norm(pd.read_csv(DATA_DIR / "test_cleaned.csv"))

    # Cân bằng class imbalance bằng oversampling
    max_n  = train_df["label"].value_counts().max()
    frames = [train_df[train_df["label"] == i].sample(n=max_n, replace=True, random_state=42)
              for i in range(3)]
    train_balanced = pd.concat(frames).sample(frac=1, random_state=42).reset_index(drop=True)

    if sample:
        train_balanced = train_balanced.sample(n=min(sample, len(train_balanced)), random_state=42)
        val_df  = val_df.sample(n=min(sample // 5, len(val_df)), random_state=42)
        test_df = test_df.sample(n=min(sample // 4, len(test_df)), random_state=42)

    print(f"[Data] Train(balanced):{len(train_balanced)} Val:{len(val_df)} Test:{len(test_df)}")
    return train_balanced, val_df, test_df


def train_and_eval(args):
    cfg      = MODEL_CONFIGS[args.model_size]
    model_id = cfg["model_id"]
    save_dir = BASE_DIR / "models" / cfg["save_subdir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs     = args.epochs or cfg["default_epochs"]
    batch_size = cfg["batch_size"]
    grad_accum = cfg["grad_accum"]
    lr         = cfg["lr"]
    rank       = cfg["lora_rank"]
    alpha      = cfg["lora_alpha"]

    print(f"\n{'='*65}")
    print(f"[Model] {model_id}")
    print(f"[Config] epochs={epochs} batch={batch_size} grad_accum={grad_accum}")
    print(f"[Config] lr={lr} rank={rank} alpha={alpha} max={MAX_TRAIN_SECS//60}min")
    print(f"[Save]   {save_dir}")
    print(f"{'='*65}\n")

    train_df, val_df, test_df = load_data(args.sample)

    import torch
    from trl import SFTTrainer, SFTConfig
    from transformers import TrainerCallback
    from datasets import Dataset

    USE_UNSLOTH = False
    try:
        from unsloth import FastLanguageModel  # already imported at top; this just gets the name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_id, max_seq_length=512, dtype=None, load_in_4bit=True,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=rank, lora_alpha=alpha, lora_dropout=0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            bias="none", use_gradient_checkpointing="unsloth", random_state=42,
        )
        USE_UNSLOTH = True
        # Unsloth 2026.x sets eos_token="<EOS_TOKEN>" placeholder; TRL 0.24 validates it against
        # vocab and rejects. Reset to the actual Qwen2.5-Instruct EOS token.
        tokenizer.eos_token = "<|im_end|>"
        print("[Model] Unsloth OK — tốc độ 2×")
    except ImportError:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True,
        )
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
            lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
        print("[Model] transformers+peft (không có Unsloth)")

    total    = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Params] {total/1e9:.2f}B tổng | {trainable/1e6:.1f}M trainable ({100*trainable/total:.2f}%)")

    tokenizer.padding_side = "right"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    # Build text datasets; SFTConfig handles tokenization internally via dataset_text_field
    train_ds = Dataset.from_dict({
        "text": [make_example(c, l) for c, l in zip(train_df["comment"].tolist(), train_df["label"].tolist())]
    })
    val_ds = Dataset.from_dict({
        "text": [make_example(c, l) for c, l in zip(val_df["comment"].tolist(), val_df["label"].tolist())]
    })

    class _TimeLimit(TrainerCallback):
        def __init__(self):
            self.start = time.time()
        def on_step_end(self, args, state, control, **kwargs):
            if time.time() - self.start > MAX_TRAIN_SECS:
                control.should_training_stop = True
                elapsed = (time.time() - self.start) / 60
                print(f"\n  [STOP] Giới hạn {MAX_TRAIN_SECS//60}min đạt được ({elapsed:.1f}min thực tế)")
            return control

    ta = SFTConfig(
        output_dir=str(save_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=batch_size * 2,
        warmup_ratio=0.05,
        learning_rate=lr,
        bf16=True,
        logging_steps=20,
        save_steps=200,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_total_limit=1,
        load_best_model_at_end=False,
        report_to="none",
        optim="adamw_8bit" if USE_UNSLOTH else "adamw_torch",
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        seed=42,
        dataloader_num_workers=2,
        max_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=ta,
        callbacks=[_TimeLimit()],
    )

    print(f"\n[Train] Bắt đầu — giới hạn {MAX_TRAIN_SECS//60} phút...")
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    print(f"[Train] Hoàn thành sau {train_secs/60:.1f} phút")

    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
    print(f"[Save] LoRA adapter → {save_dir}")

    # ─── Evaluate ─────────────────────────────────────────────────────────────
    if USE_UNSLOTH:
        FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"

    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

    comments = test_df["comment"].tolist()
    golds    = test_df["label"].tolist()
    preds    = []
    device   = next(model.parameters()).device
    IB       = cfg["infer_batch"]

    t_inf = time.time()
    for i in range(0, len(comments), IB):
        chunk   = comments[i:i+IB]
        prompts = [make_prompt(c) for c in chunk]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        for tok_ids in out:
            txt = tokenizer.decode(
                tok_ids[enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().upper()
            preds.append(2 if "HATE" in txt else (1 if "OFFENSIVE" in txt else 0))
        if (i // IB) % 5 == 0:
            print(f"  Eval [{i+len(chunk)}/{len(comments)}]", end="\r")

    inf_secs = time.time() - t_inf
    speed    = len(comments) / inf_secs

    acc = accuracy_score(golds, preds)
    f1m = f1_score(golds, preds, average="macro")
    f1p = f1_score(golds, preds, labels=[0, 1, 2], average=None, zero_division=0).tolist()
    cm  = confusion_matrix(golds, preds).tolist()

    print(f"\n[Eval] Accuracy={acc:.4f} F1-Macro={f1m:.4f} Speed={speed:.1f}sps")
    print(classification_report(golds, preds, labels=[0, 1, 2], target_names=["CLEAN", "OFFENSIVE", "HATE"], zero_division=0))

    result = {
        "model_id"      : model_id,
        "model_size"    : args.model_size,
        "timestamp"     : datetime.now().isoformat(),
        "train_minutes" : round(train_secs / 60, 1),
        "accuracy"      : round(acc, 4),
        "f1_macro"      : round(f1m, 4),
        "f1_per_class"  : [round(x, 4) for x in f1p],
        "cm"            : cm,
        "speed_sps"     : round(speed, 1),
        "params_b"      : cfg["params_b"],
        "trainable_pct" : round(100 * trainable / total, 2),
        "lora_rank"     : rank,
        "epochs_config" : epochs,
        "time_limited"  : train_secs > MAX_TRAIN_SECS - 60,
        "test_samples"  : len(golds),
    }

    # Tên file dùng dấu chấm để khớp với compare_models.py
    size_key = args.model_size  # "1.5b", "3b", "7b"
    rpath = LOG_DIR / f"llm_{size_key}_results.json"
    with open(rpath, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "comment"   : test_df["comment"].tolist(),
        "label_true": golds,
        "label_pred": preds,
        "label_name": [ID2LABEL[p] for p in preds],
    }).to_csv(LOG_DIR / f"llm_{size_key}_predictions.csv", index=False)

    print(f"[Save] {rpath}")

    del model
    import gc, torch as _t
    gc.collect()
    _t.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified LoRA fine-tuning cho Qwen2.5 models")
    parser.add_argument("--model-size", choices=["1.5b", "3b", "7b"], required=True,
                        help="Kích thước model: 1.5b | 3b | 7b")
    parser.add_argument("--sample", type=int, default=None,
                        help="Số mẫu để debug nhanh (vd: --sample 500)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override số epochs (mặc định theo config)")
    args = parser.parse_args()
    train_and_eval(args)
