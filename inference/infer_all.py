"""
Unified inference cho 4 models fine-tuned: PhoBERT, Qwen2.5-1.5B, 3B, 7B.

Sử dụng:
  python infer_all.py                         # tất cả models, full test set
  python infer_all.py --model phobert
  python infer_all.py --model llm_1_5b
  python infer_all.py --model llm_3b
  python infer_all.py --model llm_7b
  python infer_all.py --text "bình luận cần phân loại"
  python infer_all.py --sample 500
"""

import sys, json, time, argparse
import pandas as pd
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "Cleaned"
LOG_DIR    = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

PHOBERT_MODEL = "vinai/phobert-base-v2"

MODEL_REGISTRY = {
    "phobert" : {
        "type"   : "phobert",
        "weights": BASE_DIR / "models" / "phobert" / "custom_phobert_weights.pth",
        "params_b": 0.135,
    },
    "llm_1_5b": {
        "type"   : "llm",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "adapter" : BASE_DIR / "models" / "llm_1_5b" / "lora_adapter",
        "batch"   : 64,
        "params_b": 1.54,
    },
    "llm_3b"  : {
        "type"   : "llm",
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "adapter" : BASE_DIR / "models" / "llm_3b_lora",
        "batch"   : 32,
        "params_b": 3.09,
    },
    "llm_7b"  : {
        "type"   : "llm",
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter" : BASE_DIR / "models" / "llm_7b_lora",
        "batch"   : 16,
        "params_b": 7.62,
    },
}

ID2LABEL = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


def _norm(df):
    """Chuẩn hoá tên cột CSV: label_id→label, clean_text/free_text→comment."""
    if "label_id" in df.columns:
        df = df.rename(columns={"label_id": "label"})
    if "clean_text" in df.columns:
        df = df.rename(columns={"clean_text": "comment"})
    elif "free_text" in df.columns:
        df = df.rename(columns={"free_text": "comment"})
    return df[["comment", "label"]].dropna()


SYSTEM_PROMPT = (
    "Bạn là chuyên gia phân loại nội dung mạng xã hội tiếng Việt.\n"
    "Phân loại bình luận vào MỘT trong 3 nhãn:\n"
    "- CLEAN: Bình luận bình thường, không tiêu cực\n"
    "- OFFENSIVE: Xúc phạm, thô tục, thiếu tôn trọng cá nhân\n"
    "- HATE: Thù ghét nhóm người, kêu gọi bạo lực, kỳ thị\n\n"
    "Chỉ trả lời đúng MỘT TỪ: CLEAN, OFFENSIVE, hoặc HATE."
)


def make_llm_prompt(comment: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\nBình luận: {comment}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ─── PhoBERT Inference ────────────────────────────────────────────────────────
def run_phobert(comments: list, weights_path) -> tuple:
    import torch, torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModel

    class Clf(nn.Module):
        def __init__(self):
            super().__init__()
            self.phobert    = AutoModel.from_pretrained(PHOBERT_MODEL)
            self.dropout    = nn.Dropout(0.3)
            self.classifier = nn.Linear(self.phobert.config.hidden_size, 3)
        def forward(self, ids, mask):
            out = self.phobert(input_ids=ids, attention_mask=mask)
            return self.classifier(self.dropout(out.last_hidden_state[:, 0, :]))

    class InferDS(Dataset):
        def __init__(self, texts, tok):
            self.texts, self.tok = texts, tok
        def __len__(self): return len(self.texts)
        def __getitem__(self, i):
            e = self.tok(self.texts[i], max_length=256, padding="max_length",
                         truncation=True, return_tensors="pt")
            return {"input_ids"     : e["input_ids"].squeeze(0),
                    "attention_mask": e["attention_mask"].squeeze(0)}

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(PHOBERT_MODEL, use_fast=False)
    model     = Clf().to(device)
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.eval()

    loader = DataLoader(InferDS(comments, tokenizer), batch_size=128,
                        num_workers=4, pin_memory=True)
    preds, t0 = [], time.time()
    with torch.no_grad():
        for batch in loader:
            out = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            preds.extend(out.argmax(-1).cpu().numpy().tolist())
    speed = len(comments) / (time.time() - t0)
    print(f"  [PhoBERT] {speed:.1f} sps")
    return preds, speed


# ─── LLM Inference ────────────────────────────────────────────────────────────
def run_llm(comments: list, model_id: str, adapter_dir, batch_size: int = 16) -> tuple:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    adapter_path = str(adapter_dir) if (adapter_dir and Path(adapter_dir).exists()) else None

    try:
        from unsloth import FastLanguageModel
        src = adapter_path if adapter_path else model_id
        model, tokenizer = FastLanguageModel.from_pretrained(
            src, max_seq_length=512, dtype=None, load_in_4bit=True)
        FastLanguageModel.for_inference(model)
        print(f"  [Unsloth] {model_id}")
    except ImportError:
        tokenizer = AutoTokenizer.from_pretrained(
            adapter_path if adapter_path else model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, quantization_config=bnb, device_map="auto", trust_remote_code=True)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"  [LoRA] adapter: {adapter_path}")

    tokenizer.padding_side = "left"
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    device  = next(model.parameters()).device
    prompts = [make_llm_prompt(c) for c in comments]
    preds, t0 = [], time.time()

    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i:i+batch_size]
        enc = tokenizer(chunk, return_tensors="pt", padding=True,
                        truncation=True, max_length=512).to(device)
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=5, do_sample=False,
                                 pad_token_id=tokenizer.eos_token_id)
        for tok_ids in out:
            txt = tokenizer.decode(
                tok_ids[enc["input_ids"].shape[1]:], skip_special_tokens=True
            ).strip().upper()
            preds.append(2 if "HATE" in txt else (1 if "OFFENSIVE" in txt else 0))

        if (i // batch_size) % 10 == 0:
            print(f"  [{i+len(chunk)}/{len(comments)}]", end="\r")

    speed = len(comments) / (time.time() - t0)
    print(f"\n  [{model_id.split('/')[-1]}] {speed:.1f} sps")

    del model
    import gc, torch as _t
    gc.collect()
    _t.cuda.empty_cache()
    return preds, speed


# ─── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(preds: list, golds: list, name: str) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
    acc = accuracy_score(golds, preds)
    f1m = f1_score(golds, preds, average="macro")
    f1p = f1_score(golds, preds, average=None).tolist()
    cm  = confusion_matrix(golds, preds).tolist()
    print(f"\n{'─'*55}")
    print(f"[{name}] Accuracy={acc:.4f} F1-Macro={f1m:.4f}")
    print(classification_report(golds, preds, target_names=["CLEAN", "OFFENSIVE", "HATE"]))
    return {"accuracy": round(acc, 4), "f1_macro": round(f1m, 4),
            "f1_per_class": [round(x, 4) for x in f1p], "cm": cm}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",
                        choices=["phobert", "llm_1_5b", "llm_3b", "llm_7b", "all"],
                        default="all")
    parser.add_argument("--text", type=str, default=None,
                        help="Predict một bình luận đơn lẻ")
    parser.add_argument("--sample", type=int, default=None)
    args = parser.parse_args()

    # ── Single-text mode ──
    if args.text:
        models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
        print(f"\n[Input] {args.text}\n")
        for m in models_to_run:
            cfg = MODEL_REGISTRY[m]
            if cfg["type"] == "phobert":
                pred, _ = run_phobert([args.text], cfg["weights"])
            else:
                pred, _ = run_llm([args.text], cfg["model_id"],
                                   cfg.get("adapter"), cfg.get("batch", 16))
            print(f"  [{m}] → {ID2LABEL[pred[0]]}")
        return

    # ── Batch mode ──
    test_df  = _norm(pd.read_csv(DATA_DIR / "test_cleaned.csv"))
    if args.sample:
        test_df = test_df.sample(n=min(args.sample, len(test_df)), random_state=42)
    comments = test_df["comment"].tolist()
    golds    = test_df["label"].tolist()
    print(f"\n[Data] {len(test_df)} samples")

    models_to_run = list(MODEL_REGISTRY.keys()) if args.model == "all" else [args.model]
    all_results   = {}

    for m in models_to_run:
        cfg = MODEL_REGISTRY[m]
        print(f"\n{'='*55}\n[Running] {m} ({cfg['params_b']}B)")

        if cfg["type"] == "phobert":
            preds, speed = run_phobert(comments, cfg["weights"])
        else:
            preds, speed = run_llm(comments, cfg["model_id"],
                                   cfg.get("adapter"), cfg.get("batch", 16))

        r = evaluate(preds, golds, m)
        r["speed_sps"] = round(speed, 1)
        r["params_b"]  = cfg["params_b"]
        all_results[m] = r

        pd.DataFrame({
            "comment"   : comments,
            "label_true": golds,
            "label_pred": preds,
            "label_name": [ID2LABEL[p] for p in preds],
        }).to_csv(LOG_DIR / f"{m}_predictions.csv", index=False)

    out_path = LOG_DIR / "inference_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")

    print("\n=== TÓM TẮT ===")
    print(f"{'Model':<15} {'Acc':>8} {'F1':>8} {'Speed(sps)':>11}")
    print("-" * 45)
    for m, r in all_results.items():
        print(f"{m:<15} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {r.get('speed_sps', 0):>11.1f}")


if __name__ == "__main__":
    main()
