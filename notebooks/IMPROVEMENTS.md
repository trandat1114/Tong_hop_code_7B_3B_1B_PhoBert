# Kế hoạch Cải thiện — So sánh PhoBERT vs LLM (ViHSD)

Tài liệu tổng hợp các hạng mục cần cải thiện dựa trên phân tích codebase (`training/`, `inference/`, `compare_models.py`) và kết quả thực nghiệm hiện tại.

**Baseline (test ~6.680 mẫu):**

| Model | F1-Macro | F1-OFF | F1-HATE | Speed |
|-------|----------|--------|---------|-------|
| PhoBERT | **0.5991** | **0.3641** | **0.5456** | 7.0 sps |
| LLM 3B | 0.5361 | 0.3038 | 0.4732 | 11.0 sps |
| LLM 1.5B | 0.5103 | 0.3053 | 0.3684 | 3.7 sps |
| LLM 7B | 0.4244 | 0.2623 | 0.2005 | 1.6 sps |

---

## Mục lục

1. [Vấn đề gốc (root cause)](#1-vấn-đề-gốc-root-cause)
2. [Cải thiện chất lượng model](#2-cải-thiện-chất-lượng-model)
3. [Cải thiện tốc độ training](#3-cải-thiện-tốc-độ-training)
4. [Cải thiện pipeline & code](#4-cải-thiện-pipeline--code)
5. [Xử lý class imbalance — hướng dẫn chọn](#5-xử-lý-class-imbalance--hướng-dẫn-chọn)
6. [Thứ tự ưu tiên thực hiện](#6-thứ-tự-ưu-tiên-thực-hiện)
7. [Kỳ vọng sau cải thiện](#7-kỳ-vọng-sau-cải-thiện)
8. [Checklist nhanh](#8-checklist-nhanh)

---

## 1. Vấn đề gốc (root cause)

### 1.1 PhoBERT thắng vì đúng công cụ cho task

| Yếu tố | PhoBERT | LLM (Qwen2.5 + LoRA) |
|--------|---------|----------------------|
| Kiến trúc | Encoder-only, attention 2 chiều | Decoder-only, attention nhân quả |
| Task format | Classification: `[CLS] → Linear(768→3)` | Generation: sinh chữ `CLEAN`/`OFFENSIVE`/`HATE` |
| Fine-tune | 100% params (135M) | ~0.5–1.25% qua LoRA, base frozen 4-bit |
| Pre-train | ~20GB tiếng Việt chuyên biệt | Đa ngôn ngữ, tiếng Việt thiểu số |
| Loss | `CrossEntropyLoss` trực tiếp trên 3 class | Causal LM loss trên cả prompt + nhãn |
| Inference | `argmax(logits)` | `generate()` + parse string → dễ fallback CLEAN |
| Training steps (85 phút) | Nhiều epoch, batch 64 | Ít step hơn (đặc biệt 7B) |

### 1.2 Paradox: LLM 7B kém nhất

Trong cùng budget 85 phút + VRAM 8GB:
- Batch nhỏ nhất (2), gradient checkpointing, 4-bit nặng nhất
- Ít training steps nhất → underfit
- RLHF bias từ Instruct model → né HATE, over-predict CLEAN

---

## 2. Cải thiện chất lượng model

### 2.1 [Ưu tiên cao] Đổi LLM: Generation → Classification head

**Hiện tại** (`training/train_llm_lora.py`):
- Train: `make_example()` ghép prompt + nhãn text
- Inference: `model.generate(max_new_tokens=5)` → parse substring

**Nên làm:**
```python
hidden = model(...).last_hidden_state[:, -1, :]  # token cuối prompt
logits = classifier(hidden)                     # Linear(hidden_dim → 3)
pred   = logits.argmax(-1)
loss   = CrossEntropyLoss(weight=class_weights)(logits, labels)
```

**Lợi ích:** Cùng objective với PhoBERT; không parse string; gradient tập trung vào quyết định phân loại.

**Cách nhẹ hơn (ít sửa trainer):** Logit-based — so sánh xác suất token `CLEAN` / `OFFENSIVE` / `HATE` tại vị trí assistant, không `generate()`.

**File cần sửa:** `training/train_llm_lora.py`, `inference/infer_all.py`

---

### 2.2 [Ưu tiên cao] Sửa inference parsing

**Hiện tại** (`train_llm_lora.py` dòng ~297, `infer_all.py` dòng ~178):
```python
preds.append(2 if "HATE" in txt else (1 if "OFFENSIVE" in txt else 0))
```

**Vấn đề:**
- Không khớp → mặc định `CLEAN (0)` → bias class đa số (82.7% test)
- Model sinh `"CLEAN."`, refusal text, tiếng Việt lẫn → parse sai

**Cần làm:**
- [ ] Lấy token đầu tiên sau assistant, không dùng substring mơ hồ
- [ ] Không default CLEAN khi output không rõ
- [ ] Ghi log các case parse fail vào CSV để debug
- [ ] Cân nhắc map trực tiếp token ID của `CLEAN`/`OFFENSIVE`/`HATE`

**File:** `training/train_llm_lora.py`, `inference/infer_all.py`

---

### 2.3 [Ưu tiên cao] Bật `completion_only_loss` (nếu giữ SFT)

Nếu **chưa** chuyển classification head, bật ngay trong `SFTConfig`:

```python
completion_only_loss=True
```

**Ý nghĩa:** Loss chỉ tính trên token nhãn (`CLEAN`/`OFFENSIVE`/`HATE`), không trên system prompt + comment.

**Lưu ý:** Đây **không thay** weighted loss — chỉ giúp gradient không bị pha loãng bởi prompt dài.

**File:** `training/train_llm_lora.py` → `SFTConfig`

---

### 2.4 [Ưu tiên cao] `load_best_model_at_end=True`

**Hiện tại:**
```python
load_best_model_at_end=False,
eval_strategy="steps",
eval_steps=200,
```

**Cần làm:**
```python
load_best_model_at_end=True,
metric_for_best_model="eval_loss",
greater_is_better=False,
```

Lưu checkpoint **val tốt nhất**, không phải checkpoint cuối (có thể overfit).

**File:** `training/train_llm_lora.py`

---

### 2.5 [Ưu tiên trung] Weighted loss cho classification head

Khi dùng classification head, copy công thức từ PhoBERT:

```python
# training/train_phobert.py
counts  = train_df["label"].value_counts().sort_index().values
weights = torch.tensor(len(train_df) / (3 * counts), dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights)
```

**Không dùng cùng lúc** với `completion_only_loss` — đó là 2 paradigm khác nhau (xem [Mục 5](#5-xử-lý-class-imbalance--hướng-dẫn-chọn)).

---

### 2.6 [Ưu tiên trung] Focal loss (tùy chọn)

Chỉ thử **sau** weighted CE nếu F1-OFFENSIVE / F1-HATE vẫn thấp.

```python
# Focal: giảm loss sample dễ, giữ sample khó
loss = alpha[class] * (1 - p_t) ** gamma * CE
```

**Không nên** dùng weighted mạnh + focal γ cao cùng lúc → dễ over-predict HATE/OFFENSIVE.

---

### 2.7 [Ưu tiên trung] Giảm RLHF / Instruct bias

- Thử base model `Qwen2.5-1.5B` (không `-Instruct`) + classification head
- Rút gọn system prompt, bỏ framing “chuyên gia an toàn”
- Tăng weight HATE/OFFENSIVE trong loss

**Tham chiếu:** `train_1_5b_lora.py` ghi nhận zero-shot F1=0.1524 do RLHF bias.

---

### 2.8 [Ưu tiên trung] Tăng capacity fine-tune LLM

| Thay đổi | Trade-off |
|----------|-----------|
| Tăng LoRA rank (16→32 cho 3B/7B) | Chậm hơn, tốn VRAM |
| Thêm `embed_tokens`, `lm_head` vào LoRA target | Adapt output layer tốt hơn |
| 8-bit thay 4-bit (nếu VRAM đủ) | Ít mất precision weights |

**File:** `training/train_llm_lora.py` → `MODEL_CONFIGS`, `get_peft_model()`

---

### 2.9 [Ưu tiên thấp / dài hạn] Cải thiện class OFFENSIVE

F1-OFF thấp nhất ở mọi model (~0.26–0.36) vì:
- Ranh giới mờ với CLEAN và HATE
- Chỉ 6.6% test set
- Teen code, tiếng lóng đa dạng

**Hướng xử lý:**
- [ ] Phân tích `logs/*_predictions.csv` — lọc OFF↔CLEAN, OFF↔HATE
- [ ] Data augmentation (paraphrase tiếng Việt)
- [ ] Two-stage: toxic vs non-toxic → rồi OFF vs HATE
- [ ] Pseudo-labeling bằng PhoBERT (đã mô tả trong `REPORT.md` mục 7)

---

## 3. Cải thiện tốc độ training

### 3.1 Dùng ngay (không sửa code)

```bash
pip install unsloth                    # LLM nhanh ~2×
./run_all.sh --auto-skip               # Bỏ model đã train
./run_all.sh --only=phobert,1.5b       # Chỉ train cần thiết
./run_all.sh --sample=300              # Debug ~15 phút

# 2 GPU — train song song
CUDA_VISIBLE_DEVICES=0 python training/train_phobert.py
CUDA_VISIBLE_DEVICES=1 python training/train_llm_lora.py --model-size 1.5b
```

Kiểm tra log có dòng: `[Model] Unsloth OK — tốc độ 2×`

---

### 3.2 Tối ưu config LLM (`train_llm_lora.py`)

| Thay đổi | Hiện tại | Đề xuất | Tiết kiệm ước tính |
|----------|----------|---------|-------------------|
| Eval trong train | mỗi 200 steps | `eval_strategy="no"` hoặc `eval_steps=1000` | 15–25% |
| Max sequence length | 512 | 256 | 30–40% |
| System prompt | ~15 dòng + 3 ví dụ | 1–2 dòng | 10–20% |
| Save checkpoint | mỗi 200 steps | `save_strategy="epoch"` | ~5% |
| LoRA rank | 32/16 | 16/8 (nếu chấp nhận F1 giảm nhẹ) | 10–15% |
| Oversampling | ×3 train size | Bỏ nếu có weighted loss | ~3× ít step/epoch |

---

### 3.3 Tối ưu PhoBERT (`train_phobert.py`)

PhoBERT đã nhanh (~35–92 phút). Có thể thêm:

```python
with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    logits = model(ids, mask)
```

- [ ] `use_fast=True` tokenizer (hiện `use_fast=False`)
- [ ] `BATCH_SIZE=128` nếu VRAM đủ
- [ ] `torch.compile(model)` cho epoch 2+

---

### 3.4 Tách train và eval

- Train xong → chỉ lưu adapter/weights
- Eval dev: `python inference/infer_all.py --sample 500`
- Full test set → chạy 1 lần cuối pipeline

LLM 7B eval full test (~6.680 mẫu) có thể mất 1–2 giờ riêng.

---

## 4. Cải thiện pipeline & code

### 4.1 Thống nhất scripts

| File | Trạng thái | Hành động |
|------|------------|-----------|
| `training/train_llm_lora.py` | Unified, dùng chính | Giữ |
| `training/train_1_5b_lora.py` | Legacy, trùng lặp | Deprecate / xóa |
| `training/train_7b_lora.py` | Legacy, trùng lặp | Deprecate / xóa |

**Lưu ý:** `train_1_5b_lora.py` dùng `val_cleaned.csv`, unified dùng `dev_cleaned.csv` — cần thống nhất 1 split.

---

### 4.2 Logging

- [ ] PhoBERT: thêm `sys.stdout.reconfigure(line_buffering=True)` như LLM
- [ ] Lưu `total_steps`, `samples_seen` vào `*_results.json`
- [ ] Ghi số lượng parse-fail khi eval LLM

---

### 4.3 So sánh công bằng hơn

Để kết luận PhoBERT vs LLM thuyết phục:

- [ ] Báo cáo cùng **số training steps**, không chỉ cùng 85 phút
- [ ] Cùng tiêu chí chọn checkpoint (best val loss/F1)
- [ ] Ablation: generation vs classification head vs logit-based
- [ ] Ghi `trainable_pct`, effective batch, quantization mode

---

## 5. Xử lý class imbalance — hướng dẫn chọn

### Hai paradigm — chọn MỘT

```
Paradigm A: Classification head
├── Weighted CrossEntropy  ← khuyến nghị bắt đầu (giống PhoBERT)
└── Hoặc Focal Loss        ← thử nếu OFF/HATE vẫn kém

Paradigm B: Giữ SFT (generation)
├── completion_only_loss=True
└── Oversampling (đang có) hoặc sample weight trong Trainer
```

### Không nhầm lẫn

| Câu hỏi | Trả lời |
|---------|---------|
| Dùng `completion_only_loss` + weighted loss cùng lúc? | **Không** — 2 paradigm khác nhau |
| Dùng weighted + focal cùng lúc? | **Có thể** nhưng thường chọn **một**; gộp cả hai dễ over-bias minority class |
| Oversampling + weighted loss? | Chỉ khi classification head; với SFT thì oversampling + completion_only_loss là đủ để bắt đầu |

---

## 6. Thứ tự ưu tiên thực hiện

### Phase 1 — Nhanh, impact cao (~1–2 giờ sửa code)

1. Cài Unsloth
2. `completion_only_loss=True` trong SFTConfig
3. Sửa inference parsing (không default CLEAN)
4. `load_best_model_at_end=True`
5. `eval_strategy="no"` hoặc eval thưa hơn
6. `max_length=256` + rút `SYSTEM_PROMPT`

### Phase 2 — Impact lớn (~1–2 ngày)

7. Classification head cho LLM (hoặc logit-based)
8. Weighted CrossEntropy giống PhoBERT
9. Phân tích error OFFENSIVE từ `logs/*_predictions.csv`
10. Thống nhất val split (`dev_cleaned.csv`)

### Phase 3 — Tối ưu thêm

11. `torch.autocast` cho PhoBERT
12. Tăng LoRA rank / thêm target modules
13. Pseudo-labeling pipeline
14. Thử base model thay Instruct
15. Deprecate legacy training scripts

### Phase 4 — Nghiên cứu dài hạn

16. Two-stage classifier (toxic → fine-grained)
17. Vietnamese-specific LLM
18. Continued pre-training trên corpus tiếng Việt

---

## 7. Kỳ vọng sau cải thiện

| Mục tiêu | Hiện tại | Sau Phase 1 | Sau Phase 2 (ước tính) |
|----------|----------|-------------|------------------------|
| PhoBERT F1-Macro | 0.599 | ~0.60 | ~0.61–0.62 (pseudo-label) |
| LLM 1.5B F1-Macro | 0.510 | ~0.53 | ~0.55–0.58 |
| LLM 3B F1-Macro | 0.536 | ~0.55 | ~0.57–0.60 |
| LLM 7B F1-Macro | 0.424 | ~0.45 | ~0.50–0.55 |
| Thời gian 4 model | ~5–6h | ~3–4h | ~3–4h |

**Lưu ý:** Với task classification thuần tiếng Việt, PhoBERT vẫn có lợi thế kiến trúc + domain. Mục tiêu hợp lý là LLM **đuổi kịp**, không nhất thiết vượt rõ trong setup 8GB VRAM + 85 phút.

---

## 8. Checklist nhanh

### Chất lượng
- [ ] Classification head hoặc logit-based inference
- [ ] Sửa parse inference (không default CLEAN)
- [ ] `completion_only_loss=True` (nếu giữ SFT)
- [ ] `load_best_model_at_end=True`
- [ ] Weighted CE (khi có classification head)
- [ ] Phân tích lỗi OFFENSIVE / HATE

### Tốc độ
- [ ] Cài Unsloth
- [ ] Tắt/giảm eval trong train
- [ ] `max_length=256`
- [ ] Rút system prompt
- [ ] `--auto-skip` / train song song GPU
- [ ] Tách train và eval

### Pipeline
- [ ] Thống nhất `dev_cleaned.csv`
- [ ] Deprecate `train_1_5b_lora.py`, `train_7b_lora.py`
- [ ] Log `total_steps` trong results JSON
- [ ] Ablation report trong `REPORT.md`

---

## File liên quan

| File | Vai trò |
|------|---------|
| `training/train_phobert.py` | Baseline tốt nhất — tham chiếu weighted loss |
| `training/train_llm_lora.py` | Script chính cần cải thiện |
| `inference/infer_all.py` | Inference + parse LLM |
| `compare_models.py` | So sánh kết quả sau cải thiện |
| `REPORT.md` | Báo cáo thực nghiệm hiện tại |
| `run_all.sh` | Pipeline train 4 models |

---

*Tài liệu tạo: 17/06/2026 — tổng hợp từ phân tích kỹ thuật PhoBERT vs LLM, classification head, class imbalance, và tối ưu training.*
