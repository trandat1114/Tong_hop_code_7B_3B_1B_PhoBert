# Phát hiện Ngôn ngữ Độc hại Tiếng Việt — So sánh 4 Mô hình

Dự án so sánh 4 mô hình fine-tuned trên bài toán phân loại bình luận độc hại tiếng Việt (ViHSD dataset):
**PhoBERT**, **Qwen2.5-1.5B**, **Qwen2.5-3B**, **Qwen2.5-7B** — tất cả đều fine-tune trên cùng dữ liệu và điều kiện.

---

## Mục lục

1. [Tổng quan](#1-tổng-quan)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Yêu cầu hệ thống](#3-yêu-cầu-hệ-thống)
4. [Dataset](#4-dataset)
5. [Kiến trúc mô hình](#5-kiến-trúc-mô-hình)
6. [Cách chạy](#6-cách-chạy)
7. [Train 3B Full — long-run (66+ giờ)](#7-train-3b-full--long-run-66-giờ)
8. [Theo dõi training](#8-theo-dõi-training)
9. [Inference](#9-inference)
10. [Kết quả thực nghiệm](#10-kết-quả-thực-nghiệm)
11. [Cấu hình chi tiết](#11-cấu-hình-chi-tiết)

---

## 1. Tổng quan

| Model | Backbone | Phương pháp | Params |
|-------|----------|-------------|--------|
| PhoBERT | `vinai/phobert-base-v2` | Full fine-tune | 135M |
| LLM 1.5B | `Qwen/Qwen2.5-1.5B-Instruct` | LoRA (r=32) | 1.54B |
| LLM 3B | `Qwen/Qwen2.5-3B-Instruct` | LoRA (r=16) | 3.09B |
| LLM 7B | `Qwen/Qwen2.5-7B-Instruct` | LoRA (r=16) | 7.62B |

**Bài toán:** Phân loại 3 nhãn
- `CLEAN` (0) — Bình luận bình thường
- `OFFENSIVE` (1) — Xúc phạm, thô tục
- `HATE` (2) — Thù ghét, kêu gọi bạo lực

**Giới hạn thời gian:** Mỗi model tối đa **85 phút** training.

---

## 2. Cấu trúc thư mục

```
Tong_hop_code_7B_1B_PhoBert/
│
├── training/
│   ├── train_phobert.py       # Fine-tune PhoBERT (full)
│   ├── train_llm_lora.py      # Fine-tune Qwen2.5 1.5B/3B/7B (LoRA, giới hạn 85 phút)
│   └── train_3b_full.py       # Qwen2.5-3B deep fine-tune không giới hạn thời gian (v3)
│
├── inference/
│   └── infer_all.py           # Inference tất cả models hoặc từng model
│
├── models/
│   ├── phobert/               # → custom_phobert_weights.pth (sau khi train)
│   ├── llm_1_5b/lora_adapter/ # → LoRA adapter 1.5B (sau khi train)
│   ├── llm_3b_lora/           # → LoRA adapter 3B (sau khi train)
│   └── llm_7b_lora/           # → LoRA adapter 7B (sau khi train)
│
├── logs/
│   ├── train_phobert.log      # Log training PhoBERT
│   ├── train_1.5b.log         # Log training LLM 1.5B
│   ├── train_3b.log           # Log training LLM 3B
│   ├── train_7b.log           # Log training LLM 7B
│   ├── phobert_results.json   # Kết quả test set PhoBERT
│   ├── llm_1.5b_results.json  # Kết quả test set LLM 1.5B
│   ├── llm_3b_results.json    # Kết quả test set LLM 3B
│   └── llm_7b_results.json    # Kết quả test set LLM 7B
│
├── figures/                   # 7 biểu đồ so sánh (PNG)
│
├── run_all.sh                 # Pipeline chính — train + eval + report
├── run_3b_full.sh             # Launcher cho train_3b_full.py (nohup + PID)
├── status.sh                  # Kiểm tra trạng thái training
├── compare_models.py          # Sinh biểu đồ so sánh
├── generate_report.py         # Sinh REPORT.md
└── REPORT.md                  # Báo cáo hoàn chỉnh (auto-generated)
```

> **Dữ liệu** nằm tại `../ModelLLM_v3/Cleaned/` (cùng cấp với thư mục này):
> `train_cleaned.csv`, `dev_cleaned.csv`, `test_cleaned.csv`

---

## 3. Yêu cầu hệ thống

**Phần cứng tối thiểu:**

| Thành phần  | Yêu cầu                                       |
|-------------|-----------------------------------------------|
| GPU         | NVIDIA ≥ 8GB VRAM (test: RTX 4060 Laptop 8GB) |
| RAM         | ≥ 16GB                                        |
| Disk        | ≥ 30GB (models + cache)                       |

**Python packages:**

```bash
# Môi trường ảo
source ../.venv/bin/activate

# Packages chính
pip install torch transformers peft trl datasets
pip install scikit-learn pandas numpy matplotlib
pip install bitsandbytes accelerate
pip install unsloth  # tùy chọn — tăng tốc 2× cho LLM
```

> Dự án đã được test với Python 3.12, PyTorch 2.x, Unsloth 2026.6.7, GPU RTX 4060 Laptop.

---

## 4. Dataset

**ViHSD** — Vietnamese Hate Speech Detection

| Split | Tổng | CLEAN | OFFENSIVE | HATE |
|-------|------|-------|-----------|------|
| Train | 23,848 | 19,687 (83%) | 1,605 (7%) | 2,556 (11%) |
| Dev | 2,655 | 2,173 (82%) | 212 (8%) | 270 (10%) |
| Test | 6,618 | 5,486 (83%) | 444 (7%) | 688 (10%) |

**Xử lý mất cân bằng:**
- **PhoBERT:** Weighted CrossEntropyLoss (weight ∝ 1/tần_suất)
- **LLM models:** Random oversampling — cân bằng 3 class về `max_count` mỗi epoch

**Cột CSV:** `free_text`, `label_id` (0/1/2), `clean_text`
— scripts tự động chuẩn hoá qua hàm `_norm()`.

---

## 5. Kiến trúc mô hình

### PhoBERT

```
phobert-base-v2 (RoBERTa)
  → [CLS] token embedding (768-dim)
  → Dropout(0.3)
  → Linear(768 → 3)
  → CrossEntropyLoss (weighted)
```

Tối ưu: AdamW + cosine LR scheduler, gradient clipping 1.0, early stopping patience=3.

### Qwen2.5 + LoRA

```
Qwen2.5-{1.5B|3B|7B}-Instruct
  + LoRA adapters trên: q_proj, k_proj, v_proj, o_proj,
                        gate_proj, up_proj, down_proj
  → 4-bit NF4 quantization (BitsAndBytes)
  → SFTTrainer (causal LM)
```

Format prompt:
```
<|im_start|>system
Bạn là chuyên gia phân loại... Chỉ trả lời MỘT TỪ: CLEAN, OFFENSIVE, hoặc HATE.
<|im_end|>
<|im_start|>user
Bình luận: {text}
<|im_end|>
<|im_start|>assistant
{LABEL}<|im_end|>
```

Inference: lấy token đầu tiên được sinh ra, so khớp `HATE` > `OFFENSIVE` > `CLEAN`.

---

## 6. Cách chạy

```bash
cd "/home/trandat1114/dev/Năm 2026/Học UIT HCM/Môn kỹ thuật lập trình Ai/Tong_hop_code_7B_1B_PhoBert"
```

### 6.1 Chạy toàn bộ pipeline

```bash
./run_all.sh
```

Chạy tuần tự: PhoBERT → 1.5B → 3B → 7B → sinh biểu đồ → sinh báo cáo (~6 giờ).

### 6.2 Tham số của `run_all.sh`

| Tham số | Tác dụng |
|---------|----------|
| *(không có)* | Train tất cả 4 models |
| `--skip-phobert` | Bỏ qua PhoBERT, chỉ train 3 LLM |
| `--skip=phobert,7b` | Bỏ qua các model chỉ định (phân cách dấu phẩy) |
| `--only=1.5b,3b` | Chỉ train model được chỉ định |
| `--auto-skip` | Tự bỏ qua model đã có file `*_results.json` |
| `--skip-train` | Bỏ qua toàn bộ training |
| `--plots-only` | Chỉ sinh biểu đồ + báo cáo |
| `--sample=300` | Debug nhanh với 300 mẫu (~15 phút) |

**Ví dụ phổ biến:**

```bash
# PhoBERT đã train xong → chỉ train 3 LLM còn lại
./run_all.sh --skip-phobert

# Tự động phát hiện model nào chưa có kết quả và train tiếp
./run_all.sh --auto-skip

# Chỉ sinh lại biểu đồ + báo cáo (đã có JSON)
./run_all.sh --plots-only

# Debug nhanh toàn pipeline
./run_all.sh --sample=300
```

### 6.3 Chạy từng model độc lập

```bash
source ../.venv/bin/activate

# PhoBERT
python3 -u training/train_phobert.py

# LLM 1.5B
python3 -u training/train_llm_lora.py --model-size 1.5b

# LLM 3B
python3 -u training/train_llm_lora.py --model-size 3b

# LLM 7B
python3 -u training/train_llm_lora.py --model-size 7b

# Debug (500 mẫu)
python3 -u training/train_llm_lora.py --model-size 1.5b --sample 500
```

### 6.4 Sinh biểu đồ và báo cáo

```bash
python3 compare_models.py    # → figures/04_*.png ... figures/10_*.png
python3 generate_report.py   # → REPORT.md
```

---

## 7. Train 3B Full — long-run (66+ giờ)

`training/train_3b_full.py` là phiên bản nâng cấp của `train_3b_full.ipynb`, thiết kế cho việc train liên tục nhiều giờ không cần giám sát.

### Các cải tiến so với notebook

| Tính năng | Mô tả |
|-----------|-------|
| **Resume từ checkpoint** | Lưu toàn bộ trạng thái (optimizer, scheduler, epoch, best_f1) — tiếp tục không mất bước nào |
| **Time-budget stop** | Dừng graceful khi còn 30 phút trước hết giới hạn giờ |
| **Periodic checkpoint** | Lưu checkpoint mỗi N optimizer steps (mặc định 500) |
| **SIGTERM / SIGINT** | Nhận kill signal → lưu checkpoint → thoát sạch (không mất dữ liệu) |
| **Dual log** | Console + file `logs/train_3b_full_<RUN_ID>.log` với timestamp |
| **ETA hiển thị** | Mỗi LOG_EVERY steps in `elapsed`, `ETA` còn lại |
| **Eval-only mode** | `--eval-only`: chạy test evaluation trên best checkpoint mà không train lại |

### Điều kiện dừng (ưu tiên theo thứ tự)

1. **Signal** — nhận `SIGTERM` hoặc `Ctrl+C`: lưu checkpoint ngay lập tức
2. **Time budget** — `elapsed >= max_hours - 0.5h`: dừng trước khi hết giờ 30 phút
3. **Early stopping** — val F1-Macro không cải thiện trong `early_stop` epoch liên tiếp (mặc định 5)
4. **Max epochs** — đủ 20 epochs

### Cách chạy nhanh

```bash
cd "/home/trandat1114/dev/Năm 2026/Học UIT HCM/Môn kỹ thuật lập trình Ai/Tong_hop_code_7B_1B_PhoBert"

# === Khởi động lần đầu (chạy nền, log ra file) ===
bash run_3b_full.sh

# === Resume sau khi bị ngắt / hết điện / restart máy ===
bash run_3b_full.sh --resume

# === Chỉ eval best checkpoint trên test set ===
bash run_3b_full.sh --eval-only

# === Giới hạn thời gian khác (ví dụ: chỉ có 12 giờ) ===
bash run_3b_full.sh --max-hours 12

# === Chạy trực tiếp (foreground) ===
source ../.venv/bin/activate
python3 training/train_3b_full.py --max-hours 66 --max-epochs 20 --early-stop 5
```

### Tất cả CLI options

```
python3 training/train_3b_full.py [options]

  --resume            Tiếp tục từ models/llm_3b_cls/checkpoint_latest/
  --eval-only         Chỉ chạy test evaluation, không train
  --max-hours FLOAT   Giờ tối đa (mặc định 66.0)
  --max-epochs INT    Số epoch tối đa (mặc định 20)
  --early-stop INT    Patience cho early stopping (mặc định 5)
  --ckpt-every INT    Lưu checkpoint mỗi N optimizer steps (mặc định 500)
  --batch-size INT    Batch size per GPU (mặc định 2)
  --lr FLOAT          Learning rate (mặc định 1e-4)
  --seed INT          Random seed (mặc định 42)
```

### Theo dõi tiến trình (long-run)

```bash
# Xem log realtime
tail -f logs/nohup_3b_full_<RUN_TS>.log

# Hoặc log riêng của script
tail -f logs/train_3b_full_<RUN_ID>.log

# Dừng graceful (lưu checkpoint rồi thoát)
kill -TERM $(cat logs/train_3b.pid)

# Dừng ngay lập tức (không recommend — có thể mất step hiện tại)
kill -9 $(cat logs/train_3b.pid)

# Kiểm tra process còn sống không
ps -p $(cat logs/train_3b.pid) -o pid,etime,pcpu,pmem,cmd
```

### Checkpoint

Checkpoint tự động lưu tại `models/llm_3b_cls/checkpoint_latest/`:

```
checkpoint_latest/
├── adapter_config.json       # LoRA config
├── adapter_model.safetensors # LoRA weights (checkpoint hiện tại)
├── optimizer.pt              # Trạng thái optimizer
├── scheduler.pt              # Trạng thái LR scheduler
├── train_state.json          # epoch, step, best_f1, no_improve, run_id
├── step_history.json         # Lịch sử loss theo step
└── epoch_history.json        # Lịch sử val metrics theo epoch
```

Best model (dùng cho inference) lưu tại `models/llm_3b_cls/` (ghi đè mỗi khi val F1 cải thiện).

### Cấu hình training (train_3b_full.py)

| Tham số | Giá trị | Ghi chú |
|---------|---------|---------|
| Model | `Qwen/Qwen2.5-3B-Instruct` | bfloat16 (không quantize) |
| LoRA rank | 64 | 4× so với baseline 3B |
| LoRA alpha | 128 | |
| Max length | 256 | |
| Batch size | 2 | Effective 8 với grad_accum=4 |
| Learning rate | 1e-4 | LLRD decay=0.95 |
| Warmup | 6% steps | Cosine schedule |
| Early stop | 5 epochs | Trên val F1-Macro |
| Max epochs | 20 | Kỳ vọng stop ~5–8 epoch |
| R-Drop | ✓ α=0.5 | KL divergence regularization |
| SupCon | ✓ w=0.1 | Supervised contrastive loss |
| LLRD | ✓ decay=0.95 | Layer-wise LR decay |
| Focal Loss | γ=2, margin=1.0 | Hard-example mining |
| Label smoothing | 0.1 | Calibration |

---

## 8. Theo dõi training (pipeline chính)

```bash
# Tổng quan 1 lần
./status.sh

# Watch mode — tự refresh mỗi 30 giây
./status.sh -w

# Refresh mỗi 10 giây
./status.sh -w 10

# Xem log chi tiết
./status.sh -l phobert    # log PhoBERT
./status.sh -l 1.5b       # log LLM 1.5B
./status.sh -l 3b         # log LLM 3B
./status.sh -l 7b         # log LLM 7B
./status.sh -l master     # pipeline master log
./status.sh -l all        # tail tất cả logs

# Chỉ xem bảng kết quả
./status.sh -r
```

**Giải thích trạng thái:**

```
✓ XONG      — Training hoàn thành, có kết quả JSON
▶ ĐANG CHẠY — Đang train (hiển thị PID + thời gian chạy)
✗ Chưa có   — Chưa train hoặc bị lỗi
```

> **Lưu ý về log:** Python buffer stdout khi redirect vào file. Log PhoBERT sẽ hiển thị đầy đủ sau khi training kết thúc (không phải lỗi). Với LLM models (đã fix `sys.stdout.reconfigure(line_buffering=True)`), log sẽ realtime.

---

## 9. Inference

### Phân loại một câu

```bash
# Dùng PhoBERT
python3 inference/infer_all.py --model phobert --text "câu bình luận cần phân loại"

# Dùng LLM 1.5B
python3 inference/infer_all.py --model llm_1_5b --text "câu bình luận cần phân loại"

# So sánh tất cả 4 models trên cùng 1 câu
python3 inference/infer_all.py --model all --text "câu bình luận cần phân loại"
```

### Đánh giá trên test set

```bash
# Tất cả models (lưu JSON + CSV predictions)
python3 inference/infer_all.py

# Chỉ 1 model, 500 mẫu
python3 inference/infer_all.py --model llm_3b --sample 500
```

Kết quả lưu tại:
- `logs/inference_results.json` — metrics tổng hợp
- `logs/{model}_predictions.csv` — từng dự đoán

---

## 10. Kết quả thực nghiệm

> Số liệu đo thực tế trên **test set 6,618 mẫu**, GPU RTX 4060 Laptop 8GB, ngày 16–17/06/2026.

### 9.1 Bảng tổng hợp

| Model       | Params  | Acc         | F1-Macro    | F1-CLEAN   | F1-OFF     | F1-HATE     | Speed (sps) | Train   |
|-------------|---------|-------------|-------------|------------|------------|-------------|-------------|---------|
| **PhoBERT** | 135M    | **0.7936**  | **0.5991**  | **0.8876** | **0.3641** | **0.5456**  | 7.0         | 92 phút |
| LLM 1.5B    | 1.54B   | 0.7280      | 0.5103      | 0.8572     | 0.3053     | 0.3684      | 3.7         | 85 phút |
| LLM 3B      | 3.09B   | 0.6990      | 0.5361      | 0.8313     | 0.3038     | 0.4732      | 11.0        | 86 phút |
| LLM 7B      | 7.62B   | 0.6458      | 0.4244      | 0.8103     | 0.2623     | 0.2005      | 1.6         | 85 phút |

> Tất cả models bị giới hạn **85 phút** → `time_limited: true` (chưa train hết epoch).

### 9.2 Nhận xét

- **PhoBERT thắng tuyệt đối** về F1-Macro (+7.4pp so với LLM tốt nhất là 3B) dù chỉ có 135M tham số
- **LLM 3B > LLM 1.5B** về F1-HATE (0.4732 vs 0.3684) — model lớn hơn hiểu ngữ cảnh hate speech tốt hơn
- **LLM 7B kém nhất** — lớn nhất nhưng train được ít bước nhất do 8GB VRAM phải chạy 4-bit quantization nặng hơn
- **F1-OFFENSIVE** thấp ở tất cả models (0.26–0.36) — class này ít mẫu và ranh giới với HATE mơ hồ
- **Speed**: LLM 3B nhanh nhất trong 3 LLM (11 sps), LLM 7B chậm nhất (1.6 sps)

### 9.3 Lý do PhoBERT vượt trội LLM

| Yếu tố        | PhoBERT | LLM (Qwen2.5) |
|---------------|---------|---------------|
| Kiến trúc     | Encoder-only (BERT) — tối ưu cho classification | Decoder-only — thiết kế để sinh văn bản |
| Fine-tuning   | **100% params** cập nhật | Chỉ **0.92–3.5%** qua LoRA |
| Ngôn ngữ      | Pre-trained chuyên biệt **tiếng Việt** | Đa ngôn ngữ, tiếng Việt chiếm tỷ lệ nhỏ |
| Quantization  | Full precision | **4-bit NF4** — mất độ chính xác weights |
| Số bước train | Nhiều hơn (model nhỏ → nhanh/bước) | Ít bước hơn trong cùng thời gian |

### 9.4 Biểu đồ so sánh

| File                                | Nội dung                                |
|-------------------------------------|-----------------------------------------|
| `figures/04_f1_comparison.png`      | F1-Macro + F1 theo từng class           |
| `figures/05_confusion_matrices.png` | 4 confusion matrix cạnh nhau            |
| `figures/06_speed_size.png`         | Tốc độ inference vs kích thước model    |
| `figures/07_radar.png`              | Radar chart 6 chiều (Acc, F1, speed, …) |
| `figures/08_tradeoff.png`           | Bubble chart: speed vs F1 vs kích thước |
| `figures/09_training_time.png`      | Thời gian training từng model           |
| `figures/10_summary_table.png`      | Bảng tóm tắt trực quan                  |

---

## 11. Cấu hình chi tiết

### PhoBERT

| Tham số | Giá trị |
|---------|---------|
| Model | `vinai/phobert-base-v2` |
| Max length | 256 tokens |
| Batch size | 64 |
| Learning rate | 2e-5 |
| Optimizer | AdamW (weight_decay=0.01) |
| LR scheduler | Cosine với 10% warmup |
| Max epochs | 10 (early stop patience=3) |
| Time limit | 85 phút |
| Loss | CrossEntropyLoss (weighted) |

### Qwen2.5 LoRA

| Tham số           | 1.5B        | 3B          | 7B          |
|-------------------|-------------|-------------|-------------|
| Batch/device      | 8           | 4           | 2           |
| Grad accumulation | 1           | 2           | 4           |
| Effective batch   | 8           | 8           | 8           |
| LoRA rank (r)     | 32          | 16          | 16          |
| LoRA alpha        | 64          | 32          | 32          |
| LoRA dropout      | 0.05        | 0.05        | 0.05        |
| Learning rate     | 3e-4        | 2e-4        | 2e-4        |
| LR scheduler      | cosine      | cosine      | cosine      |
| Max epochs        | 3           | 3           | 2           |
| Quantization      | 4-bit NF4   | 4-bit NF4   | 4-bit NF4   |
| Optimizer         | adamw_8bit  | adamw_8bit  | adamw_8bit  |
| Time limit        | 85 phút     | 85 phút     | 85 phút     |

**LoRA target modules** (tất cả LLM):
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Ước tính VRAM

| Model     | VRAM khi train  | VRAM khi infer  |
|-----------|-----------------|-----------------|
| PhoBERT   | ~2 GB           | ~0.5 GB         |
| LLM 1.5B  | ~3 GB           | ~1 GB           |
| LLM 3B    | ~5 GB           | ~2 GB           |
| LLM 7B    | ~7.5 GB         | ~4 GB           |

---

## Ghi chú kỹ thuật

**Tại sao log PhoBERT không cập nhật trong khi train?**
Python buffer stdout (8KB) khi redirect vào file. Do `print()` chỉ gọi mỗi 30 bước (~5KB tích lũy), buffer không đủ để flush. Output hiển thị đầy đủ khi training kết thúc. Đây là behavior bình thường, không phải lỗi.

**Tại sao LLM cần `--model-size` flag?**
Cả 3 model LLM dùng chung một script `train_llm_lora.py` với config riêng cho từng size trong `MODEL_CONFIGS`. Giúp tránh code trùng lặp.

**`_norm()` function là gì?**
CSV gốc dùng cột `label_id` và `clean_text`, không phải `label`/`comment`. Hàm `_norm()` tự động rename cột khi đọc CSV, đảm bảo compatibility với mọi phiên bản dataset.

**Tại sao LLM 7B dùng `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` thay vì `Qwen/Qwen2.5-7B-Instruct`?**
Unsloth 2026.6.7 cố tải model tên mới `unsloth/qwen2.5-7b-instruct-unsloth-bnb-4bit` nhưng download thất bại liên tục (0 bytes). Model cache cũ `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (5.4GB) đã có sẵn từ lần chạy trước nên được dùng trực tiếp.

---

## Môi trường thực nghiệm

| Thành phần        | Phiên bản                                 |
|-------------------|-------------------------------------------|
| OS                | Ubuntu (WSL2 — Linux 5.15)                |
| GPU               | NVIDIA GeForce RTX 4060 Laptop (8GB VRAM) |
| Python            | 3.12                                      |
| PyTorch           | 2.12.0+cu130                              |
| Transformers      | 5.5.0                                     |
| Unsloth           | 2026.6.7                                  |
| CUDA              | 13.0 (compute 8.9)                        |
| Ngày thực nghiệm  | 16–17/06/2026                             |
