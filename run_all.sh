#!/usr/bin/env bash
# ============================================================
# run_all.sh — Pipeline đầy đủ: Train 4 models + Evaluate + Report
#
# Sử dụng:
#   ./run_all.sh                        # Train tất cả 4 models
#   ./run_all.sh --skip-phobert         # Bỏ qua PhoBERT, chỉ train 3 LLM
#   ./run_all.sh --skip-train           # Bỏ qua toàn bộ training
#   ./run_all.sh --plots-only           # Chỉ sinh biểu đồ + báo cáo
#   ./run_all.sh --auto-skip            # Tự bỏ qua model đã có kết quả JSON
#   ./run_all.sh --only=1.5b,3b         # Chỉ train model được chỉ định
#   ./run_all.sh --skip=phobert,7b      # Bỏ qua các model được chỉ định
#   ./run_all.sh --sample=300           # Debug nhanh (~15 phút)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/../.venv"
LOG_DIR="$SCRIPT_DIR/logs"
FIG_DIR="$SCRIPT_DIR/figures"
DATA_DIR="$SCRIPT_DIR/../ModelLLM_v3/Cleaned"

mkdir -p "$LOG_DIR" "$FIG_DIR"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log()  { echo -e "${CYAN}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
err()  { echo -e "${RED}[ERR]${NC} $1" >&2; }
skip() { echo -e "${YELLOW}[SKIP]${NC} $1"; }
step() { echo -e "\n${BLUE}${BOLD}╔══ $1 ══╗${NC}"; }

# ─── Argument parsing ─────────────────────────────────────────────────────────
SKIP_TRAIN=false
PLOTS_ONLY=false
AUTO_SKIP=false
SAMPLE_ARG=""
# Mảng model sẽ train: phobert 1.5b 3b 7b
TRAIN_PHOBERT=true
TRAIN_1_5B=true
TRAIN_3B=true
TRAIN_7B=true

for arg in "$@"; do
    case $arg in
        --skip-train)    SKIP_TRAIN=true ;;
        --plots-only)    PLOTS_ONLY=true; SKIP_TRAIN=true ;;
        --skip-phobert)  TRAIN_PHOBERT=false ;;
        --auto-skip)     AUTO_SKIP=true ;;
        --sample=*)      SAMPLE_ARG="--sample ${arg#*=}" ;;

        --only=*)
            # --only=1.5b,3b  → chỉ train model được liệt kê
            TRAIN_PHOBERT=false; TRAIN_1_5B=false
            TRAIN_3B=false;      TRAIN_7B=false
            IFS=',' read -ra MLIST <<< "${arg#*=}"
            for m in "${MLIST[@]}"; do
                case "$m" in
                    phobert) TRAIN_PHOBERT=true ;;
                    1.5b)    TRAIN_1_5B=true ;;
                    3b)      TRAIN_3B=true ;;
                    7b)      TRAIN_7B=true ;;
                    *) err "Model không hợp lệ: $m (dùng: phobert|1.5b|3b|7b)" ;;
                esac
            done
            ;;

        --skip=*)
            # --skip=phobert,7b  → bỏ qua model được liệt kê
            IFS=',' read -ra MLIST <<< "${arg#*=}"
            for m in "${MLIST[@]}"; do
                case "$m" in
                    phobert) TRAIN_PHOBERT=false ;;
                    1.5b)    TRAIN_1_5B=false ;;
                    3b)      TRAIN_3B=false ;;
                    7b)      TRAIN_7B=false ;;
                    *) err "Model không hợp lệ: $m (dùng: phobert|1.5b|3b|7b)" ;;
                esac
            done
            ;;

        --help|-h)
            sed -n '3,12p' "$0" | sed 's/^#//'
            exit 0 ;;
    esac
done

# ─── Auto-skip: bỏ qua model đã có JSON kết quả ──────────────────────────────
if $AUTO_SKIP && ! $SKIP_TRAIN; then
    [ -f "$LOG_DIR/phobert_results.json" ]     && TRAIN_PHOBERT=false
    [ -f "$LOG_DIR/llm_1.5b_results.json" ]    && TRAIN_1_5B=false
    [ -f "$LOG_DIR/llm_3b_results.json" ]      && TRAIN_3B=false
    [ -f "$LOG_DIR/llm_7b_results.json" ]      && TRAIN_7B=false
fi

# ─── Banner ───────────────────────────────────────────────────────────────────
echo -e "${BOLD}"
cat << 'BANNER'
╔═══════════════════════════════════════════════════════════════╗
║  Pipeline: LLM Fine-tuning vs PhoBERT cho Độc hại Tiếng Việt ║
║  Models: PhoBERT | Qwen2.5-1.5B | 3B | 7B (all LoRA)         ║
║  Dataset: ViHSD — Mỗi model tối đa 85 phút training          ║
╚═══════════════════════════════════════════════════════════════╝
Tham số:
  (không có)            Train tất cả 4 models
  --skip-phobert        Bỏ qua PhoBERT, chỉ train 3 LLM
  --skip=phobert,7b     Bỏ qua các model chỉ định (dấu phẩy)
  --only=1.5b,3b        Chỉ train model chỉ định
  --auto-skip           Tự bỏ qua model đã có kết quả JSON
  --skip-train          Bỏ qua toàn bộ training
  --plots-only          Chỉ sinh biểu đồ + báo cáo
  --sample=300          Debug nhanh với 300 mẫu
BANNER
echo -e "${NC}"
echo "Bắt đầu: $(date '+%Y-%m-%d %H:%M:%S')"

# ─── Step 1: Môi trường ───────────────────────────────────────────────────────
step "BƯỚC 1: Kiểm tra Môi trường"

if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
    ok "venv: $VENV_DIR"
else
    warn "Không tìm thấy venv tại $VENV_DIR, dùng Python hệ thống"
fi

PYTHON=python3
PY_VER=$($PYTHON --version 2>&1)
ok "Python: $PY_VER"

GPU_INFO=$($PYTHON -c "
import torch
if torch.cuda.is_available():
    n = torch.cuda.get_device_name(0)
    m = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'GPU: {n} ({m:.1f}GB VRAM)')
else:
    print('CPU only (training sẽ rất chậm)')
" 2>/dev/null || echo "torch chưa cài")
ok "$GPU_INFO"

# ─── Step 2: Kiểm tra dữ liệu ────────────────────────────────────────────────
step "BƯỚC 2: Kiểm tra Dữ liệu"

if [ ! -d "$DATA_DIR" ]; then
    err "Không tìm thấy data dir: $DATA_DIR"
    exit 1
fi

for split in train dev test; do
    f="$DATA_DIR/${split}_cleaned.csv"
    if [ -f "$f" ]; then
        n=$(($(wc -l < "$f") - 1))
        ok "${split}: $n samples"
    else
        err "Thiếu file: $f"
        exit 1
    fi
done

# ─── Step 3: Training ─────────────────────────────────────────────────────────
if ! $SKIP_TRAIN; then
    step "BƯỚC 3: Training — tối đa 85 phút/model"

    # Hiển thị kế hoạch train
    echo ""
    for label_flag in "PhoBERT:$TRAIN_PHOBERT" "LLM 1.5B:$TRAIN_1_5B" "LLM 3B:$TRAIN_3B" "LLM 7B:$TRAIN_7B"; do
        lbl="${label_flag%%:*}"; flag="${label_flag##*:}"
        if $flag; then log "  → Sẽ train: $lbl"
        else       skip "  → Bỏ qua:   $lbl"
        fi
    done
    echo ""

    _run_model() {
        local label="$1" script="$2" extra="$3" logfile="$4"
        log "$label bắt đầu..."
        local START; START=$(date +%s)
        python3 -u "$SCRIPT_DIR/$script" $extra $SAMPLE_ARG 2>&1 | tee "$LOG_DIR/$logfile"
        local ELAPSED=$(( $(date +%s) - START ))
        ok "$label xong — ${ELAPSED}s ($(( ELAPSED/60 ))min)"
    }

    # 3a. PhoBERT (~35-85 phút)
    if $TRAIN_PHOBERT; then
        _run_model "PhoBERT" "training/train_phobert.py" "" "train_phobert.log"
    else
        skip "PhoBERT — bỏ qua (dùng kết quả hiện có)"
    fi

    # 3b. LLM 1.5B (~60-85 phút)
    if $TRAIN_1_5B; then
        _run_model "LLM 1.5B" "training/train_llm_lora.py" "--model-size 1.5b" "train_1.5b.log"
    else
        skip "LLM 1.5B — bỏ qua"
    fi

    # 3c. LLM 3B (~75-85 phút)
    if $TRAIN_3B; then
        _run_model "LLM 3B" "training/train_llm_lora.py" "--model-size 3b" "train_3b.log"
    else
        skip "LLM 3B — bỏ qua"
    fi

    # 3d. LLM 7B (~85-90 phút)
    if $TRAIN_7B; then
        _run_model "LLM 7B" "training/train_llm_lora.py" "--model-size 7b" "train_7b.log"
    else
        skip "LLM 7B — bỏ qua"
    fi

    ok "Training hoàn thành!"
fi

# ─── Step 4: Kiểm tra model checkpoints ──────────────────────────────────────
step "BƯỚC 4: Kiểm tra Checkpoints"

check_exists() {
    local path="$1"
    local name="$2"
    if [ -f "$path" ]; then
        SIZE=$(du -sh "$path" | cut -f1)
        ok "$name: $path ($SIZE)"
    else
        warn "$name chưa có: $path"
    fi
}

check_exists "$SCRIPT_DIR/models/phobert/custom_phobert_weights.pth" "PhoBERT weights"
check_exists "$SCRIPT_DIR/models/llm_1_5b/lora_adapter/adapter_config.json" "1.5B LoRA"
check_exists "$SCRIPT_DIR/models/llm_3b_lora/adapter_config.json" "3B LoRA"
check_exists "$SCRIPT_DIR/models/llm_7b_lora/adapter_config.json" "7B LoRA"

# ─── Step 5: So sánh + Biểu đồ ───────────────────────────────────────────────
step "BƯỚC 5: So sánh Models & Sinh Biểu đồ"
log "Đọc kết quả từ logs/*.json và sinh biểu đồ..."
$PYTHON "$SCRIPT_DIR/compare_models.py" 2>&1 | tee "$LOG_DIR/compare.log"

echo ""
FIGS=0
for f in "$FIG_DIR/"*.png; do
    [ -f "$f" ] && { FIGS=$((FIGS+1)); echo "  ✓ $(basename "$f")"; }
done
ok "$FIGS biểu đồ đã sinh → $FIG_DIR/"

# ─── Step 6: Sinh báo cáo ────────────────────────────────────────────────────
step "BƯỚC 6: Sinh Báo cáo"
$PYTHON "$SCRIPT_DIR/generate_report.py" 2>&1 | tee "$LOG_DIR/report.log"
REPORT_SIZE=$(wc -c < "$SCRIPT_DIR/REPORT.md" 2>/dev/null || echo 0)
ok "REPORT.md ($REPORT_SIZE bytes) → $SCRIPT_DIR/REPORT.md"

# ─── Step 7: Tóm tắt ─────────────────────────────────────────────────────────
step "BƯỚC 7: Tóm tắt Kết quả"

$PYTHON - << 'PYEOF'
import json
from pathlib import Path

log_dir = Path(__file__).parent / "logs" if False else Path("logs")
log_dir = Path.cwd() / "logs"

configs = [
    ("phobert_results.json",   "PhoBERT     "),
    ("llm_1.5b_results.json",  "LLM 1.5B   "),
    ("llm_3b_results.json",    "LLM 3B     "),
    ("llm_7b_results.json",    "LLM 7B     "),
]

print(f"\n{'Model':<15} {'Acc':>8} {'F1':>8} {'CLEAN':>8} {'OFF':>8} {'HATE':>8} {'Sps':>8} {'Train':>7}")
print("-" * 70)
found_any = False
for fname, name in configs:
    p = log_dir / fname
    if p.exists():
        d = json.load(open(p))
        f = d["f1_per_class"]
        tm = d.get("train_minutes", 0)
        print(f"{name:<15} {d['accuracy']:>8.4f} {d['f1_macro']:>8.4f}"
              f" {f[0]:>8.4f} {f[1]:>8.4f} {f[2]:>8.4f} {d.get('speed_sps',0):>8.1f} {tm:>6.0f}m")
        found_any = True
    else:
        print(f"{name:<15} {'[chưa có kết quả]':>55}")

if not found_any:
    print("Chưa có kết quả nào — hãy chạy training trước!")
PYEOF

echo ""
echo -e "${GREEN}${BOLD}Pipeline hoàn thành!${NC}"
echo "Kết thúc: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Xem báo cáo : $SCRIPT_DIR/REPORT.md"
echo "Xem biểu đồ : $FIG_DIR/"
echo "Xem logs    : $LOG_DIR/"
