#!/usr/bin/env bash
# ============================================================
# status.sh — Kiểm tra trạng thái pipeline training
#
# Sử dụng:
#   ./status.sh              # Xem tổng quan 1 lần
#   ./status.sh -w           # Watch mode (cập nhật mỗi 30 giây)
#   ./status.sh -l phobert   # Xem log chi tiết: phobert|1.5b|3b|7b
#   ./status.sh -l all       # Xem tail tất cả logs
#   ./status.sh -r           # Chỉ xem kết quả (JSON đã có)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
MODEL_DIR="$SCRIPT_DIR/models"
FIG_DIR="$SCRIPT_DIR/figures"

# ── Colors ────────────────────────────────────────────────────────────────────
R='\033[0;31m'; G='\033[0;32m'; Y='\033[1;33m'
B='\033[0;34m'; C='\033[0;36m'; M='\033[0;35m'
BOLD='\033[1m'; DIM='\033[2m'; NC='\033[0m'

# ── Helpers ───────────────────────────────────────────────────────────────────
hr()   { printf "${DIM}%s${NC}\n" "$(printf '─%.0s' {1..62})"; }
ok()   { printf "  ${G}✓${NC} $1\n"; }
warn() { printf "  ${Y}⚠${NC}  $1\n"; }
run()  { printf "  ${C}▶${NC} $1\n"; }
fail() { printf "  ${R}✗${NC} $1\n"; }
info() { printf "  ${DIM}·${NC} $1\n"; }

# ── Kiểm tra 1 model ──────────────────────────────────────────────────────────
check_model() {
    local key="$1"   # phobert | 1.5b | 3b | 7b
    local name="$2"
    local json_file="$3"
    local pid_pattern="$4"
    local weight_path="$5"

    local json_path="$LOG_DIR/$json_file"
    local is_running=false
    local pid=""

    pid=$(pgrep -f "$pid_pattern" 2>/dev/null | head -1)
    [ -n "$pid" ] && is_running=true

    printf "\n${BOLD}%-12s${NC}" "[$name]"

    if [ -f "$json_path" ]; then
        # Kết quả có sẵn
        local acc f1 speed train_min
        acc=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d['accuracy']:.4f}\")" 2>/dev/null)
        f1=$(python3 -c  "import json; d=json.load(open('$json_path')); print(f\"{d['f1_macro']:.4f}\")" 2>/dev/null)
        speed=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d.get('speed_sps',0):.0f}\")" 2>/dev/null)
        train_min=$(python3 -c "import json; d=json.load(open('$json_path')); print(f\"{d.get('train_minutes',0):.0f}\")" 2>/dev/null)
        printf "${G}XONG${NC}  Acc=${BOLD}$acc${NC}  F1=${BOLD}$f1${NC}  Speed=${speed}sps  Train=${train_min}min\n"
    elif $is_running; then
        # Đang chạy — tính thời gian
        local elapsed=""
        local log_key="${key//./_}"
        local log_path="$LOG_DIR/train_${log_key}.log"
        [ "$key" = "phobert" ] && log_path="$LOG_DIR/train_phobert.log"
        [ "$key" = "1.5b" ]   && log_path="$LOG_DIR/train_1.5b.log"
        [ "$key" = "3b" ]     && log_path="$LOG_DIR/train_3b.log"
        [ "$key" = "7b" ]     && log_path="$LOG_DIR/train_7b.log"

        if [ -f "$log_path" ]; then
            local mtime now
            mtime=$(stat -c %Y "$log_path" 2>/dev/null || echo 0)
            now=$(date +%s)
            elapsed_s=$(( now - mtime ))
            # Thử lấy thời gian từ process start
            local pstart
            pstart=$(stat -c %Y /proc/$pid/stat 2>/dev/null || echo $now)
            elapsed_s=$(( now - pstart ))
            elapsed="${elapsed_s}s ($((elapsed_s/60))min)"
        fi
        printf "${C}ĐANG CHẠY${NC}  PID=$pid  ${elapsed}\n"
        # Hiện dòng cuối log (lọc control chars)
        if [ -f "$log_path" ]; then
            local last
            last=$(strings "$log_path" 2>/dev/null | grep -v "^$" | tail -1)
            [ -n "$last" ] && printf "            ${DIM}└ $last${NC}\n"
        fi
    elif [ -f "$weight_path" ]; then
        warn "Model có nhưng thiếu JSON kết quả — cần chạy inference"
    else
        fail "Chưa train / chưa có kết quả"
    fi
}

# ── GPU status ────────────────────────────────────────────────────────────────
show_gpu() {
    local util mem_used mem_total
    read -r util mem_used mem_total <<< \
        "$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
           --format=csv,noheader,nounits 2>/dev/null | tr ',' ' ')"

    if [ -n "$util" ]; then
        local mem_pct=$(( mem_used * 100 / mem_total ))
        # Bar for GPU util
        local bars=$(( util * 20 / 100 ))
        local bar=""
        for ((i=0; i<20; i++)); do
            [ $i -lt $bars ] && bar+="█" || bar+="░"
        done

        local util_color="$G"
        [ "$util" -gt 50 ] && util_color="$Y"
        [ "$util" -gt 85 ] && util_color="$C"

        printf "  GPU  ${util_color}${bar}${NC} ${BOLD}%3d%%${NC}  VRAM: ${mem_used}/${mem_total}MB (${mem_pct}%%)\n" "$util"
    else
        warn "Không đọc được GPU (nvidia-smi)"
    fi
}

# ── Pipeline master log ───────────────────────────────────────────────────────
show_pipeline() {
    local master="$LOG_DIR/pipeline_master.log"
    if [ -f "$master" ]; then
        printf "\n${BOLD}Pipeline log:${NC}\n"
        while IFS= read -r line; do
            if echo "$line" | grep -q "XONG"; then
                printf "  ${G}$line${NC}\n"
            elif echo "$line" | grep -q "ĐANG\|bat dau"; then
                printf "  ${C}$line${NC}\n"
            elif echo "$line" | grep -q "LOI\|ERROR"; then
                printf "  ${R}$line${NC}\n"
            else
                printf "  ${DIM}$line${NC}\n"
            fi
        done < "$master"
    fi
}

# ── Kết quả JSON ─────────────────────────────────────────────────────────────
show_results() {
    local found=0
    printf "\n${BOLD}%-14s %8s %8s %8s %8s %8s %7s${NC}\n" \
        "Model" "Acc" "F1" "CLEAN" "OFF" "HATE" "Train"
    hr

    for entry in "phobert_results.json:PhoBERT" "llm_1.5b_results.json:LLM 1.5B" \
                 "llm_3b_results.json:LLM 3B" "llm_7b_results.json:LLM 7B"; do
        local fname="${entry%%:*}"
        local label="${entry##*:}"
        local p="$LOG_DIR/$fname"
        if [ -f "$p" ]; then
            python3 - "$p" "$label" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
f = d["f1_per_class"]
print(f"  {sys.argv[2]:<12} {d['accuracy']:>8.4f} {d['f1_macro']:>8.4f} "
      f"{f[0]:>8.4f} {f[1]:>8.4f} {f[2]:>8.4f} {d.get('train_minutes',0):>6.0f}m")
PYEOF
            found=$((found+1))
        else
            printf "  ${DIM}%-12s %8s${NC}\n" "$label" "[chưa có]"
        fi
    done
    hr
    [ $found -eq 0 ] && warn "Chưa có kết quả nào. Hãy đợi training hoàn thành."
    return $found
}

# ── Figures ───────────────────────────────────────────────────────────────────
show_figures() {
    local n
    n=$(ls "$FIG_DIR/"*.png 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        ok "$n biểu đồ → $FIG_DIR/"
        ls "$FIG_DIR/"*.png 2>/dev/null | while read -r f; do
            local mtime
            mtime=$(date -r "$f" '+%H:%M %d/%m' 2>/dev/null)
            info "$(basename "$f")  ($mtime)"
        done
    else
        warn "Chưa có biểu đồ"
    fi
}

# ── Main view ─────────────────────────────────────────────────────────────────
show_main() {
    clear
    printf "${BOLD}${B}╔══════════════════════════════════════════════════════════════╗${NC}\n"
    printf "${BOLD}${B}║  ViHSD Training Monitor — $(date '+%H:%M:%S %d/%m/%Y')                    ║${NC}\n"
    printf "${BOLD}${B}╚══════════════════════════════════════════════════════════════╝${NC}\n"

    # GPU
    printf "\n${BOLD}GPU:${NC}\n"
    show_gpu

    # Models
    printf "\n${BOLD}Trạng thái Models:${NC}\n"
    hr
    check_model "phobert" "PhoBERT"  "phobert_results.json" \
        "train_phobert.py" "$MODEL_DIR/phobert/custom_phobert_weights.pth"
    check_model "1.5b" "LLM 1.5B" "llm_1.5b_results.json" \
        "train_llm_lora.py.*1.5b" "$MODEL_DIR/llm_1_5b/lora_adapter/adapter_config.json"
    check_model "3b"   "LLM 3B"   "llm_3b_results.json" \
        "train_llm_lora.py.*3b" "$MODEL_DIR/llm_3b_lora/adapter_config.json"
    check_model "7b"   "LLM 7B"   "llm_7b_results.json" \
        "train_llm_lora.py.*7b" "$MODEL_DIR/llm_7b_lora/adapter_config.json"
    hr

    # Pipeline log
    show_pipeline

    # Results table
    printf "\n${BOLD}Kết quả Test Set:${NC}\n"
    show_results

    # Figures
    printf "\n${BOLD}Biểu đồ:${NC}\n"
    show_figures

    printf "\n${DIM}Gõ './status.sh -h' để xem thêm tùy chọn${NC}\n\n"
}

# ── Log viewer ────────────────────────────────────────────────────────────────
show_log() {
    local key="$1"
    local log_path=""
    case "$key" in
        phobert) log_path="$LOG_DIR/train_phobert.log" ;;
        1.5b)    log_path="$LOG_DIR/train_1.5b.log" ;;
        3b)      log_path="$LOG_DIR/train_3b.log" ;;
        7b)      log_path="$LOG_DIR/train_7b.log" ;;
        master)  log_path="$LOG_DIR/pipeline_master.log" ;;
        all)
            for f in "$LOG_DIR/"train_*.log; do
                [ -f "$f" ] || continue
                printf "\n${BOLD}=== $(basename "$f") ===${NC}\n"
                strings "$f" | tail -15
            done
            return ;;
        *)
            printf "${R}Log không hợp lệ: $key${NC}\n"
            printf "Dùng: phobert | 1.5b | 3b | 7b | master | all\n"
            return 1 ;;
    esac

    if [ ! -f "$log_path" ]; then
        warn "Log chưa có: $log_path"
        return 1
    fi

    printf "${BOLD}=== $log_path ===${NC}\n"
    printf "${DIM}(tail -50, strings để decode ctrl chars)${NC}\n\n"
    strings "$log_path" | tail -50
}

# ── Help ──────────────────────────────────────────────────────────────────────
show_help() {
    cat << 'EOF'

HƯỚNG DẪN SỬ DỤNG status.sh
══════════════════════════════

  ./status.sh              Xem tổng quan (1 lần)
  ./status.sh -w           Watch mode — tự refresh mỗi 30 giây
  ./status.sh -w 10        Watch mode — refresh mỗi 10 giây
  ./status.sh -l phobert   Xem 50 dòng cuối log PhoBERT
  ./status.sh -l 1.5b      Xem log LLM 1.5B
  ./status.sh -l 3b        Xem log LLM 3B
  ./status.sh -l 7b        Xem log LLM 7B
  ./status.sh -l master    Xem pipeline master log
  ./status.sh -l all       Xem tất cả logs (tail 15 mỗi file)
  ./status.sh -r           Chỉ hiện bảng kết quả
  ./status.sh -h           Hiện hướng dẫn này

TRẠNG THÁI TRAINING:
  ✓ XONG    — Training hoàn thành, có kết quả JSON
  ▶ ĐANG CHẠY — Đang train (hiển thị PID + thời gian)
  ✗ Chưa có — Chưa train hoặc bị lỗi

CỘT KẾT QUẢ:
  Acc   — Accuracy trên test set (6,618 mẫu)
  F1    — F1-Macro (trung bình 3 class)
  CLEAN — F1 class CLEAN (bình thường)
  OFF   — F1 class OFFENSIVE (xúc phạm)
  HATE  — F1 class HATE (thù ghét)
  Train — Thời gian training (phút)

PIPELINE COMMANDS:
  # Chạy lại toàn bộ từ đầu:
  bash /tmp/pipeline_v2.sh &

  # Chỉ sinh biểu đồ + báo cáo (sau khi đã có JSON):
  cd <project_dir> && python3 compare_models.py && python3 generate_report.py

  # Inference một câu:
  python3 inference/infer_all.py --model phobert --text "câu cần phân loại"
  python3 inference/infer_all.py --model llm_1_5b --text "câu cần phân loại"
  python3 inference/infer_all.py --model all --text "câu cần phân loại"

EOF
}

# ── Argument parsing ──────────────────────────────────────────────────────────
MODE="main"
LOG_KEY=""
WATCH_SECS=30

while [[ $# -gt 0 ]]; do
    case "$1" in
        -w|--watch)
            MODE="watch"
            shift
            [[ "$1" =~ ^[0-9]+$ ]] && { WATCH_SECS="$1"; shift; }
            ;;
        -l|--log)
            MODE="log"
            LOG_KEY="${2:-all}"
            shift 2
            ;;
        -r|--results)
            MODE="results"
            shift
            ;;
        -h|--help)
            MODE="help"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$MODE" in
    main)
        show_main
        ;;
    watch)
        printf "${C}Watch mode — cập nhật mỗi ${WATCH_SECS}s. Ctrl+C để thoát.${NC}\n"
        while true; do
            show_main
            printf "${DIM}Refresh sau ${WATCH_SECS}s... (Ctrl+C thoát)${NC}\n"
            sleep "$WATCH_SECS"
        done
        ;;
    log)
        show_log "$LOG_KEY"
        ;;
    results)
        printf "\n${BOLD}Kết quả Test Set — $(date '+%H:%M:%S'):${NC}\n"
        show_results
        printf "\n"
        ;;
    help)
        show_help
        ;;
esac
