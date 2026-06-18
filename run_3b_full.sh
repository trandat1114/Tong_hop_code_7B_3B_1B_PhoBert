#!/usr/bin/env bash
# run_3b_full.sh  —  Launch / resume Qwen2.5-3B long-run training
#
# Usage:
#   bash run_3b_full.sh              # fresh start
#   bash run_3b_full.sh --resume     # resume from checkpoint_latest/
#   bash run_3b_full.sh --eval-only  # evaluate best checkpoint on test set
#
# Flags are forwarded to train_3b_full.py, e.g.:
#   bash run_3b_full.sh --max-hours 24 --early-stop 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/../../../../.venv"  # /home/trandat1114/dev/.venv
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
NOHUP_LOG="$LOG_DIR/nohup_3b_full_${RUN_TS}.log"

# Activate venv if it exists
if [ -d "$VENV" ]; then
    source "$VENV/bin/activate"
    echo "[INFO] venv activated: $VENV"
fi

PYTHON="${PYTHON:-python3}"
TRAIN_SCRIPT="$SCRIPT_DIR/training/train_3b_full.py"

echo "============================================================"
echo " Qwen2.5-3B Full Training — $(date)"
echo " Script : $TRAIN_SCRIPT"
echo " Args   : $*"
echo " Log    : $NOHUP_LOG"
echo "============================================================"
echo ""
echo "Starting in background with nohup..."
echo "Watch progress:  tail -f $NOHUP_LOG"
echo "Kill gracefully: kill -TERM \$(cat $LOG_DIR/train_3b.pid)"
echo ""

nohup "$PYTHON" "$TRAIN_SCRIPT" \
    --max-hours 66 \
    --max-epochs 20 \
    --early-stop 5 \
    --ckpt-every 500 \
    "$@" \
    >> "$NOHUP_LOG" 2>&1 &

PID=$!
echo "$PID" > "$LOG_DIR/train_3b.pid"
echo "[INFO] PID=$PID saved to $LOG_DIR/train_3b.pid"
echo "[INFO] nohup log: $NOHUP_LOG"
