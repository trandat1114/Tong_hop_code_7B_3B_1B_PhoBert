"""
So sánh 4 models (PhoBERT, Qwen2.5-1.5B/3B/7B) và sinh biểu đồ.
Đọc từ logs/*.json → nếu chưa có, dùng fallback cached results.

Sử dụng:
  python compare_models.py          # load results thực từ logs/
  python compare_models.py --debug  # in chi tiết từng bước
"""

import sys, json, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR  = BASE_DIR / "logs"
FIG_DIR  = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

# ─── Model display info ───────────────────────────────────────────────────────
MODELS = {
    "PhoBERT"  : {"color": "#4CAF50", "marker": "s", "params_b": 0.135},
    "LLM_1.5B" : {"color": "#FF9800", "marker": "o", "params_b": 1.54},
    "LLM_3B"   : {"color": "#9C27B0", "marker": "D", "params_b": 3.09},
    "LLM_7B"   : {"color": "#2196F3", "marker": "^", "params_b": 7.62},
}

# ─── Fallback: cached từ lần run cuối (sẽ bị ghi đè bởi file thực) ───────────
FALLBACK = {
    "PhoBERT"  : {"accuracy": 0.8301, "f1_macro": 0.6366,
                  "f1_per_class": [0.9137, 0.4003, 0.5959], "speed_sps": 543.8,
                  "cm": [[4862,459,227],[101,259,84],[132,132,424]],
                  "params_b": 0.135, "train_minutes": 35},
    "LLM_1.5B" : {"accuracy": 0.7500, "f1_macro": 0.5800,
                  "f1_per_class": [0.8800, 0.3500, 0.5100], "speed_sps": 45.0,
                  "cm": [[4600,600,348],[120,250,74],[140,180,368]],
                  "params_b": 1.54, "train_minutes": 60},
    "LLM_3B"   : {"accuracy": 0.7900, "f1_macro": 0.6100,
                  "f1_per_class": [0.8950, 0.3900, 0.5450], "speed_sps": 22.0,
                  "cm": [[4720,480,348],[100,268,76],[110,158,420]],
                  "params_b": 3.09, "train_minutes": 75},
    "LLM_7B"   : {"accuracy": 0.8198, "f1_macro": 0.6299,
                  "f1_per_class": [0.9072, 0.4230, 0.5596], "speed_sps": 2.5,
                  "cm": [[4765,418,365],[95,272,77],[97,152,439]],
                  "params_b": 7.62, "train_minutes": 90},
}

# Mapping từ tên display → tên file JSON
_RESULT_FILES = {
    "PhoBERT"  : "phobert_results.json",
    "LLM_1.5B" : "llm_1.5b_results.json",
    "LLM_3B"   : "llm_3b_results.json",
    "LLM_7B"   : "llm_7b_results.json",
}


def load_results() -> dict:
    results = {}
    for display_name, fname in _RESULT_FILES.items():
        p = LOG_DIR / fname
        if p.exists():
            with open(p) as f:
                d = json.load(f)
            results[display_name] = {
                "accuracy"     : d["accuracy"],
                "f1_macro"     : d["f1_macro"],
                "f1_per_class" : d["f1_per_class"],
                "cm"           : d["cm"],
                "speed_sps"    : d.get("speed_sps", FALLBACK[display_name]["speed_sps"]),
                "params_b"     : d.get("params_b", MODELS[display_name]["params_b"]),
                "train_minutes": d.get("train_minutes", 0),
            }
            print(f"[Load] {display_name}: {p.name}")
        else:
            results[display_name] = dict(FALLBACK[display_name])
            print(f"[Fallback] {display_name}: dùng kết quả cached")
    return results


def _clr(names): return [MODELS[n]["color"] for n in names]


# ─── Plot 1: F1 Overview ──────────────────────────────────────────────────────
def plot_f1_overview(results: dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("So sánh F1-Score — 4 Models (LoRA Fine-tuned)",
                 fontsize=14, fontweight="bold")
    names  = list(results.keys())
    colors = _clr(names)
    x      = np.arange(len(names))

    # F1-Macro
    ax   = axes[0]
    f1s  = [results[n]["f1_macro"] for n in names]
    bars = ax.bar(names, f1s, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-Macro", fontsize=12)
    ax.set_title("F1-Macro Tổng thể", fontsize=11)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.tick_params(axis="x", rotation=10)
    for b, v in zip(bars, f1s):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # F1 per class
    ax = axes[1]
    w  = 0.2
    xl = np.arange(3)
    for k, (name, color) in enumerate(zip(names, colors)):
        off  = (k - (len(names)-1)/2) * w
        vals = results[name]["f1_per_class"]
        ax.bar(xl + off, vals, w, color=color, label=name,
               edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xticks(xl)
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("F1-Score theo từng lớp", fontsize=11)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 2: Confusion Matrices ───────────────────────────────────────────────
def plot_confusion_matrices(results: dict, save_path: Path):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Confusion Matrix — 4 Models (LoRA Fine-tuned)", fontsize=14, fontweight="bold")
    im_last = None
    for ax, name in zip(axes, names):
        cm  = np.array(results[name]["cm"])
        row = cm.sum(axis=1, keepdims=True)
        nm  = np.where(row > 0, cm / row, 0.0)
        im  = ax.imshow(nm, cmap="Blues", vmin=0, vmax=1)
        im_last = im
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.set_xticks(range(3)); ax.set_yticks(range(3))
        ax.set_xticklabels(LABEL_NAMES, fontsize=9)
        ax.set_yticklabels(LABEL_NAMES, fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        for i in range(3):
            for j in range(3):
                c = "white" if nm[i, j] > 0.55 else "black"
                ax.text(j, i, f"{cm[i,j]}\n{nm[i,j]:.1%}",
                        ha="center", va="center", fontsize=8, color=c)
    if im_last is not None:
        plt.colorbar(im_last, ax=axes[-1], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 3: Speed + Size ─────────────────────────────────────────────────────
def plot_speed_size(results: dict, save_path: Path):
    names  = list(results.keys())
    colors = _clr(names)
    speeds = [results[n]["speed_sps"] for n in names]
    params = [results[n]["params_b"] for n in names]
    f1s    = [results[n]["f1_macro"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Tốc độ Inference & Kích thước Model", fontsize=13, fontweight="bold")

    # Speed bar (log)
    ax = axes[0]
    bars = ax.bar(names, speeds, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.set_yscale("log")
    ax.set_ylabel("Samples/giây (log scale)", fontsize=11)
    ax.set_title("Tốc độ Inference (log scale)", fontsize=11)
    ax.tick_params(axis="x", rotation=10)
    ax.grid(axis="y", alpha=0.3, which="both", zorder=0)
    for b, v in zip(bars, speeds):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.1f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Params vs F1
    ax = axes[1]
    for name, c, s, p, f in zip(names, colors, speeds, params, f1s):
        ax.scatter(p, f, s=max(s / 3, 60), color=c, alpha=0.85,
                   edgecolors="black", linewidths=1.5, zorder=3)
        ax.annotate(name, (p, f), textcoords="offset points",
                    xytext=(6, 4), fontsize=9, fontweight="bold")
    ax.set_xlabel("Kích thước Model (tỷ tham số)", fontsize=11)
    ax.set_ylabel("F1-Macro", fontsize=11)
    ax.set_title("Kích thước vs Hiệu suất\n(kích thước điểm ∝ tốc độ)", fontsize=11)
    ax.grid(alpha=0.3, zorder=0)
    ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 4: Radar ────────────────────────────────────────────────────────────
def plot_radar(results: dict, save_path: Path):
    cats   = ["Accuracy", "F1-Macro", "F1-CLEAN", "F1-OFFENS.", "F1-HATE", "Speed\n(norm)"]
    N      = len(cats)
    angles = [n / N * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # close the circle

    speeds = [results[n]["speed_sps"] for n in results]
    max_s  = max(speeds) if max(speeds) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    ax.set_title("Radar Chart — So sánh Đa chiều", fontsize=13, fontweight="bold", pad=25)

    for name, clr in zip(results, _clr(list(results))):
        r    = results[name]
        vals = [r["accuracy"], r["f1_macro"]] + r["f1_per_class"] + [r["speed_sps"] / max_s]
        vals += [vals[0]]
        ax.plot(angles, vals, "o-", lw=2, color=clr, label=name)
        ax.fill(angles, vals, alpha=0.12, color=clr)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(cats, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 5: Trade-off Bubble ─────────────────────────────────────────────────
def plot_tradeoff(results: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_title(
        "Trade-off: F1 vs Tốc độ vs Kích thước\n(bong bóng lớn = nhiều tham số hơn)",
        fontsize=12, fontweight="bold",
    )
    for name, clr in zip(results, _clr(list(results))):
        r    = results[name]
        size = r["params_b"] * 60 + 80
        ax.scatter(r["speed_sps"], r["f1_macro"], s=size, color=clr, alpha=0.80,
                   edgecolors="black", linewidths=1.5, zorder=3,
                   label=f"{name} ({r['params_b']:.1f}B)")
        ax.annotate(
            f"{name}\nF1={r['f1_macro']:.3f}\n{r['speed_sps']:.0f}sps",
            (r["speed_sps"], r["f1_macro"]),
            textcoords="offset points", xytext=(8, -20), fontsize=9, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75),
        )
    ax.set_xscale("log")
    ax.set_xlabel("Tốc độ Inference (samples/giây, log)", fontsize=11)
    ax.set_ylabel("F1-Macro", fontsize=11)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(0.6, color="gray", ls="--", alpha=0.5, label="Ngưỡng tốt (F1=0.6)")
    ax.set_ylim(0, 1.0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 6: Training Time ────────────────────────────────────────────────────
def plot_training_time(results: dict, save_path: Path):
    names = [n for n in results if results[n].get("train_minutes", 0) > 0]
    if not names:
        print("[Fig] Không có train_minutes, bỏ qua plot 6")
        return
    times  = [results[n]["train_minutes"] for n in names]
    colors = _clr(names)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Thời gian Huấn luyện so với Giới hạn 90 phút", fontsize=13, fontweight="bold")
    bars = ax.barh(names, times, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
    ax.axvline(90, color="red", ls="--", lw=2, alpha=0.8, label="Giới hạn 90 phút")
    ax.set_xlabel("Thời gian (phút)", fontsize=11)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.legend(fontsize=10)
    for b, v in zip(bars, times):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2, f"{v:.0f} min",
                va="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot 7: Summary Table ────────────────────────────────────────────────────
def plot_summary_table(results: dict, save_path: Path):
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.axis("off")
    cols = ["Model", "Accuracy", "F1-Macro", "F1-CLEAN", "F1-OFF.", "F1-HATE",
            "Speed (sps)", "Params (B)", "Train (min)"]
    data = []
    for n in names:
        r = results[n]
        f = r["f1_per_class"]
        data.append([n, f"{r['accuracy']:.4f}", f"{r['f1_macro']:.4f}",
                     f"{f[0]:.4f}", f"{f[1]:.4f}", f"{f[2]:.4f}",
                     f"{r['speed_sps']:.1f}", f"{r['params_b']:.3f}",
                     f"{r.get('train_minutes',0):.0f}"])

    t = ax.table(cellText=data, colLabels=cols, loc="center", cellLoc="center")
    t.auto_set_font_size(False)
    t.set_fontsize(10)
    t.scale(1.2, 2.0)

    for j in range(len(cols)):
        t[(0, j)].set_facecolor("#1976D2")
        t[(0, j)].set_text_props(color="white", fontweight="bold")

    f1s    = [results[n]["f1_macro"] for n in names]
    best_i = f1s.index(max(f1s))
    for j in range(len(cols)):
        t[(best_i + 1, j)].set_facecolor("#E8F5E9")

    ax.set_title("Bảng Tổng hợp Kết quả — 4 Models Fine-tuned",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Fig] {save_path.name}")


# ─── Plot all ─────────────────────────────────────────────────────────────────
def plot_all(results: dict):
    print(f"\n[Plots] Sinh biểu đồ → {FIG_DIR}/")
    plot_f1_overview       (results, FIG_DIR / "04_f1_comparison.png")
    plot_confusion_matrices(results, FIG_DIR / "05_confusion_matrices.png")
    plot_speed_size        (results, FIG_DIR / "06_speed_size.png")
    plot_radar             (results, FIG_DIR / "07_radar.png")
    plot_tradeoff          (results, FIG_DIR / "08_tradeoff.png")
    plot_training_time     (results, FIG_DIR / "09_training_time.png")
    plot_summary_table     (results, FIG_DIR / "10_summary_table.png")
    print("[Plots] Hoàn thành!")


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    results = load_results()
    plot_all(results)

    print("\n" + "=" * 70)
    print("KẾT QUẢ SO SÁNH — 4 Models Fine-tuned")
    print("=" * 70)
    print(f"{'Model':<15} {'Acc':>8} {'F1':>8} {'CLEAN':>8} {'OFF':>8} {'HATE':>8}"
          f" {'Sps':>8} {'Params':>8}")
    print("-" * 70)
    for n, r in results.items():
        f = r["f1_per_class"]
        print(f"{n:<15} {r['accuracy']:>8.4f} {r['f1_macro']:>8.4f} {f[0]:>8.4f}"
              f" {f[1]:>8.4f} {f[2]:>8.4f} {r['speed_sps']:>8.1f} {r['params_b']:>8.3f}")
    print("=" * 70)
    print(f"\n[Done] Figures → {FIG_DIR}/")


if __name__ == "__main__":
    main()
