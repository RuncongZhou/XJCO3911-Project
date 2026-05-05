"""
Thesis figure: pipeline stage times + bottleneck (matches /api/simulate logic).
Run: python figures/generate_pipeline_stage_fig.py
Output: figures/fig10_pipeline_stage_times.png
"""
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import app as app_module

FIG_DIR = PROJECT_ROOT / "figures"
HM_COLOR = "#1f4e79"
EP_COLOR = "#c55a11"
HM_EDGE = "#0d2840"
EP_EDGE = "#7a3408"
FIG_DPI = 180
FACE = "#f7f8fa"


def apply_theme():
    for name in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
        try:
            plt.style.use(name)
            break
        except OSError:
            continue
    plt.rcParams.update(
        {
            "figure.facecolor": FACE,
            "axes.facecolor": FACE,
            "font.family": "sans-serif",
        }
    )


def main():
    apply_theme()
    # Same RNG order as run_batch_experiments.run_single_compare (Table 4.1 baselines);
    # two separate /api/simulate calls each re-seed(1) and are NOT comparable to the CSV.
    paired = app_module.simulate_metrics_chained_batch_order(
        "AlexNet",
        5,
        ("HiveMind", "EdgePipe"),
        [21, 31],
        [41, 60],
        seed=1,
        link_eff=1.0,
    )
    assert len(paired) == 2
    hm, ep = paired[0], paired[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.8))

    def _one(ax, data, algo_name):
        stages = data.get("stageTimes") or []
        bn = int(data.get("bottleneckStageIndex", -1))
        n = len(stages)
        vals = [float(s) if s is not None and np.isfinite(s) else 0.0 for s in stages]
        y = np.arange(n)
        base = HM_COLOR if algo_name == "HiveMind" else EP_COLOR
        edge = HM_EDGE if algo_name == "HiveMind" else EP_EDGE
        bar_colors = [base] * n
        bar_edges = [edge] * n
        if 0 <= bn < n:
            bar_colors[bn] = "#c2410c"
            bar_edges[bn] = "#7c2d12"
        ax.barh(y, vals, color=bar_colors, edgecolor=bar_edges, linewidth=0.45)
        for yi, v in zip(y, vals):
            if v <= 0.0 and n > 0:
                ax.text(
                    max(vals) * 0.01 if max(vals) > 0 else 1e-4,
                    yi,
                    "0",
                    ha="left",
                    va="center",
                    fontsize=7,
                    color="#555555",
                )
        ax.set_yticks(y)
        do = data.get("deviceOrder") or list(range(n))
        labels = []
        for i in range(n):
            dev = do[i] if i < len(do) else i
            mark = " (bottleneck)" if i == bn else ""
            labels.append(f"Stage {i} (dev {dev}){mark}")
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Stage time (s)")
        tp = data.get("throughput")
        ti = data.get("inferenceTime")
        ax.set_title(
            f"{algo_name} — stage times (batch/s={tp:.2f}, latency={ti:.4f}s)"
            if ti is not None
            else f"{algo_name} — stage times (batch/s={tp:.2f})"
        )
        ax.grid(True, axis="x", alpha=0.35)
        vmax = max(vals) if vals else 1.0
        ax.set_xlim(0, max(vmax * 1.12, 1e-6))

    _one(ax1, hm, "HiveMind")
    _one(ax2, ep, "EdgePipe")

    fig.suptitle(
        "Pipeline per-stage time (AlexNet, 5 devices; same RNG order as Table 4.1 / run_batch_experiments Exp 1)",
        fontsize=12,
        fontweight="600",
        y=1.02,
    )
    plt.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    out = FIG_DIR / "fig10_pipeline_stage_times.png"
    plt.savefig(out, dpi=FIG_DPI, bbox_inches="tight", facecolor=FACE, edgecolor="none")
    plt.close()
    print(f"Saved {out}")
    final = FIG_DIR / "final_figures" / out.name
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out, final)
    print(f"Copied -> {final}")


if __name__ == "__main__":
    main()
