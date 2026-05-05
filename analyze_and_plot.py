"""
Experiment analysis and plotting for thesis.
Reads experiments_results.csv, generates charts, and prints analysis summary.

Usage: python analyze_and_plot.py

Output: figures/ folder with PNG charts; same fig*.png (+ simulation_insights.md) are copied to figures/final_figures/.
"""
import csv
import shutil
from pathlib import Path
import sys

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import numpy.ma as ma
except ImportError:
    print("Install matplotlib: pip install matplotlib")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent
CSV_PATH = PROJECT_ROOT / "experiments_results.csv"
if not CSV_PATH.exists():
    alt = PROJECT_ROOT / "experiments_results_new.csv"
    if alt.exists():
        CSV_PATH = alt
FIG_DIR = PROJECT_ROOT / "figures"
# Thesis “final” bundle (PPT / Word); keep in sync with figures/ batch outputs
FIG_FINAL_DIR = FIG_DIR / "final_figures"

# Unified thesis style: HiveMind vs EdgePipe (colorblind-friendly blues / oranges)
HM_COLOR = "#1f4e79"
EP_COLOR = "#c55a11"
HM_EDGE = "#0d2840"
EP_EDGE = "#7a3408"
FIG_DPI = 180
FACE = "#f7f8fa"

# X-axis order for fig3/fig4/fig7 (low→high; must match run_batch_experiments & heatmap grid)
BANDWIDTH_SWEEP_ORDER = [(10, 20), (21, 31), (40, 60)]
PERF_SWEEP_ORDER = [(20, 40), (41, 60), (60, 100)]


def apply_theme():
    """Grid + typography; try seaborn whitegrid if available."""
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
            "axes.edgecolor": "#2b2b2b",
            "axes.linewidth": 0.85,
            "axes.labelcolor": "#1a1a1a",
            "axes.titleweight": "600",
            "axes.titlesize": 11.5,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": True,
            "legend.framealpha": 0.95,
            "legend.edgecolor": "#d0d4dc",
            "grid.alpha": 0.4,
            "grid.linestyle": "-",
            "font.family": "sans-serif",
        }
    )


def save_figure(path: Path) -> None:
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight", facecolor=FACE, edgecolor="none")


def sync_batch_figures_to_final():
    """Copy figures/fig*.png and simulation_insights.md into figures/final_figures/."""
    FIG_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    copied = 0
    for p in sorted(FIG_DIR.glob("fig*.png")):
        if p.parent != FIG_DIR:
            continue
        shutil.copy2(p, FIG_FINAL_DIR / p.name)
        copied += 1
    md = FIG_DIR / "simulation_insights.md"
    if md.exists():
        shutil.copy2(md, FIG_FINAL_DIR / "simulation_insights.md")
    if copied:
        print(f"Synced {copied} file(s) under fig*.png (+ simulation_insights.md) -> {FIG_FINAL_DIR}")


def load_data():
    """Prefer experiments_results.csv (primary batch output); fallback to *_new.csv if needed."""
    rows = []
    for p in [PROJECT_ROOT / "experiments_results.csv", PROJECT_ROOT / "experiments_results_new.csv"]:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    row["deviceNum"] = int(row["deviceNum"])
                    row["bandwidthMin"] = int(row["bandwidthMin"])
                    row["bandwidthMax"] = int(row["bandwidthMax"])
                    row["perfMin"] = int(row["perfMin"])
                    row["perfMax"] = int(row["perfMax"])
                    row["throughput"] = float(row["throughput"]) if row["throughput"] else 0
                    row["inferenceTime"] = float(row["inferenceTime"]) if row["inferenceTime"] else None
                    rows.append(row)
            break
    return rows


def filter_by(rows, **kwargs):
    return [r for r in rows if all(r.get(k) == v for k, v in kwargs.items())]


def plot_model_comparison(rows):
    """Throughput and inference time by model (5 devices, bw 21-31, perf 41-60)."""
    data = filter_by(rows, deviceNum=5, bandwidthMin=21, bandwidthMax=31, perfMin=41, perfMax=60)
    models = sorted(set(r["model"] for r in data))
    hm_tp = [next((r["throughput"] for r in data if r["model"] == m and r["algorithm"] == "HiveMind"), 0) for m in models]
    ep_tp = [next((r["throughput"] for r in data if r["model"] == m and r["algorithm"] == "EdgePipe"), 0) for m in models]
    hm_ti = [next((r["inferenceTime"] or 0 for r in data if r["model"] == m and r["algorithm"] == "HiveMind"), 0) for m in models]
    ep_ti = [next((r["inferenceTime"] or 0 for r in data if r["model"] == m and r["algorithm"] == "EdgePipe"), 0) for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.arange(len(models))
    w = 0.35
    ax1.bar(x - w / 2, hm_tp, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax1.bar(x + w / 2, ep_tp, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.set_ylabel("Throughput (batches/s)")
    ax1.set_title("Throughput by model (5 devices, baseline)")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.35)

    ax2.bar(x - w / 2, hm_ti, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax2.bar(x + w / 2, ep_ti, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.set_ylabel("Inference time (s)")
    ax2.set_title("Inference time by model (5 devices, baseline)")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig1_model_comparison.png")
    plt.close()
    print("Saved fig1_model_comparison.png")


def _format_device_count_axis(ax, devs):
    """Integer ticks 3,4,5,6 (no 3.0) for device-count line plots."""
    devs_int = [int(d) for d in devs]
    ax.set_xticks(devs_int)
    ax.set_xticklabels([str(n) for n in devs_int])
    ax.set_xlim(min(devs_int) - 0.25, max(devs_int) + 0.25)


def _grouped_sweep_bars(
    ax,
    keys: list,
    hm_vals: list,
    ep_vals: list,
    *,
    xlabel: str,
) -> None:
    """Grouped bars left→right in exact `keys` order (categorical x). Avoids tick↔label misalignment on numeric x."""
    n = len(keys)
    x = np.arange(n, dtype=float)
    w = 0.35
    off = w * 0.5
    ax.bar(x - off, hm_vals, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax.bar(x + off, ep_vals, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    tick_labels = [f"{a}–{b}" for a, b in keys]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(xlabel)
    ax.margins(x=0.08)


def _device_count_sweep(rows, model: str):
    """Return (devs, hm_tp, ep_tp) for device-count sweep, or None if no rows."""
    data = [r for r in rows if r["model"] == model and r["bandwidthMin"] == 21 and r["perfMin"] == 41]
    if not data:
        return None
    devs = sorted({int(r["deviceNum"]) for r in data})
    hm_tp = [next((r["throughput"] for r in data if int(r["deviceNum"]) == d and r["algorithm"] == "HiveMind"), 0) for d in devs]
    ep_tp = [next((r["throughput"] for r in data if int(r["deviceNum"]) == d and r["algorithm"] == "EdgePipe"), 0) for d in devs]
    return devs, hm_tp, ep_tp


def _plot_device_count_on_ax(ax, devs, hm_tp, ep_tp, title: str, *, show_legend: bool) -> None:
    ax.plot(devs, hm_tp, "o-", color=HM_COLOR, label="HiveMind", markersize=7, linewidth=2.2)
    ax.plot(devs, ep_tp, "s-", color=EP_COLOR, label="EdgePipe", markersize=7, linewidth=2.2)
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title(title)
    _format_device_count_axis(ax, devs)
    if show_legend:
        ax.legend(loc="best")
    ax.grid(True, alpha=0.4)


def plot_device_count(rows):
    """Throughput vs device count (AlexNet, bw 21-31, perf 41-60)."""
    s = _device_count_sweep(rows, "AlexNet")
    if not s:
        print("Skip fig2_device_count (no AlexNet device sweep)")
        return
    devs, hm_tp, ep_tp = s
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    _plot_device_count_on_ax(ax, devs, hm_tp, ep_tp, "Throughput vs device count (AlexNet, baseline)", show_legend=True)
    ax.set_xlabel("Number of devices")
    plt.tight_layout()
    save_figure(FIG_DIR / "fig2_device_count.png")
    plt.close()
    print("Saved fig2_device_count.png")


def plot_bandwidth(rows):
    """Throughput vs bandwidth range (AlexNet, 5 devices, perf 41–60). X: 10–20 → 21–31 → 40–60 MB/s (fixed order)."""
    data = [
        r
        for r in rows
        if r["model"] == "AlexNet"
        and r["deviceNum"] == 5
        and r["perfMin"] == 41
        and r["perfMax"] == 60
    ]
    keys = list(BANDWIDTH_SWEEP_ORDER)
    hm_tp, ep_tp = [], []
    for a, b in keys:
        hm = next(
            (x["throughput"] for x in data if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "HiveMind"),
            0,
        )
        ep = next(
            (x["throughput"] for x in data if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "EdgePipe"),
            0,
        )
        hm_tp.append(hm)
        ep_tp.append(ep)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    _grouped_sweep_bars(ax, keys, hm_tp, ep_tp, xlabel="Inter-device link bandwidth (MB/s)")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs bandwidth range (AlexNet, 5 devices, perf 41–60)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig3_bandwidth.png")
    plt.close()
    print("Saved fig3_bandwidth.png")


def plot_performance_range(rows):
    """Throughput vs device performance (AlexNet, 5 dev, bw 21–31). X: 20–40 → 41–60 → 60–100 GFlop/s (fixed order)."""
    data = [
        r
        for r in rows
        if r["model"] == "AlexNet"
        and r["deviceNum"] == 5
        and r["bandwidthMin"] == 21
        and r["bandwidthMax"] == 31
    ]
    keys = list(PERF_SWEEP_ORDER)
    hm_tp, ep_tp = [], []
    for a, b in keys:
        hm = next(
            (x["throughput"] for x in data if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "HiveMind"),
            0,
        )
        ep = next(
            (x["throughput"] for x in data if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "EdgePipe"),
            0,
        )
        hm_tp.append(hm)
        ep_tp.append(ep)
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    _grouped_sweep_bars(ax, keys, hm_tp, ep_tp, xlabel="Device performance (GFlop/s)")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs device performance (AlexNet, 5 devices, bw 21–31 MB/s)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig4_performance.png")
    plt.close()
    print("Saved fig4_performance.png")


def print_analysis(rows):
    """Print concise analysis summary."""
    print("\n" + "=" * 50)
    print("ANALYSIS SUMMARY")
    print("=" * 50)
    by_model = filter_by(rows, deviceNum=5, bandwidthMin=21, perfMin=41)
    print("\n1. Model comparison (5 devices, bw 21-31, perf 41-60):")
    for m in sorted(set(r["model"] for r in by_model)):
        hm = next((r for r in by_model if r["model"] == m and r["algorithm"] == "HiveMind"), None)
        ep = next((r for r in by_model if r["model"] == m and r["algorithm"] == "EdgePipe"), None)
        if hm and ep:
            winner_tp = "EdgePipe" if ep["throughput"] > hm["throughput"] else "HiveMind"
            winner_ti = "HiveMind" if (hm["inferenceTime"] or 999) < (ep["inferenceTime"] or 999) else "EdgePipe"
            print(f"   {m}: EdgePipe throughput {ep['throughput']:.2f} vs HiveMind {hm['throughput']:.2f} | "
                  f"Lower latency: {winner_ti}")
    print("\n2. Key finding: EdgePipe generally achieves higher throughput; HiveMind often has lower inference latency.")
    print("=" * 50)


def _baseline_rows(rows):
    """Baseline sweep: 5 dev, bw 21–31, perf 41–60."""
    return filter_by(rows, deviceNum=5, bandwidthMin=21, bandwidthMax=31, perfMin=41, perfMax=60)


def plot_pareto_throughput_latency(rows):
    """
    Scatter: throughput vs inference time (trade-off), baseline config, all models.
    """
    data = _baseline_rows(rows)
    models = sorted(set(r["model"] for r in data))
    fig, ax = plt.subplots(figsize=(7.2, 5))
    colors = {"HiveMind": HM_COLOR, "EdgePipe": EP_COLOR}
    for algo in ("HiveMind", "EdgePipe"):
        xs, ys, labels = [], [], []
        for m in models:
            r = next((x for x in data if x["model"] == m and x["algorithm"] == algo), None)
            if not r or r["inferenceTime"] is None:
                continue
            xs.append(r["inferenceTime"])
            ys.append(r["throughput"])
            labels.append(m[:3])
        ax.scatter(
            xs,
            ys,
            c=colors[algo],
            s=110,
            label=algo,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.75,
            zorder=3,
        )
        for x, y, lb in zip(xs, ys, labels):
            ax.annotate(lb, (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8.5, color=colors[algo])
    ax.set_xlabel("Inference time (s) — lower is better")
    ax.set_ylabel("Throughput (batches/s) — higher is better")
    ax.set_title("Throughput–latency trade-off (baseline topology)")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig5_pareto_tradeoff.png")
    plt.close()
    print("Saved fig5_pareto_tradeoff.png")


def plot_edgepipe_speedup(rows):
    """Bar chart: EdgePipe / HiveMind throughput ratio per model (baseline)."""
    data = _baseline_rows(rows)
    models = sorted(set(r["model"] for r in data))
    ratios = []
    labels = []
    for m in models:
        hm = next((r for r in data if r["model"] == m and r["algorithm"] == "HiveMind"), None)
        ep = next((r for r in data if r["model"] == m and r["algorithm"] == "EdgePipe"), None)
        if hm and ep and hm["throughput"] and hm["throughput"] > 0:
            ratios.append(ep["throughput"] / hm["throughput"])
            labels.append(m)
    if not ratios:
        print("Skip fig6_edgepipe_speedup (no baseline rows)")
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    x = np.arange(len(labels))
    ax.bar(x, ratios, color="#2d6a4f", edgecolor="#1b4332", linewidth=0.4)
    ax.axhline(1.0, color="#6c757d", linestyle="--", linewidth=1.1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Throughput ratio (EdgePipe / HiveMind)")
    ax.set_title("Relative throughput gain of EdgePipe (baseline)")
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig6_edgepipe_speedup.png")
    plt.close()
    print("Saved fig6_edgepipe_speedup.png")


def plot_inference_alexnet_bw_perf(rows):
    """Mirror fig3/fig4 but for inference time (AlexNet, 5 dev)."""
    # Bandwidth sweep — same filter as plot_bandwidth; fixed BANDWIDTH_SWEEP_ORDER
    data_bw = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["perfMin"] == 41 and r["perfMax"] == 60]
    bw_keys = list(BANDWIDTH_SWEEP_ORDER)
    hm_ti, ep_ti = [], []
    for a, b in bw_keys:
        hm = next((x for x in data_bw if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "HiveMind"), None)
        ep = next((x for x in data_bw if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "EdgePipe"), None)
        hm_ti.append(hm["inferenceTime"] or 0 if hm else 0)
        ep_ti.append(ep["inferenceTime"] or 0 if ep else 0)

    # Perf sweep — same as plot_performance_range; fixed PERF_SWEEP_ORDER
    data_pf = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["bandwidthMin"] == 21 and r["bandwidthMax"] == 31]
    pf_keys = list(PERF_SWEEP_ORDER)
    hm_ti2, ep_ti2 = [], []
    for a, b in pf_keys:
        hm = next((x for x in data_pf if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "HiveMind"), None)
        ep = next((x for x in data_pf if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "EdgePipe"), None)
        hm_ti2.append(hm["inferenceTime"] or 0 if hm else 0)
        ep_ti2.append(ep["inferenceTime"] or 0 if ep else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    _grouped_sweep_bars(ax1, bw_keys, hm_ti, ep_ti, xlabel="Inter-device link bandwidth (MB/s)")
    ax1.set_ylabel("Inference time (s)")
    ax1.set_title("Inference time vs bandwidth (AlexNet, 5 dev, perf 41–60)")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.35)

    _grouped_sweep_bars(ax2, pf_keys, hm_ti2, ep_ti2, xlabel="Device performance (GFlop/s)")
    ax2.set_ylabel("Inference time (s)")
    ax2.set_title("Inference time vs device performance (AlexNet, 5 dev, bw 21–31 MB/s)")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig7_inference_bw_perf.png")
    plt.close()
    print("Saved fig7_inference_bw_perf.png")


def _matrix_alexnet_grid(rows, algo: str, metric: str):
    """3×3 grid: bandwidth rows × performance cols. metric: 'throughput' or 'inferenceTime'."""
    bw_order = list(BANDWIDTH_SWEEP_ORDER)
    perf_order = list(PERF_SWEEP_ORDER)
    mat = np.full((3, 3), np.nan, dtype=float)
    for i, (bmin, bmax) in enumerate(bw_order):
        for j, (pmin, pmax) in enumerate(perf_order):
            r = next(
                (
                    x
                    for x in rows
                    if x["model"] == "AlexNet"
                    and x["deviceNum"] == 5
                    and x["bandwidthMin"] == bmin
                    and x["bandwidthMax"] == bmax
                    and x["perfMin"] == pmin
                    and x["perfMax"] == pmax
                    and x["algorithm"] == algo
                ),
                None,
            )
            if r:
                v = r[metric]
                if metric == "inferenceTime" and v is None:
                    continue
                mat[i, j] = float(v) if v is not None else np.nan
    return mat, bw_order, perf_order


def plot_bw_perf_heatmaps(rows):
    """Two heatmaps: throughput for HiveMind and EdgePipe (AlexNet 5 dev, 3×3 grid)."""
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    plotted = False
    for ax, algo, title in zip(
        axes,
        ("HiveMind", "EdgePipe"),
        ("Throughput (HiveMind)", "Throughput (EdgePipe)"),
    ):
        mat, bw_order, perf_order = _matrix_alexnet_grid(rows, algo, "throughput")
        if np.all(np.isnan(mat)):
            ax.axis("off")
            ax.text(
                0.5,
                0.5,
                f"No grid data for {algo}\n(run batch with Exp 6 corners)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
            )
            continue
        plotted = True
        mat_ma = ma.masked_invalid(mat)
        # origin="lower": row 0 = bottom = lowest bandwidth; Y increases upward (consistent with fig3/4)
        im = ax.imshow(mat_ma, cmap="cividis", aspect="equal", origin="lower")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"{p[0]}–{p[1]}" for p in perf_order])
        ax.set_yticklabels([f"{b[0]}–{b[1]}" for b in bw_order])
        ax.set_xlabel("Device performance (GFlop/s)")
        ax.set_ylabel("Bandwidth (MB/s)")
        ax.set_title(title)
        for i in range(3):
            for j in range(3):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center", color="white", fontsize=9, fontweight="600")
        plt.colorbar(im, ax=ax, fraction=0.046, label="Throughput (batches/s)")
    plt.suptitle(
        "Cross-sensitivity: bandwidth × device performance (AlexNet, 5 devices)",
        fontsize=12,
        fontweight="600",
        y=1.02,
    )
    plt.tight_layout()
    save_figure(FIG_DIR / "fig8_bw_perf_heatmap.png")
    plt.close()
    if plotted:
        print("Saved fig8_bw_perf_heatmap.png")
    else:
        print("fig8_bw_perf_heatmap: empty (run run_batch_experiments.py with Exp 6 configs)")


def plot_vgg19_device_count(rows):
    """Throughput vs device count for Vgg19 (same as fig2 for AlexNet)."""
    s = _device_count_sweep(rows, "Vgg19")
    if not s:
        print("Skip fig9_vgg19_device_count (no Vgg19 device sweep — run batch with Exp 5)")
        return
    devs, hm_tp, ep_tp = s
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    _plot_device_count_on_ax(ax, devs, hm_tp, ep_tp, "Throughput vs device count (Vgg19, baseline)", show_legend=True)
    ax.set_xlabel("Number of devices")
    plt.tight_layout()
    save_figure(FIG_DIR / "fig9_vgg19_device_count.png")
    plt.close()
    print("Saved fig9_vgg19_device_count.png")


def write_extended_summary_md(rows):
    """Auto-generated bullets for thesis / 答辩."""
    lines = [
        "# Simulation extended insights (auto-generated)",
        "",
        "Generated by `analyze_and_plot.py` from `experiments_results.csv`.",
        "",
        "## Baseline (5 devices, bw 21–31, perf 41–60)",
        "",
    ]
    data = _baseline_rows(rows)
    models = sorted(set(r["model"] for r in data))
    for m in models:
        hm = next((r for r in data if r["model"] == m and r["algorithm"] == "HiveMind"), None)
        ep = next((r for r in data if r["model"] == m and r["algorithm"] == "EdgePipe"), None)
        if hm and ep:
            ratio = ep["throughput"] / hm["throughput"] if hm["throughput"] else 0
            lines.append(
                f"- **{m}**: EdgePipe throughput **{ep['throughput']:.2f}** vs HiveMind **{hm['throughput']:.2f}** "
                f"(×{ratio:.2f}); single-sample latency **{hm['inferenceTime']:.4f} s** (HM) vs **{ep['inferenceTime']:.4f} s** (EP)."
            )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "| File | Content |",
            "|------|---------|",
            "| fig1–fig4 | Original model / device / bandwidth / performance sweeps |",
            "| fig5_pareto_tradeoff | Throughput vs inference time (trade-off) |",
            "| fig6_edgepipe_speedup | EdgePipe / HiveMind throughput ratio |",
            "| fig7_inference_bw_perf | Inference time vs bandwidth and vs performance |",
            "| fig8_bw_perf_heatmap | 3×3 cross-sensitivity heatmap (needs full batch) |",
            "| fig9_vgg19_device_count | Vgg19 throughput vs device count |",
            "| fig10_pipeline_stage_times | Per-stage time + bottleneck (AlexNet, 5 devices, seed=1; auto in `generate_all_thesis_figures.py`) |",
            "",
        ]
    )
    path = FIG_DIR / "simulation_insights.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main():
    FIG_DIR.mkdir(exist_ok=True)
    apply_theme()
    if not CSV_PATH.exists():
        print("Run run_batch_experiments.py first to generate experiments_results.csv")
        return
    rows = load_data()
    print(f"Loaded {len(rows)} rows")
    plot_model_comparison(rows)
    plot_device_count(rows)
    plot_bandwidth(rows)
    plot_performance_range(rows)
    plot_pareto_throughput_latency(rows)
    plot_edgepipe_speedup(rows)
    plot_inference_alexnet_bw_perf(rows)
    plot_bw_perf_heatmaps(rows)
    plot_vgg19_device_count(rows)
    write_extended_summary_md(rows)
    sync_batch_figures_to_final()
    print_analysis(rows)


if __name__ == "__main__":
    main()
