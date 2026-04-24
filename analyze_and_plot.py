"""
Experiment analysis and plotting for thesis.
Reads experiments_results.csv, generates charts, and prints analysis summary.

Usage: python analyze_and_plot.py

Output: figures/ folder with PNG charts (180 dpi, unified HiveMind/EdgePipe colors, light grid).
"""
import csv
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

# Unified thesis style: HiveMind vs EdgePipe (colorblind-friendly blues / oranges)
HM_COLOR = "#1f4e79"
EP_COLOR = "#c55a11"
HM_EDGE = "#0d2840"
EP_EDGE = "#7a3408"
FIG_DPI = 180
FACE = "#f7f8fa"


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


def plot_device_count(rows):
    """Throughput vs device count (AlexNet, bw 21-31, perf 41-60)."""
    data = [r for r in rows if r["model"] == "AlexNet" and r["bandwidthMin"] == 21 and r["perfMin"] == 41]
    devs = sorted(set(r["deviceNum"] for r in data))
    hm_tp = [next((r["throughput"] for r in data if r["deviceNum"] == d and r["algorithm"] == "HiveMind"), 0) for d in devs]
    ep_tp = [next((r["throughput"] for r in data if r["deviceNum"] == d and r["algorithm"] == "EdgePipe"), 0) for d in devs]

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(devs, hm_tp, "o-", color=HM_COLOR, label="HiveMind", markersize=7, linewidth=2.2)
    ax.plot(devs, ep_tp, "s-", color=EP_COLOR, label="EdgePipe", markersize=7, linewidth=2.2)
    ax.set_xlabel("Number of devices")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs device count (AlexNet, baseline)")
    ax.set_xticks(devs)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig2_device_count.png")
    plt.close()
    print("Saved fig2_device_count.png")


def plot_bandwidth(rows):
    """Throughput vs bandwidth range (AlexNet, 5 devices, perf 41-60)."""
    data = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["perfMin"] == 41]
    bw_labels = []
    hm_tp, ep_tp = [], []
    seen = set()
    for r in data:
        key = (r["bandwidthMin"], r["bandwidthMax"])
        if key in seen:
            continue
        seen.add(key)
        bw_labels.append(f"{r['bandwidthMin']}-{r['bandwidthMax']}")
        hm = next((x["throughput"] for x in data if x["bandwidthMin"] == r["bandwidthMin"] and x["algorithm"] == "HiveMind"), 0)
        ep = next((x["throughput"] for x in data if x["bandwidthMin"] == r["bandwidthMin"] and x["algorithm"] == "EdgePipe"), 0)
        hm_tp.append(hm)
        ep_tp.append(ep)
    x = np.arange(len(bw_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.bar(x - w / 2, hm_tp, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax.bar(x + w / 2, ep_tp, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\nMB/s" for lb in bw_labels])
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs bandwidth range (AlexNet, 5 devices)")
    ax.legend(loc="upper left")
    ax.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig3_bandwidth.png")
    plt.close()
    print("Saved fig3_bandwidth.png")


def plot_performance_range(rows):
    """Throughput vs device performance range (AlexNet, 5 devices, bw 21-31)."""
    data = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["bandwidthMin"] == 21]
    perf_labels = []
    hm_tp, ep_tp = [], []
    seen = set()
    for r in data:
        key = (r["perfMin"], r["perfMax"])
        if key in seen:
            continue
        seen.add(key)
        perf_labels.append(f"{r['perfMin']}-{r['perfMax']}")
        hm = next((x["throughput"] for x in data if x["perfMin"] == r["perfMin"] and x["algorithm"] == "HiveMind"), 0)
        ep = next((x["throughput"] for x in data if x["perfMin"] == r["perfMin"] and x["algorithm"] == "EdgePipe"), 0)
        hm_tp.append(hm)
        ep_tp.append(ep)
    x = np.arange(len(perf_labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.bar(x - w / 2, hm_tp, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax.bar(x + w / 2, ep_tp, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb}\nGFlops/s" for lb in perf_labels])
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs device performance range (AlexNet, 5 devices)")
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
    # Bandwidth sweep — same filter as plot_bandwidth
    data_bw = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["perfMin"] == 41 and r["perfMax"] == 60]
    seen_bw = set()
    bw_keys = []
    for r in data_bw:
        key = (r["bandwidthMin"], r["bandwidthMax"])
        if key in seen_bw:
            continue
        seen_bw.add(key)
        bw_keys.append(key)
    bw_keys.sort(key=lambda t: t[0])
    bw_labels = [f"{a}-{b}" for a, b in bw_keys]
    hm_ti, ep_ti = [], []
    for a, b in bw_keys:
        hm = next((x for x in data_bw if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "HiveMind"), None)
        ep = next((x for x in data_bw if x["bandwidthMin"] == a and x["bandwidthMax"] == b and x["algorithm"] == "EdgePipe"), None)
        hm_ti.append(hm["inferenceTime"] or 0 if hm else 0)
        ep_ti.append(ep["inferenceTime"] or 0 if ep else 0)

    # Perf sweep
    data_pf = [r for r in rows if r["model"] == "AlexNet" and r["deviceNum"] == 5 and r["bandwidthMin"] == 21 and r["bandwidthMax"] == 31]
    seen_pf = set()
    pf_keys = []
    for r in data_pf:
        key = (r["perfMin"], r["perfMax"])
        if key in seen_pf:
            continue
        seen_pf.add(key)
        pf_keys.append(key)
    pf_keys.sort(key=lambda t: t[0])
    pf_labels = [f"{a}-{b}" for a, b in pf_keys]
    hm_ti2, ep_ti2 = [], []
    for a, b in pf_keys:
        hm = next((x for x in data_pf if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "HiveMind"), None)
        ep = next((x for x in data_pf if x["perfMin"] == a and x["perfMax"] == b and x["algorithm"] == "EdgePipe"), None)
        hm_ti2.append(hm["inferenceTime"] or 0 if hm else 0)
        ep_ti2.append(ep["inferenceTime"] or 0 if ep else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    xb = np.arange(len(bw_labels))
    w = 0.35
    ax1.bar(xb - w / 2, hm_ti, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax1.bar(xb + w / 2, ep_ti, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax1.set_xticks(xb)
    ax1.set_xticklabels([f"{lb}\nMB/s" for lb in bw_labels])
    ax1.set_ylabel("Inference time (s)")
    ax1.set_title("Inference time vs bandwidth (AlexNet, 5 dev)")
    ax1.legend(loc="upper left")
    ax1.grid(True, axis="y", alpha=0.35)

    xp = np.arange(len(pf_labels))
    ax2.bar(xp - w / 2, hm_ti2, w, label="HiveMind", color=HM_COLOR, edgecolor=HM_EDGE, linewidth=0.45)
    ax2.bar(xp + w / 2, ep_ti2, w, label="EdgePipe", color=EP_COLOR, edgecolor=EP_EDGE, linewidth=0.45)
    ax2.set_xticks(xp)
    ax2.set_xticklabels([f"{lb}\nGFlops/s" for lb in pf_labels])
    ax2.set_ylabel("Inference time (s)")
    ax2.set_title("Inference time vs device performance (AlexNet, 5 dev)")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.35)
    plt.tight_layout()
    save_figure(FIG_DIR / "fig7_inference_bw_perf.png")
    plt.close()
    print("Saved fig7_inference_bw_perf.png")


def _matrix_alexnet_grid(rows, algo: str, metric: str):
    """3×3 grid: bandwidth rows × performance cols. metric: 'throughput' or 'inferenceTime'."""
    bw_order = [(10, 20), (21, 31), (40, 60)]
    perf_order = [(20, 40), (41, 60), (60, 100)]
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
        im = ax.imshow(mat_ma, cmap="cividis", aspect="equal")
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels([f"{p[0]}-{p[1]}" for p in perf_order])
        ax.set_yticklabels([f"{b[0]}-{b[1]}" for b in bw_order])
        ax.set_xlabel("Device perf range (GFlops/s)")
        ax.set_ylabel("Bandwidth range (MB/s)")
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
    data = [r for r in rows if r["model"] == "Vgg19" and r["bandwidthMin"] == 21 and r["perfMin"] == 41]
    if not data:
        print("Skip fig9_vgg19_device_count (no Vgg19 device sweep — run batch with Exp 5)")
        return
    devs = sorted(set(r["deviceNum"] for r in data))
    hm_tp = [next((r["throughput"] for r in data if r["deviceNum"] == d and r["algorithm"] == "HiveMind"), 0) for d in devs]
    ep_tp = [next((r["throughput"] for r in data if r["deviceNum"] == d and r["algorithm"] == "EdgePipe"), 0) for d in devs]

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    ax.plot(devs, hm_tp, "o-", color=HM_COLOR, label="HiveMind", markersize=7, linewidth=2.2)
    ax.plot(devs, ep_tp, "s-", color=EP_COLOR, label="EdgePipe", markersize=7, linewidth=2.2)
    ax.set_xlabel("Number of devices")
    ax.set_ylabel("Throughput (batches/s)")
    ax.set_title("Throughput vs device count (Vgg19, baseline)")
    ax.set_xticks(devs)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.4)
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
    print_analysis(rows)


if __name__ == "__main__":
    main()
