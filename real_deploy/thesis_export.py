"""
Export thesis-friendly assets from real-deploy JSON payload (same output folder as simulation figures).

Outputs (under project figures/):
  - real_deploy_last.json          (copy of payload)
  - real_deploy_table.md           Markdown tables for Word/PDF paste
  - real_deploy_pipeline_stages.png  Bar chart: stage latency (ms)
  - real_deploy_split_vs_full.png    Bar chart: two-process vs single forward
  - real_deploy_pipeline_share.png   Pie chart: share of connect/recv/classifier
  - real_deploy_summary_table.png    Table figure (matplotlib.table)

Usage:
  python -m real_deploy.plot_real_deploy_results [path/to.json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = PROJECT_ROOT / "figures" / "real_deploy_last.json"


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def save_payload_json(payload: dict[str, Any], path: Path) -> None:
    _ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown_tables(payload: dict[str, Any], path: Path) -> None:
    d = payload["derived"]
    baseline = payload["baseline"]
    server = payload["server"]
    lines: list[str] = []

    lines.append("# 真实双进程 AlexNet 实验结果（论文/答辩用表）\n")
    lines.append(f"*生成时间：{payload.get('generated_at_utc', '')}*\n")

    lines.append("## 表 1 主链路耗时（可相加）\n")
    lines.append("| 阶段 | 耗时 (ms) | 说明 |")
    lines.append("|------|-----------|------|")
    lines.append(f"| TCP 连接 | {d['connect_ms']:.2f} | 客户端建连 |")
    lines.append(f"| 接收张量 | {d['recv_ms']:.2f} | 含对端 part1+发送、TCP、torch.load |")
    lines.append(f"| Classifier | {d['classifier_ms']:.2f} | 后半段分类器前向 |")
    lines.append(f"| **合计** | **{d['total_pipeline_ms']:.2f}** | connect + recv + classifier |")
    lines.append("")

    lines.append("## 表 2 Server 侧分项（与「接收」重叠，勿与表 1 重复相加）\n")
    lines.append("| 阶段 | 耗时 (ms) | 说明 |")
    lines.append("|------|-----------|------|")
    lines.append(f"| Part1 计算 | {d['part1_ms']:.2f} | features+avgpool+flatten |")
    lines.append(f"| 序列化+发送 | {d['send_ms']:.2f} | torch.save + sendall |")
    lines.append("")

    lines.append("## 表 3 与单进程整网对比\n")
    lines.append("| 项目 | 数值 (ms) | 说明 |")
    lines.append("|------|-----------|------|")
    lines.append(
        f"| 整网一次前向 (×{baseline['runs']}) | {d['full_mean_ms']:.2f} ± {d['full_stdev_ms']:.2f} | 单进程基准 |"
    )
    lines.append(f"| 双进程合计 | {d['total_pipeline_ms']:.2f} | 表 1 合计 |")
    lines.append(f"| 差值 | {d['overhead_ms']:+.2f} | 合计 − 整网均值 |")
    lines.append("")

    lines.append("## 结论要点\n")
    lines.append(
        f"1. 端到端合计约 **{d['total_pipeline_ms']:.2f} ms**；接收张量约占 **{d['pct_recv']:.1f}%**。\n"
    )
    lines.append(
        f"2. 同机单进程整网前向平均 **{d['full_mean_ms']:.2f} ms**；差值 **{d['overhead_ms']:+.2f} ms**。\n"
    )
    lines.append(
        f"3. 中间张量形状（server）：{server.get('tensor_shape', '—')}。\n"
    )

    _ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _try_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        return None


def _setup_fonts(plt) -> None:
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "SimSun",
        "Microsoft JhengHei",
        "DejaVu Sans",
    ]


def export_figures(payload: dict[str, Any], fig_dir: Path) -> list[Path]:
    """Write PNG figures; returns list of paths written (empty if matplotlib missing)."""
    plt = _try_matplotlib()
    if plt is None:
        return []

    _setup_fonts(plt)
    import numpy as np

    _ensure_dir(fig_dir)
    d = payload["derived"]
    written: list[Path] = []

    # --- 1) Horizontal bar: stages ---
    fig1, ax1 = plt.subplots(figsize=(9, 4.2))
    labels = ["TCP connect", "Recv tensor", "Classifier"]
    vals = [d["connect_ms"], d["recv_ms"], d["classifier_ms"]]
    y = np.arange(len(labels))
    ax1.barh(y, vals, color=["#4C72B0", "#55A868", "#C44E52"])
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("Time (ms)")
    ax1.set_title("Real AlexNet: two-process pipeline — stage latency")
    for i, v in enumerate(vals):
        ax1.text(v + max(vals) * 0.02, i, f"{v:.2f}", va="center", fontsize=10)
    fig1.tight_layout()
    p1 = fig_dir / "real_deploy_pipeline_stages.png"
    fig1.savefig(p1, dpi=160)
    plt.close(fig1)
    written.append(p1)

    # --- 2) Two bars: split vs full ---
    fig2, ax2 = plt.subplots(figsize=(6, 4.5))
    names = ["Two-process\n(total)", "Single-process\nfull forward"]
    x = [0, 1]
    heights = [d["total_pipeline_ms"], d["full_mean_ms"]]
    colors = ["#8172B3", "#CCB974"]
    bars = ax2.bar(x, heights, color=colors, width=0.55)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names)
    ax2.set_ylabel("Time (ms)")
    ax2.set_title("Real AlexNet: two-process vs single-process forward")
    for b, h in zip(bars, heights):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            h + max(heights) * 0.02,
            f"{h:.2f}",
            ha="center",
            fontsize=10,
        )
    fig2.tight_layout()
    p2 = fig_dir / "real_deploy_split_vs_full.png"
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    written.append(p2)

    # --- 3) Pie: share of pipeline (sum) ---
    fig3, ax3 = plt.subplots(figsize=(5.5, 5))
    sizes = [d["connect_ms"], d["recv_ms"], d["classifier_ms"]]
    pie_labels = ["TCP connect", "Recv tensor", "Classifier"]
    if sum(sizes) > 0:
        ax3.pie(
            sizes,
            labels=pie_labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=["#4C72B0", "#55A868", "#C44E52"],
        )
        ax3.set_title("Share of end-to-end pipeline time")
    fig3.tight_layout()
    p3 = fig_dir / "real_deploy_pipeline_share.png"
    fig3.savefig(p3, dpi=160)
    plt.close(fig3)
    written.append(p3)

    # --- 4) Table as image ---
    fig4, ax4 = plt.subplots(figsize=(10, 3.2))
    ax4.axis("off")
    table_data = [
        ["Stage", "Time (ms)", "Note"],
        ["TCP connect", f"{d['connect_ms']:.2f}", "Client side"],
        ["Recv tensor", f"{d['recv_ms']:.2f}", "Incl. wait + transfer + load"],
        ["Classifier", f"{d['classifier_ms']:.2f}", "Client side"],
        ["Total (sum)", f"{d['total_pipeline_ms']:.2f}", "connect + recv + classifier"],
        [
            "Full forward (mean)",
            f"{d['full_mean_ms']:.2f} ± {d['full_stdev_ms']:.2f}",
            f"Single process, n={payload['baseline']['runs']}",
        ],
        ["Delta (total − full)", f"{d['overhead_ms']:+.2f}", "Overhead vs single-process"],
    ]
    tbl = ax4.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
    )
    tbl.scale(1.05, 1.8)
    ax4.set_title("Real AlexNet two-process summary (for slides/thesis)", pad=12)
    fig4.tight_layout()
    p4 = fig_dir / "real_deploy_summary_table.png"
    fig4.savefig(p4, dpi=160)
    plt.close(fig4)
    written.append(p4)

    return written


def export_thesis_assets(
    payload: dict[str, Any],
    project_root: Path,
) -> dict[str, Any]:
    """Save JSON, markdown, and figures under figures/."""
    fig_dir = project_root / "figures"
    json_path = fig_dir / "real_deploy_last.json"
    md_path = fig_dir / "real_deploy_table.md"

    save_payload_json(payload, json_path)
    write_markdown_tables(payload, md_path)

    fig_paths = export_figures(payload, fig_dir)
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "figures": [str(p) for p in fig_paths],
    }


def load_payload(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate real-deploy figures/tables from JSON payload.")
    parser.add_argument(
        "json_path",
        nargs="?",
        default=str(DEFAULT_JSON),
        help=f"Path to payload JSON (default: {DEFAULT_JSON})",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root (figures/ output directory)",
    )
    args = parser.parse_args(argv)

    p = Path(args.json_path)
    if not p.exists():
        print(f"File not found: {p}", file=sys.stderr)
        print("Run: python real_deploy/run_validation.py", file=sys.stderr)
        return 1

    payload = load_payload(p)
    fig_dir = args.root / "figures"
    save_payload_json(payload, fig_dir / "real_deploy_last.json")
    write_markdown_tables(payload, fig_dir / "real_deploy_table.md")
    paths = export_figures(payload, fig_dir)
    if not paths:
        print("matplotlib not installed; only JSON and Markdown were written.")
        print("Install: pip install matplotlib")
    else:
        print("Wrote:")
        for x in [fig_dir / "real_deploy_last.json", fig_dir / "real_deploy_table.md", *paths]:
            print(f"  {x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
