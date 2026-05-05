"""
Thesis Figure 3.1 — implementation stack, Chapter 3.
Libraries, app.py, API paths, module names. No in-image title; large type.
Fig. 2.1 is only logical; this figure is implementation detail.

Run: python figures/generate_figure_3_1.py
Outputs: figure_3_1_implementation_stack.png, .svg
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent
OUT = FIG_DIR / "figure_3_1_implementation_stack"
DPI = 300
TITLE_PT = 14.5
LINE_PT = 12.0
MONO_PT = 10.5


def main():
    fig, ax = plt.subplots(figsize=(11, 6.0), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    edge = "#37474f"

    def box(x, y, w, h, fc, title, lines, use_mono=None):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08",
                facecolor=fc, edgecolor=edge, linewidth=1.2,
            )
        )
        ty = y + h - 0.36
        ax.text(
            x + w / 2, ty, title, ha="center", va="top",
            fontsize=TITLE_PT, fontweight="600", color="#111",
        )
        ly = ty - 0.4
        if use_mono is None:
            use_mono = [False] * len(lines)
        for line, um in zip(lines, use_mono):
            ax.text(
                x + w / 2, ly, line, ha="center", va="top",
                fontsize=MONO_PT if um else LINE_PT,
                color="#222",
                family="monospace" if um else "sans-serif",
            )
            ly -= 0.32 if not um else 0.36

    # Stacked layout: top pair, full-width API, bottom pair
    box(0.2, 4.35, 4.1, 1.75, "#e8eaf6", "Browser (front end)", [
        "D3.js: model layer hierarchy",
        "vis-network: device graph",
        "HTML/CSS/JS: tabs, forms, charts, export",
    ])
    box(4.4, 4.35, 5.4, 1.75, "#e3f2fd", "HTTP", [
        "fetch() POST, JSON",
        "No server session; state in browser",
    ])
    box(
        0.2, 2.45, 9.6, 1.7, "#bbdefb", "Flask (app.py)",
        [
            "/  /api/simulate  /api/compare  /api/device-topology  /api/model/<name>/layers",
            "CORS (dev)  ·  JSON error payloads",
        ],
        use_mono=[True, True],
    )
    box(0.2, 0.38, 4.2, 1.75, "#c8e6c9", "Engine modules (Python)", [
        "HiveMind_*: graph placement, enhancedDijkstraTime",
        "EdgePipe_*: dynamic_planning",
        "Inject: band, fn, first_band, tensors",
    ], use_mono=[False, False, False])
    box(4.5, 0.38, 5.3, 1.75, "#eceff1", "Data (project CSV)", [
        "data/<Model>.csv: Flops, DataSize",
        "AlexNet, Vgg19, YOLONet, SqueezeNet",
    ], use_mono=[True, False])

    cy = 4.35 + 1.75 / 2
    ax.add_patch(
        FancyArrowPatch((4.2, cy), (4.42, cy), arrowstyle="-|>", mutation_scale=16, linewidth=1.5, color=edge)
    )
    ax.add_patch(
        FancyArrowPatch((5.0, 2.42), (5.0, 2.15), arrowstyle="-|>", mutation_scale=15, linewidth=1.3, color=edge)
    )
    ax.add_patch(
        FancyArrowPatch((2.2, 2.42), (2.0, 2.1), arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=edge)
    )
    ax.add_patch(
        FancyArrowPatch((5.0, 2.42), (3.5, 2.1), arrowstyle="-|>", mutation_scale=14, linewidth=1.2, color=edge)
    )
    ax.add_patch(
        FancyArrowPatch(
            (5.0, 0.9), (3.4, 0.8), arrowstyle="-|>", mutation_scale=12, linewidth=1.0, color="#666", connectionstyle="arc3,rad=0.1",
        )
    )

    plt.tight_layout()
    plt.savefig(f"{OUT}.png", dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT}.svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()

    final = FIG_DIR / "final_figures" / "figure_3_1_implementation_stack"
    if (FIG_DIR / "final_figures").is_dir():
        for ext in (".png", ".svg"):
            shutil.copy2(f"{OUT}{ext}", f"{final}{ext}")
        print(f"  (copied to {final}.png / .svg)")

    print(f"Saved:\n  {OUT}.png\n  {OUT}.svg")


if __name__ == "__main__":
    main()
