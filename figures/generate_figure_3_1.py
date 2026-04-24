"""
Thesis Figure 3.1 — implementation-oriented component stack (Chapter 3).
Distinction from Figure 2.1: names concrete client libraries and API routes.

Run: python figures/generate_figure_3_1.py
Outputs: figure_3_1_implementation_stack.png, .svg (300 dpi PNG)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent
OUT = FIG_DIR / "figure_3_1_implementation_stack"
DPI = 300


def main():
    fig, ax = plt.subplots(figsize=(11, 6.5), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.8)
    ax.axis("off")
    edge = "#37474f"

    def box(x, y, w, h, fc, title, lines, ts=10.2, ls=7.8):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.05",
                facecolor=fc, edgecolor=edge, linewidth=1.1,
            )
        )
        ty = y + h - 0.32
        ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=ts, color="#111")
        ly = ty - 0.36
        for line in lines:
            ax.text(x + w / 2, ly, line, ha="center", va="top", fontsize=ls, color="#333")
            ly -= 0.28

    # Top: client tech stack
    box(0.2, 4.5, 4.2, 2.0, "#e8eaf6", "Browser client (front end)", [
        "D3.js: model layer hierarchy",
        "vis-network: device graph",
        "Plain HTML/CSS/JS: tabs, forms, bar charts, export",
    ], ts=10.5, ls=7.5)

    box(4.6, 4.5, 5.2, 2.0, "#e3f2fd", "HTTP transport", [
        "fetch() POST, JSON request/response",
        "No server-side session; state in browser",
    ], ts=10.5, ls=7.8)

    # Middle: Flask
    box(0.2, 2.55, 9.6, 1.65, "#bbdefb", "Flask application (app.py)", [
        "Routes:  /  /api/simulate  /api/compare  /api/device-topology  /api/model/<name>/layers",
        "CORS for local dev; JSON error payloads",
    ], ts=10.5, ls=7.5)

    # Bottom row: engines + data
    box(0.2, 0.4, 4.3, 1.9, "#c8e6c9", "Engine modules (Python)", [
        "HiveMind_*: graph placement, enhancedDijkstraTime",
        "EdgePipe_*: Pipeline, dynamic_planning",
        "Injected: band, fn, layer tensors, first_band",
    ], ts=10, ls=7.2)

    box(4.7, 0.4, 5.1, 1.9, "#f5f5f5", "Layer data (project CSV)", [
        "data/<Model>.csv: Flops, DataSize per layer",
        "AlexNet, Vgg19, YOLONet, SqueezeNet, etc.",
    ], ts=10, ls=7.2)

    # Arrows: client -> http (already adjacent), http -> flask implied by position
    ax.add_patch(FancyArrowPatch((2.2, 4.48), (6.0, 4.6), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color="#555"))
    ax.add_patch(FancyArrowPatch((7.0, 4.48), (5.0, 4.2), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#999", linestyle="dashed"))
    # flask down to engines
    ax.add_patch(FancyArrowPatch((2.5, 2.52), (2.4, 2.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=edge))
    ax.add_patch(FancyArrowPatch((6.0, 2.52), (4.0, 2.33), arrowstyle="-|>", mutation_scale=12, linewidth=1.3, color=edge))
    # data up to engines (inject)
    ax.add_patch(FancyArrowPatch((5.0, 0.85), (3.5, 0.4), arrowstyle="-|>", mutation_scale=10, linewidth=1.0, color="#666", connectionstyle="arc3,rad=0.2"))
    ax.text(5, 0.2, "load at simulation time", ha="center", fontsize=6.5, color="#666")

    ax.text(5, 6.5, "Implementation stack (Chapter 3)", ha="center", fontsize=10.2, color="#333")

    plt.tight_layout()
    plt.savefig(f"{OUT}.png", dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT}.svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved:\n  {OUT}.png\n  {OUT}.svg")


if __name__ == "__main__":
    main()
