"""
Thesis Figure 2.1 — platform architecture (simulation), Chapter 2.
Logical components only: no in-image title; large labels. Contrast: Fig. 3.1 is the tech stack.

Run: python figures/generate_figure_2_1.py
Outputs: figure_2_1_platform_architecture.png, .svg
"""
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent
OUT = FIG_DIR / "figure_2_1_platform_architecture"
DPI = 300
TITLE_PT = 14.5
LINE_PT = 12.0
ARROW_PT = 10.0


def main():
    fig, ax = plt.subplots(figsize=(11, 5.8), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    c_browser = "#e3f2fd"
    c_flask = "#bbdefb"
    c_engine = "#c8e6c9"
    c_data = "#f5f5f5"
    edge = "#37474f"

    def box(x, y, w, h, fc, title, lines, line_sp=0.4):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.08",
                facecolor=fc, edgecolor=edge, linewidth=1.3,
            )
        )
        ty = y + h - 0.4
        ax.text(
            x + w / 2, ty, title, ha="center", va="top",
            fontsize=TITLE_PT, fontweight="600", color="#111",
        )
        ly = ty - 0.44
        for line in lines:
            ax.text(
                x + w / 2, ly, line, ha="center", va="top",
                fontsize=LINE_PT, color="#333",
            )
            ly -= line_sp

    # Conceptual: who does what (no D3, no app.py, no module names)
    box(0.35, 2.6, 2.85, 2.1, c_browser, "Browser (client)", [
        "Configuration: model, devices, ranges",
        "Topology & partition views",
        "Metrics & algorithm comparison",
    ])
    box(3.4, 2.6, 2.9, 2.1, c_flask, "Flask server (REST / JSON)", [
        "/simulate  /compare",
        "/device-topology  /model/.../layers",
    ])
    box(6.5, 2.6, 2.85, 2.1, c_engine, "Simulation engines", [
        "HiveMind-style placement",
        "EdgePipe-style planning",
        "Layer tensors & bandwidth (injected)",
    ])
    box(1.75, 0.35, 6.5, 1.45, c_data, "Layer profiles (CSV)", [
        "Per-layer Flops, DataSize  —  AlexNet, Vgg19, YOLO, SqueezeNet",
    ])

    ax.add_patch(
        FancyArrowPatch((3.22, 3.6), (3.38, 3.6), arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=edge)
    )
    # Two short lines, just under the arrow, inside the white gap (avoids export crop / overlap)
    ax.text(
        3.3, 3.52, "HTTPS\nJSON", ha="center", va="top", fontsize=ARROW_PT, color="#555", linespacing=0.85,
    )
    ax.add_patch(
        FancyArrowPatch((6.32, 3.6), (6.48, 3.6), arrowstyle="-|>", mutation_scale=16, linewidth=1.8, color=edge)
    )
    ax.add_patch(
        FancyArrowPatch((4.1, 1.82), (4.4, 2.58), arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#666", connectionstyle="arc3,rad=0.05")
    )
    ax.add_patch(
        FancyArrowPatch((5.8, 1.82), (7.2, 2.58), arrowstyle="-|>", mutation_scale=14, linewidth=1.4, color="#666", connectionstyle="arc3,rad=-0.08")
    )

    plt.tight_layout()
    pad = 0.12
    plt.savefig(
        f"{OUT}.png", dpi=DPI, bbox_inches="tight", pad_inches=pad, facecolor="white", edgecolor="none"
    )
    plt.savefig(
        f"{OUT}.svg", bbox_inches="tight", pad_inches=pad, facecolor="white", edgecolor="none"
    )
    plt.close()

    final = FIG_DIR / "final_figures" / "figure_2_1_platform_architecture"
    if (FIG_DIR / "final_figures").is_dir():
        for ext in (".png", ".svg"):
            shutil.copy2(f"{OUT}{ext}", f"{final}{ext}")

    print(f"Saved:\n  {OUT}.png\n  {OUT}.svg")
    if (FIG_DIR / "final_figures").is_dir():
        print(f"  (copied to {final}.png / .svg)")


if __name__ == "__main__":
    main()
