"""
Thesis Figure 2.1 — platform architecture (Chapter 2).
Run: python figures/generate_figure_2_1.py

Outputs:
  figure_2_1_platform_architecture.png  (300 dpi, Word)
  figure_2_1_platform_architecture.svg  (vector)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = Path(__file__).resolve().parent
OUT = FIG_DIR / "figure_2_1_platform_architecture"
DPI = 300


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    c_browser = "#e3f2fd"
    c_flask = "#bbdefb"
    c_engine = "#c8e6c9"
    c_data = "#f5f5f5"
    edge = "#37474f"

    def box(x, y, w, h, fc, title, lines, title_size=11, line_size=8.5):
        p = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06",
            facecolor=fc, edgecolor=edge, linewidth=1.2,
        )
        ax.add_patch(p)
        ty = y + h - 0.38
        ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=title_size, color="#111")
        ly = ty - 0.42
        for line in lines:
            ax.text(x + w / 2, ly, line, ha="center", va="top", fontsize=line_size, color="#333")
            ly -= 0.34

    # Row 1: Browser
    box(0.35, 3.35, 2.85, 2.35, c_browser, "Browser (client)", [
        "Configuration: model, devices, ranges",
        "Topology & partition views",
        "Metrics & algorithm comparison",
    ])

    # Flask
    box(3.55, 3.35, 2.9, 2.35, c_flask, "Flask server (REST / JSON)", [
        "/simulate  /compare",
        "/device-topology  /model/.../layers",
    ])

    # Engines
    box(6.85, 3.35, 2.8, 2.35, c_engine, "Simulation engines", [
        "HiveMind-style placement",
        "EdgePipe-style planning",
        "Injected layer tensors & bandwidth context",
    ])

    # Data
    box(1.8, 0.45, 6.4, 1.55, c_data, "Layer profiles (CSV)", [
        "Per-layer Flops, DataSize — AlexNet, VGG19, YOLO, SqueezeNet",
    ])

    # Arrows browser -> flask
    ax.add_patch(FancyArrowPatch((3.22, 4.45), (3.52, 4.45), arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=edge))
    ax.text(3.37, 4.62, "HTTPS JSON", ha="center", fontsize=7.5, color="#555")

    # flask -> engines
    ax.add_patch(FancyArrowPatch((6.48, 4.45), (6.82, 4.45), arrowstyle="-|>", mutation_scale=14, linewidth=1.8, color=edge))

    # data -> flask & engines (split)
    ax.add_patch(FancyArrowPatch((4.2, 2.02), (4.5, 3.32), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#666", connectionstyle="arc3,rad=0.05"))
    ax.add_patch(FancyArrowPatch((5.8, 2.02), (7.5, 3.32), arrowstyle="-|>", mutation_scale=12, linewidth=1.4, color="#666", connectionstyle="arc3,rad=-0.08"))

    # Title only inside figure (short); full "Figure 2.1" caption goes in Word below the image.
    ax.text(5, 6.05, "Platform architecture (simulation)", ha="center", fontsize=10.5, color="#333")

    plt.tight_layout()
    plt.savefig(f"{OUT}.png", dpi=DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT}.svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved:\n  {OUT}.png\n  {OUT}.svg")


if __name__ == "__main__":
    main()
