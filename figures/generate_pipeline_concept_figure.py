"""
Generate Figure: multi-device pipeline + pipelined schedule (thesis-ready).
Outputs: pipeline_partition_and_schedule.svg, pipeline_partition_and_schedule.png (300 DPI)

Run: python figures/generate_pipeline_concept_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrow

FIG_DIR = Path(__file__).resolve().parent
OUT_BASE = FIG_DIR / "pipeline_partition_and_schedule"

# Colours aligned with original sketch
C_DEV1 = "#1a237e"
C_DEV2 = "#827717"
C_DEV3 = "#b71c1c"
C_TASK_LIGHT = "#90caf9"
C_TASK_DEV2 = "#c5e1a5"
C_GRID1 = "#43a047"
C_GRID2 = "#fbc02d"
C_ARROW = "#1565c0"


def draw_rounded_label(ax, xy, text, face, size=9):
    """xy = center in data coordinates 0..1 for our ax"""
    w, h = 0.055, 0.04
    box = FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2),
        w,
        h,
        boxstyle="round,pad=0.008",
        facecolor=face,
        edgecolor="#333",
        linewidth=0.8,
        zorder=5,
    )
    ax.add_patch(box)
    ax.text(
        xy[0],
        xy[1],
        text,
        ha="center",
        va="center",
        fontsize=size,
        fontweight="600",
        color="#111",
        zorder=6,
    )


def main():
    fig = plt.figure(figsize=(10.2, 7.2), facecolor="white")
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.94])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ----- Top: Device chain -----
    y_top = 0.92
    dev_w, dev_h = 0.14, 0.055
    positions = [(0.22, y_top), (0.5, y_top), (0.78, y_top)]
    labels = ["Device 1", "Device 2", "Device 3"]
    colors = [C_DEV1, C_DEV2, C_DEV3]
    text_colors = ["#ffeb3b", "#e8f5e9", "#90caf9"]

    for (cx, cy), lab, fc, tc in zip(positions, labels, colors, text_colors):
        rect = FancyBboxPatch(
            (cx - dev_w / 2, cy - dev_h / 2),
            dev_w,
            dev_h,
            boxstyle="round,pad=0.01",
            facecolor=fc,
            edgecolor="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(cx, cy, lab, ha="center", va="center", fontsize=11, fontweight="700", color=tc, zorder=4)

    for i in range(2):
        x0 = positions[i][0] + dev_w / 2 + 0.01
        x1 = positions[i + 1][0] - dev_w / 2 - 0.01
        arr = FancyArrow(
            x0,
            y_top,
            x1 - x0 - 0.02,
            0,
            width=0.012,
            head_width=0.035,
            head_length=0.025,
            length_includes_head=True,
            facecolor=C_ARROW,
            edgecolor=C_ARROW,
            zorder=2,
        )
        ax.add_patch(arr)

    ax.text(0.5, 0.985, "Logical device chain (data flow)", ha="center", fontsize=10, fontweight="600", color="#333")

    # ----- Middle: Layer / stage groups -----
    ax.text(0.5, 0.855, "Partitioned layers (stages) per device", ha="center", fontsize=10, fontweight="600", color="#333")

    # Outer groups
    grp_y = 0.68
    grp_h = 0.12
    groups = [
        (0.18, 0.26, C_DEV1, False, [(0.10, 1), (0.18, 2), (0.26, 3)]),
        (0.50, 0.20, C_DEV2, True, [(0.44, 4), (0.56, 5)]),
        (0.82, 0.26, C_DEV3, True, [(0.73, 6), (0.81, 7), (0.89, 8)]),
    ]

    for cx, gw, edge_c, dashed, tasks in groups:
        gx0 = cx - gw / 2
        style = "dashed" if dashed else "solid"
        rect = FancyBboxPatch(
            (gx0, grp_y - grp_h / 2),
            gw,
            grp_h,
            boxstyle="round,pad=0.015",
            facecolor="white",
            edgecolor=edge_c,
            linewidth=2.5,
            linestyle=(0, (6, 4)) if dashed else "solid",
            zorder=1,
        )
        ax.add_patch(rect)

    # Tasks and internal arrows
    task_y = grp_y
    # Device 1 tasks
    pts1 = [(0.10, task_y), (0.18, task_y), (0.26, task_y)]
    for i, p in enumerate(pts1):
        draw_rounded_label(ax, p, str(i + 1), C_TASK_LIGHT)
    for i in range(2):
        ax.annotate(
            "",
            xy=(pts1[i + 1][0] - 0.04, task_y),
            xytext=(pts1[i][0] + 0.04, task_y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2, mutation_scale=10),
        )

    # Device 2
    pts2 = [(0.44, task_y), (0.56, task_y)]
    draw_rounded_label(ax, pts2[0], "4", C_TASK_DEV2)
    draw_rounded_label(ax, pts2[1], "5", C_TASK_DEV2)
    ax.annotate(
        "",
        xy=(pts2[1][0] - 0.04, task_y),
        xytext=(pts2[0][0] + 0.04, task_y),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2, mutation_scale=10),
    )

    # Device 3
    pts3 = [(0.73, task_y), (0.81, task_y), (0.89, task_y)]
    for i, p in enumerate(pts3):
        draw_rounded_label(ax, p, str(6 + i), C_TASK_LIGHT)
    for i in range(2):
        ax.annotate(
            "",
            xy=(pts3[i + 1][0] - 0.035, task_y),
            xytext=(pts3[i][0] + 0.035, task_y),
            arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.2, mutation_scale=10),
        )

    # Cross-device arrows 3->4, 5->6
    ax.annotate(
        "",
        xy=(0.40, task_y),
        xytext=(0.28, task_y),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=12),
    )
    ax.annotate(
        "",
        xy=(0.66, task_y),
        xytext=(0.60, task_y),
        arrowprops=dict(arrowstyle="-|>", color=C_ARROW, lw=1.8, mutation_scale=12),
    )

    # ----- Bottom: Gantt-style pipeline -----
    ax.text(0.5, 0.38, "Pipelined execution (two in-flight batches)", ha="center", fontsize=10, fontweight="600", color="#333")

    rows, cols = 5, 8
    gx0, gy0 = 0.12, 0.06
    cell_w = 0.76 / cols
    cell_h = 0.22 / rows

    # Grid lines
    for r in range(rows + 1):
        ax.plot([gx0, gx0 + cols * cell_w], [gy0 + r * cell_h, gy0 + r * cell_h], "k-", linewidth=0.6, zorder=0)
    for c in range(cols + 1):
        ax.plot([gx0 + c * cell_w, gx0 + c * cell_w], [gy0, gy0 + rows * cell_h], "k-", linewidth=0.6, zorder=0)

    # Time axis
    ax.annotate(
        "",
        xy=(gx0 + cols * cell_w + 0.02, gy0 - 0.02),
        xytext=(gx0 - 0.02, gy0 - 0.02),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2, mutation_scale=12),
    )
    ax.text(gx0 + cols * cell_w / 2, gy0 - 0.055, "Time step", ha="center", fontsize=9, color="#333")

    # Row labels (stages)
    for r in range(rows):
        ax.text(
            gx0 - 0.06,
            gy0 + (rows - 0.5 - r) * cell_h,
            f"S{r + 1}",
            ha="right",
            va="center",
            fontsize=8,
            color="#555",
        )

    # Batch 1 diagonal (green "1")
    b1 = [(0, 0), (1, 1), (2, 2), (3, 3)]
    for col, row_from_top in b1:
        r = row_from_top
        cx = gx0 + (col + 0.5) * cell_w
        cy = gy0 + (rows - 1 - r + 0.5) * cell_h
        sq = mpatches.Rectangle(
            (cx - cell_w * 0.38, cy - cell_h * 0.38),
            cell_w * 0.76,
            cell_h * 0.76,
            facecolor=C_GRID1,
            edgecolor="#2e7d32",
            linewidth=1,
            zorder=4,
        )
        ax.add_patch(sq)
        ax.text(cx, cy, "1", ha="center", va="center", fontsize=10, fontweight="700", color="white", zorder=5)

    # Batch 2 diagonal (yellow "2"), offset
    b2 = [(3, 0), (4, 1), (5, 2), (6, 3)]
    for col, row_from_top in b2:
        r = row_from_top
        cx = gx0 + (col + 0.5) * cell_w
        cy = gy0 + (rows - 1 - r + 0.5) * cell_h
        sq = mpatches.Rectangle(
            (cx - cell_w * 0.38, cy - cell_h * 0.38),
            cell_w * 0.76,
            cell_h * 0.76,
            facecolor=C_GRID2,
            edgecolor="#f57f17",
            linewidth=1,
            zorder=4,
        )
        ax.add_patch(sq)
        ax.text(cx, cy, "2", ha="center", va="center", fontsize=10, fontweight="700", color="#333", zorder=5)

    plt.savefig(f"{OUT_BASE}.png", dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.savefig(f"{OUT_BASE}.svg", bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved:\n  {OUT_BASE}.png\n  {OUT_BASE}.svg")


if __name__ == "__main__":
    main()
