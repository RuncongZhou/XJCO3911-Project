"""
Generate Timeline figure for PPT.
Run: python generate_timeline.py
Output: timeline.png
Timeline: Mar 9 (today) -> Defense early May
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

fig, ax = plt.subplots(figsize=(10, 5))

# X-axis: Mid-Mar, Late Mar, Early Apr, Mid-Apr, Late Apr, Early May
labels = ["Mid-Mar", "Late Mar", "Early Apr", "Mid-Apr", "Late Apr", "Early May"]
n = len(labels)
ax.set_xlim(-0.5, n - 0.5)
ax.set_ylim(0, 6)
ax.set_yticks([])

# Task bars (start_idx, end_idx)
tasks = [
    ("Experiments", 0, 1, "#4A90D9"),
    ("Platform refinement", 1, 2, "#50C878"),
    ("Thesis drafting", 2, 4, "#E07C24"),
    ("Thesis revision", 3.5, 4.8, "#F4A460"),
    ("Defense prep", 4.2, 5, "#9B59B6"),
]

for i, (name, start, end, color) in enumerate(tasks):
    y = 5 - i
    bar = mpatches.FancyBboxPatch((start, y - 0.25), end - start, 0.5,
                                   boxstyle="round,pad=0.02", facecolor=color, edgecolor="black", linewidth=0.8)
    ax.add_patch(bar)
    ax.text(-0.55, y, name, va="center", ha="right", fontsize=9)

for i in range(n):
    ax.axvline(x=i, color="gray", linestyle="--", alpha=0.5)
ax.set_xticks(np.arange(n))
ax.set_xticklabels(labels, fontsize=9)
ax.set_xlabel("Time", fontsize=10)
ax.set_title("Timeline (Defense: Early May)", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(Path(__file__).parent / "timeline.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved timeline.png")
