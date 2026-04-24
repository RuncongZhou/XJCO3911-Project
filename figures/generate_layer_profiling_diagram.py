"""
Generate layer profiling data flow diagram for midterm report.
Run: python figures/generate_layer_profiling_diagram.py
Output: figures/layer_profiling_flow.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(0, 8)
ax.set_ylim(0, 4)
ax.axis('off')

# CSV box
csv = mpatches.FancyBboxPatch((0.5, 1.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                              facecolor='#E8F4F8', edgecolor='#2E86AB', linewidth=1.5)
ax.add_patch(csv)
ax.text(1.4, 2.4, 'Model CSV', ha='center', fontsize=10, fontweight='bold')
ax.text(1.4, 2.0, 'Flops per layer', ha='center', fontsize=9)
ax.text(1.4, 1.7, 'DataSize per layer', ha='center', fontsize=9)

# Backend box
backend = mpatches.FancyBboxPatch((3.2, 1.5), 1.8, 1.2, boxstyle="round,pad=0.05",
                                  facecolor='#B8D4E3', edgecolor='#2E86AB', linewidth=1.5)
ax.add_patch(backend)
ax.text(4.1, 2.4, 'Backend', ha='center', fontsize=10, fontweight='bold')
ax.text(4.1, 2.0, 'Load & parse', ha='center', fontsize=9)
ax.text(4.1, 1.7, 'Inject to engine', ha='center', fontsize=9)

# Engine boxes
hm = mpatches.FancyBboxPatch((5.8, 2.2), 1.6, 0.8, boxstyle="round,pad=0.03",
                             facecolor='#C6EFCE', edgecolor='#2E86AB', linewidth=1)
ep = mpatches.FancyBboxPatch((5.8, 1.2), 1.6, 0.8, boxstyle="round,pad=0.03",
                             facecolor='#FFEB9C', edgecolor='#2E86AB', linewidth=1)
ax.add_patch(hm)
ax.add_patch(ep)
ax.text(6.6, 2.6, 'HiveMind', ha='center', fontsize=9)
ax.text(6.6, 1.6, 'EdgePipe', ha='center', fontsize=9)

# Arrows
ax.annotate('', xy=(3.1, 2.1), xytext=(2.4, 2.1),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.annotate('', xy=(5.7, 2.4), xytext=(5.1, 2.1),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))
ax.annotate('', xy=(5.7, 1.6), xytext=(5.1, 1.8),
            arrowprops=dict(arrowstyle='->', color='#333', lw=1.5))

ax.set_title('Layer Profiling Data Flow: CSV → Backend → Partition Engines', fontsize=11, fontweight='bold')
plt.tight_layout()
out = Path(__file__).parent / 'layer_profiling_flow.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
