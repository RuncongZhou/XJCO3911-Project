"""
Generate HiveMind vs EdgePipe conceptual comparison diagram.
Run: python figures/generate_algo_comparison_diagram.py
Output: figures/algo_comparison_concept.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# HiveMind: minimize latency (critical path)
ax1.set_xlim(0, 6)
ax1.set_ylim(0, 4)
ax1.axis('off')
ax1.set_title('HiveMind: Minimize Latency', fontsize=11, fontweight='bold')
# Pipeline stages
for i, (w, label) in enumerate([(1.2, 'D1'), (1.0, 'D2'), (1.4, 'D3')]):
    rect = mpatches.Rectangle((i*2 + 0.3, 1.5), w, 1, facecolor='#C6EFCE', edgecolor='#2E86AB')
    ax1.add_patch(rect)
    ax1.text(i*2 + 0.3 + w/2, 2, label, ha='center', va='center', fontsize=10)
# Arrow showing critical path
ax1.annotate('', xy=(5.5, 2), xytext=(0.2, 2),
             arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax1.text(2.8, 2.4, 'Critical path', fontsize=9, color='red')
ax1.text(3, 0.8, 'Optimize: single-sample latency', ha='center', fontsize=9)

# EdgePipe: maximize throughput (balance stages)
ax2.set_xlim(0, 6)
ax2.set_ylim(0, 4)
ax2.axis('off')
ax2.set_title('EdgePipe: Maximize Throughput', fontsize=11, fontweight='bold')
# Balanced stages
for i in range(3):
    rect = mpatches.Rectangle((i*2 + 0.3, 1.5), 1.2, 1, facecolor='#FFEB9C', edgecolor='#2E86AB')
    ax2.add_patch(rect)
    ax2.text(i*2 + 0.9, 2, f'D{i+1}', ha='center', va='center', fontsize=10)
ax2.text(3, 0.8, 'Optimize: balance stage times', ha='center', fontsize=9)
ax2.text(3, 0.4, 'Bottleneck = 1/throughput', ha='center', fontsize=8)

plt.suptitle('HiveMind vs EdgePipe: Different Optimization Objectives', fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
out = Path(__file__).parent / 'algo_comparison_concept.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
