"""
Generate platform architecture diagram for midterm report.
Run: python figures/generate_platform_diagram.py
Output: figures/platform_architecture.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
c_front = '#E8F4F8'
c_back = '#B8D4E3'
c_api = '#4A90D9'
c_algo = '#50C878'

# Frontend box
front = FancyBboxPatch((0.5, 3), 3.5, 2.2, boxstyle="round,pad=0.05",
                        facecolor=c_front, edgecolor='#2E86AB', linewidth=1.5)
ax.add_patch(front)
ax.text(2.25, 4.6, 'Frontend (D3.js / Vis.js)', ha='center', fontsize=11, fontweight='bold')
ax.text(2.25, 4.1, 'Model structure', ha='center', fontsize=9)
ax.text(2.25, 3.7, 'Device topology', ha='center', fontsize=9)
ax.text(2.25, 3.3, 'Partition scheme', ha='center', fontsize=9)
ax.text(2.25, 2.9, 'Metrics & comparison', ha='center', fontsize=9)

# Backend box
back = FancyBboxPatch((5.5, 3), 3.5, 2.2, boxstyle="round,pad=0.05",
                      facecolor=c_back, edgecolor='#2E86AB', linewidth=1.5)
ax.add_patch(back)
ax.text(7.25, 4.6, 'Backend (Flask)', ha='center', fontsize=11, fontweight='bold')
ax.text(7.25, 4.1, '/layers, /simulate', ha='center', fontsize=9)
ax.text(7.25, 3.7, '/device-topology', ha='center', fontsize=9)
ax.text(7.25, 3.3, '/compare', ha='center', fontsize=9)
ax.text(7.25, 2.9, 'HiveMind / EdgePipe', ha='center', fontsize=9)

# Arrow: Frontend -> Backend
ax.annotate('', xy=(5.3, 4.1), xytext=(4, 4.1),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(4.65, 4.35, 'POST', ha='center', fontsize=8)

# Arrow: Backend -> Frontend
ax.annotate('', xy=(4, 3.9), xytext=(5.3, 3.9),
            arrowprops=dict(arrowstyle='->', color='#333', lw=2))
ax.text(4.65, 3.55, 'JSON', ha='center', fontsize=8)

# Data layer
data = FancyBboxPatch((2, 0.3), 6, 1.4, boxstyle="round,pad=0.05",
                      facecolor='#F5F5F5', edgecolor='#999', linewidth=1)
ax.add_patch(data)
ax.text(5, 1.2, 'Layer profiles (CSV): Flops, DataSize per layer', ha='center', fontsize=10)
ax.text(5, 0.7, 'AlexNet | Vgg19 | YOLONet | SqueezeNet', ha='center', fontsize=9)

# Arrows from data to backend
ax.annotate('', xy=(5.5, 2.8), xytext=(5, 1.7),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5, connectionstyle='arc3,rad=0.1'))
ax.annotate('', xy=(6.5, 2.8), xytext=(6, 1.7),
            arrowprops=dict(arrowstyle='->', color='#666', lw=1.5, connectionstyle='arc3,rad=-0.1'))

ax.set_title('Multi-Device DNN Collaborative Inference Visualization Platform Architecture', fontsize=12, fontweight='bold')
plt.tight_layout()
out = Path(__file__).parent / 'platform_architecture.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
