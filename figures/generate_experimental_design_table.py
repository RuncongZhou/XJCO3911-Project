"""
Generate experimental design table for midterm report Section 3.3.
Run: python figures/generate_experimental_design_table.py
Output: figures/experimental_design_table.png
"""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Experimental design: factors and levels (Table 2 in midterm report)
headers = ['Factor', 'Levels / Range', 'Baseline (fixed)']
rows = [
    ['Models', 'AlexNet, Vgg19, YOLONet, SqueezeNet', '—'],
    ['Device count', '3, 4, 5, 6', '5'],
    ['Bandwidth (MB/s)', '10–20, 21–31, 40–60', '21–31'],
    ['Device performance (GFlops/s)', '20–40, 41–60, 60–100', '41–60'],
    ['Metrics', 'Throughput (batches/s), Inference time (s)', '—'],
]
cell_text = [headers] + rows

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.axis('off')
table = ax.table(cellText=cell_text, loc='center', cellLoc='left',
                 colWidths=[0.25, 0.45, 0.25])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.0)

# Header row
for j in range(3):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Data rows
for i in range(1, 6):
    for j in range(3):
        table[(i, j)].set_facecolor('#F5F5F5' if i % 2 == 0 else 'white')

out = Path(__file__).parent / 'experimental_design_table.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved {out}')
