"""
Generate schedule table as PNG image for midterm report.
Run: python generate_schedule_table_image.py
Output: figures/schedule_table.png
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.table import Table

# Table data
headers = ['Phase', 'Time Period', 'Key Tasks', 'Status']
rows = [
    ['Topic Selection & Literature Survey', 'Sep 2025',
     'Finalize research direction; collect related work; preliminary study of HiveMind and EdgePipe', 'Completed'],
    ['Algorithm Implementation & Platform Development', 'Oct 2025 – Jan 2026',
     'Implement HiveMind & EdgePipe; Flask backend; D3.js/Vis.js frontend; support AlexNet, Vgg19, YOLONet, SqueezeNet', 'Completed'],
    ['Experiment Design & Batch Execution', 'Jan – Mar 2026',
     'Define factors (models, device count, bandwidth, performance); batch scripts; HiveMind vs EdgePipe comparison', 'Completed'],
    ['Result Analysis & Figure Generation', 'Mar – Apr 2026',
     'Analyze data; generate figures; summarize conclusions (throughput-bound vs latency-bound)', 'In Progress'],
    ['Thesis Writing & Defense Preparation', 'Apr – May 2026',
     'Complete dissertation draft; finalize related work; prepare defense slides and Q&A', 'Planned'],
]

# Build cell text: headers + rows
cell_text = [headers] + rows

# Create figure
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

# Create table
table = ax.table(cellText=cell_text, loc='center', cellLoc='left',
                 colWidths=[0.18, 0.12, 0.45, 0.12])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 2.2)

# Style: header row
for j in range(4):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Style: data rows - color by status
for i in range(1, 6):
    status = cell_text[i][3]
    if status == 'Completed':
        color = '#C6EFCE'  # light green
    elif status == 'In Progress':
        color = '#FFEB9C'  # light yellow
    else:
        color = '#D9D9D9'  # light grey
    for j in range(4):
        table[(i, j)].set_facecolor(color)

# Save
import os
os.makedirs('figures', exist_ok=True)
out_path = os.path.join('figures', 'schedule_table.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close()
