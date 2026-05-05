"""
Generate all thesis figures into figures/final_figures/
Includes: batch plots (fig1–10), design tables, diagrams, Chapter 1 & 2 figures.

Run from project root: python generate_all_thesis_figures.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "figures" / "final_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))


def run_script(script_path):
    try:
        subprocess.run([sys.executable, script_path], check=True, cwd=PROJECT_ROOT)
        return True
    except subprocess.CalledProcessError as e:
        print(f"  Warning: {script_path} failed: {e}")
        return False


def copy_if_exists(src_name, dest_name=None):
    src = PROJECT_ROOT / "figures" / src_name
    dest = OUT_DIR / (dest_name or src_name)
    if src.exists():
        shutil.copy(str(src), str(dest))
        print(f"  Copied: {src_name}")
        return True
    return False


print("=== Generating thesis figures -> figures/final_figures/ ===\n")

print("0. Chapter 1 & 2 figures (pipeline concept, platform Fig 2.1)...")
run_script("figures/generate_pipeline_concept_figure.py")
run_script("figures/generate_figure_2_1.py")
copy_if_exists("pipeline_partition_and_schedule.png")
copy_if_exists("pipeline_partition_and_schedule.svg")
copy_if_exists("figure_2_1_platform_architecture.png")
copy_if_exists("figure_2_1_platform_architecture.svg")

print("0b. Chapter 3 Figure 3.1 (implementation stack)...")
run_script("figures/generate_figure_3_1.py")
copy_if_exists("figure_3_1_implementation_stack.png")
copy_if_exists("figure_3_1_implementation_stack.svg")

print("1. Running batch experiments...")
run_script("run_batch_experiments.py")

print("2. Running analyze_and_plot...")
run_script("analyze_and_plot.py")
for f in sorted((PROJECT_ROOT / "figures").glob("fig*.png")):
    copy_if_exists(f.name)
copy_if_exists("simulation_insights.md")

print("3. Generating experimental design table...")
run_script("figures/generate_experimental_design_table.py")
copy_if_exists("experimental_design_table.png")

print("4. Generating platform architecture (legacy midterm diagram)...")
run_script("figures/generate_platform_diagram.py")
copy_if_exists("platform_architecture.png")

print("5. Generating layer profiling diagram...")
run_script("figures/generate_layer_profiling_diagram.py")
copy_if_exists("layer_profiling_flow.png")

print("6. Generating algo comparison diagram...")
run_script("figures/generate_algo_comparison_diagram.py")
copy_if_exists("algo_comparison_concept.png")

print("7. Generating pipeline stage / bottleneck figure (fig10)...")
run_script("figures/generate_pipeline_stage_fig.py")
copy_if_exists("fig10_pipeline_stage_times.png")

print("8. Extra figures in figures/ (schedule, timeline, optional)...")
for extra in (
    "schedule_table.png",
    "timeline.png",
    "figure3.3.png",
    "real_deploy_pipeline_stages.png",
    "real_deploy_split_vs_full.png",
    "real_deploy_pipeline_share.png",
    "real_deploy_summary_table.png",
):
    copy_if_exists(extra)

files = sorted(OUT_DIR.glob("*.png"))
print(f"\n=== Done. {len(files)} PNG files in: {OUT_DIR} ===")
for f in files:
    print(f"  - {f.name}")
