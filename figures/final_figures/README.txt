Thesis final figure bundle (Leeds COMP3931)
==========================================

This folder was renamed from midterm_figures and holds figures for the final report.

Chapter-oriented assets (also copied here):
  - pipeline_partition_and_schedule.png / .svg  ->  Chapter 1 (Figure 1.1, pipeline concept)
  - figure_2_1_platform_architecture.png / .svg  ->  Chapter 2 (Figure 2.1, platform architecture)

Chapter 3:
  - figure_3_1_implementation_stack.png / .svg  (auto: python figures/generate_figure_3_1.py)
  - For Figure 3.2 and 3.3, see CHAPTER3_SCREENSHOTS.txt in figures/ (manual screenshots)

Batch / results bundle (fig1–fig10, tables, diagrams):
  - fig1_model_comparison.png … fig10_pipeline_stage_times.png
  - experimental_design_table.png, platform_architecture.png (older midterm diagram), layer_profiling_flow.png, algo_comparison_concept.png
  - schedule_table.png, timeline.png
  - simulation_insights.md  (auto text summary from analyze_and_plot.py)

Regenerate everything from project root:
  python generate_all_thesis_figures.py
