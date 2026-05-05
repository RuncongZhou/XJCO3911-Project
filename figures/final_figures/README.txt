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
  - schedule_table.png, timeline.png, figure3.3.png (if present)
  - real_deploy_*.png (optional, if generated under figures/)
  - simulation_insights.md  (auto from analyze_and_plot.py)

Regenerate the full bundle (batch + all diagrams) into this folder:
  python generate_all_thesis_figures.py

Update only batch charts (fig1–fig9 + insights) after code/CSV changes — copies into this folder automatically:
  python analyze_and_plot.py
  (fig10 stays as last full run unless you re-run generate_all or figures/generate_pipeline_stage_fig.py)
