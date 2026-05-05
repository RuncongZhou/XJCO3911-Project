# Assessor and examiner guide

Short orientation for **code review, reproduction, and alignment with the thesis (Chapter 4)**. This is the author-facing companion to the [README](../README.md).

## 1. What to run first

From the **repository root** (same directory as `app.py`):

```bash
python -m pip install -r requirements.txt
python app.py
```

The terminal prints a local URL (e.g. `http://127.0.0.1:5000`). Open it in a browser: the web UI drives HiveMind- vs EdgePipe-style simulations and exports batch-style metrics consistent with the batch pipeline.

**No server required for the full numerical sweep** used in Chapter 4: use the batch path below.

## 2. Reproducing batch results and `experiments_results.csv`

1. **Generate or refresh the main results file**

   ```bash
   python run_batch_experiments.py
   ```

   By default this writes **`experiments_results.csv`** in the project root. If the primary path is unavailable, the script may write **`experiments_results_new.csv`** instead (the analysis script accepts either).

2. **Regenerate analysis plots**

   ```bash
   python analyze_and_plot.py
   ```

   This reads the CSV, writes **PNG figures under `figures/`** (and `simulation_insights.md`), and uses the same HiveMind/EdgePipe colour theme as the thesis figures.

3. **One command for all thesis figure assets**

   ```bash
   python generate_all_thesis_figures.py
   ```

   This runs the batch script, `analyze_and_plot.py`, and several generators under `figures/`, then copies the outputs into **`figures/final_figures/`** (e.g. `fig1`–`fig10` style plots, pipeline/architecture/diagram images as produced by the repo).

**Thesis link:** tables and discussion in **Chapter 4** refer to the batch sweep encoded in the CSV and the corresponding plots; running the two-step path (or the single aggregate script) reproduces that material from source.

## 3. Data and layer profiles

- **Layer FLOP / tensor profiles:** `data/AlexNet.csv`, `data/vgg19.csv`, `data/YOLONet.csv`, `data/SqueezeNet.csv`, and optional ResNet CSVs. On disk, `vgg19.csv` is lowercase; the UI label may show **Vgg19**.
- A parallel tree **`code-secV2.0/`** is reference material; with default layout, the running code uses **`data/` at the repository root** when you launch scripts from the root folder.

## 4. Version snapshot (thesis tag)

A **git tag** **`thesis-2025-final`** points at the commit intended to match the submitted thesis and frozen figures. For an exact file-level match, check out that tag; pinned Python dependency versions for the write-up are listed in the thesis (Appendix B) and reflected in `requirements.txt` for this repository.

## 5. If something fails

- **Missing Matplotlib:** `pip install -r requirements.txt` (thesis plotting stack is listed there).
- **Empty or missing CSV:** run `run_batch_experiments.py` first, then `analyze_and_plot.py`.
- **Path errors:** run all commands from the project root (where `app.py` lives), not from `figures/` or `code-secV2.0/`.
