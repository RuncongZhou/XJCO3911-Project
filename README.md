# Multi-Device DNN Collaborative Inference — Edge Simulation Platform

Individual project (University of Leeds): web platform to simulate HiveMind-style vs EdgePipe-style collaborative DNN inference with shared layer profiles and reproducible batch evaluation.

**Thesis snapshot:** the git tag **`thesis-2025-final`** marks the commit aligned with the submitted dissertation (Chapter 4 batch results and figures). **Reproduction and examiner notes:** see [`docs/ASSESSOR.md`](docs/ASSESSOR.md).

## Quick start

```bash
python -m pip install -r requirements.txt
python app.py
```

Open the URL shown in the terminal (e.g. `http://127.0.0.1:5000`).

**Dependencies:** versions in **`requirements.txt`** are pinned to match the thesis **Appendix B** environment (Flask, NumPy, Pandas, Matplotlib, seaborn, etc.) for reproducible plots and the web stack.

## Batch results (thesis / report alignment)

- **`run_batch_experiments.py`** → writes **`experiments_results.csv`** in the project root (same data used for Chapter 4 tables/figures). The file is **not** git-ignored: a copy can stay in the repo for inspection; to regenerate it from code, run this script. If the primary file cannot be created, the script falls back to **`experiments_results_new.csv`**.
- **`analyze_and_plot.py`** reads the CSV above and writes plots under **`figures/`**; use **`python generate_all_thesis_figures.py`** to refresh assets under **`figures/final_figures/`** (e.g. `fig1_model_comparison.png`, `fig2_device_count.png`, … per chapter draft).
- **Layer profiles:** **`data/AlexNet.csv`**, **`data/vgg19.csv`**, **`data/YOLONet.csv`**, **`data/SqueezeNet.csv`**, and optional `Resnet50.csv` / `Resnet101.csv` (on-disk `vgg19.csv` is lowercase; display name in the app is **Vgg19**).
- A duplicate tree exists at **`code-secV2.0/`** (including `code-secV2.0/data/`) for reference; the runtime paths use **`data/`** at the repository root when you run from this folder.

## Repository layout (high level)

| Area | Role |
|------|------|
| `app.py`, `templates/`, `static/` | Flask API + web UI |
| `HiveMind_*.py`, `EdgePipe_*.py`, `PartEnum_*.py`, `SwarmDP_*.py`, `torch*.py` | Simulation engines and helpers |
| `data/*.csv` | Per-model layer FLOP / tensor size profiles |
| `run_batch_experiments.py` | Non-HTTP batch sweep → CSV |
| `analyze_and_plot.py` | Plots + `simulation_insights.md` |
| `real_deploy/` | Optional real-hardware comparison utilities |

See `README.txt` for internal version notes on simulation parameters.

## License

This project is released under the **MIT License**; see [`LICENSE`](LICENSE).
