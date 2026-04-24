"""
Export baseline model comparison (Table 3 / thesis results summary) from experiments_results.csv.
Run: python export_results_summary_table.py
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "experiments_results.csv"
if not CSV_PATH.exists():
    print("Run run_batch_experiments.py first.")
    raise SystemExit(1)

rows = []
with open(CSV_PATH, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)


def baseline_row(model, algo):
    for r in rows:
        if (
            r["model"] == model
            and r["algorithm"] == algo
            and int(r["deviceNum"]) == 5
            and int(r["bandwidthMin"]) == 21
            and int(r["perfMin"]) == 41
        ):
            return r
    return None


models = ["AlexNet", "Vgg19", "YOLONet", "SqueezeNet"]
print("| Model | Algorithm | Throughput (batches/s) | Inference time (s) |")
print("|-------|-----------|------------------------|---------------------|")
for m in models:
    for algo in ["HiveMind", "EdgePipe"]:
        r = baseline_row(m, algo)
        if not r:
            continue
        tp = float(r["throughput"])
        ti = float(r["inferenceTime"] or 0)
        print(f"| {m} | {algo} | {tp:.2f} | {ti:.5f} |")
