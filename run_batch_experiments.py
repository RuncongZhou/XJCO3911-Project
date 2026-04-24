"""
Batch experiment script for thesis comparison.
Runs HiveMind vs EdgePipe under various configs and exports results to CSV.

Usage: python run_batch_experiments.py

Runs directly (no server needed). Results saved to experiments_results.csv

Experiments include: (1) four models baseline, (2) AlexNet device count,
(3) AlexNet bandwidth sweep, (4) AlexNet performance sweep, (5) Vgg19 device count,
(6) AlexNet 3×3 bandwidth×performance grid corners for cross-sensitivity / heatmaps.
Then run: python analyze_and_plot.py
"""
import csv
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from app import (
    _get_engine_classes,
    _inject_model_layer_data,
    _inject_device_context,
    _populate_bandwidth_matrix,
)

# Experiment configs: vary one dimension at a time for controlled comparison
EXPERIMENT_CONFIGS = [
    # Exp 1: Different models (fixed: 5 devices, bw 21-31, perf 41-60)
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "Vgg19", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "YOLONet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "SqueezeNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    # Exp 2: Different device counts (fixed: AlexNet, bw 21-31, perf 41-60)
    {"model": "AlexNet", "deviceNum": 3, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 4, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 6, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    # Exp 3: Different bandwidth ranges (fixed: AlexNet, 5 devices, perf 41-60)
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [10, 20], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [40, 60], "performanceRange": [41, 60]},
    # Exp 4: Different performance ranges (fixed: AlexNet, 5 devices, bw 21-31)
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [20, 40]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [60, 100]},
    # Exp 5: Vgg19 — device count sweep (same topology as AlexNet Exp 2, cross-model comparison)
    {"model": "Vgg19", "deviceNum": 3, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "Vgg19", "deviceNum": 4, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "Vgg19", "deviceNum": 5, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    {"model": "Vgg19", "deviceNum": 6, "bandwidthRange": [21, 31], "performanceRange": [41, 60]},
    # Exp 6: AlexNet — complete 3×3 bandwidth × performance grid (corners missing from Exp 3+4)
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [10, 20], "performanceRange": [20, 40]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [10, 20], "performanceRange": [60, 100]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [40, 60], "performanceRange": [20, 40]},
    {"model": "AlexNet", "deviceNum": 5, "bandwidthRange": [40, 60], "performanceRange": [60, 100]},
]


def run_single_compare(config: dict) -> dict:
    """Run compare logic directly (no HTTP)."""
    model_name = config.get("model", "AlexNet")
    device_count = config.get("deviceNum", 5)
    bw_range = config.get("bandwidthRange", [21, 31])
    perf_range = config.get("performanceRange", [41, 60])
    device_order = list(range(device_count))

    np.random.seed(1)
    results = {}
    for algo in ["HiveMind", "EdgePipe"]:
        EngineClass = _get_engine_classes(model_name, algo)
        if EngineClass is None:
            continue
        engine = EngineClass(device_count)
        engine.assignment()
        _inject_model_layer_data(engine, model_name, algo)
        _populate_bandwidth_matrix(engine, device_count, bw_range[0], bw_range[1])
        engine.fn = np.linspace(perf_range[0], perf_range[1], num=device_count)
        _inject_device_context(device_order, device_count, model_name, algo, bw_range=bw_range)
        if algo == "HiveMind":
            tp, ti = engine.enhancedDijkstraTime()
        else:
            tp, ti = engine.dynamic_planning()
        results[algo] = {
            "throughput": float(tp) if tp != float("inf") else 0,
            "inferenceTime": float(ti) if ti != float("inf") else None,
        }
    return {"success": True, "results": results}


def run_batch():
    """Run all experiments and save to CSV."""
    rows = []
    for i, config in enumerate(EXPERIMENT_CONFIGS):
        try:
            print(f"[{i+1}/{len(EXPERIMENT_CONFIGS)}] model={config['model']}, devices={config['deviceNum']}, "
                  f"bw={config['bandwidthRange']}, perf={config['performanceRange']} ... ", end="")
            data = run_single_compare(config)
            if not data.get("success") or "results" not in data:
                print("FAILED (no results)")
                continue
            for algo, res in data["results"].items():
                rows.append({
                    "model": config["model"],
                    "deviceNum": config["deviceNum"],
                    "bandwidthMin": config["bandwidthRange"][0],
                    "bandwidthMax": config["bandwidthRange"][1],
                    "perfMin": config["performanceRange"][0],
                    "perfMax": config["performanceRange"][1],
                    "algorithm": algo,
                    "throughput": res.get("throughput"),
                    "inferenceTime": res.get("inferenceTime"),
                })
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    if not rows:
        print("No results to save.")
        return

    out_path = Path(__file__).parent / "experiments_results.csv"
    try:
        f = open(out_path, "w", newline="", encoding="utf-8")
    except OSError:
        out_path = Path(__file__).parent / "experiments_results_new.csv"
        f = open(out_path, "w", newline="", encoding="utf-8")
    with f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    print("Batch experiments (direct run, no server needed)")
    print("Configs:", len(EXPERIMENT_CONFIGS))
    run_batch()
