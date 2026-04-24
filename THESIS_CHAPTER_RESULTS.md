# Thesis: Experimental Results (Chapter for Dissertation)

Use this as **Chapter X** (e.g., “Experimental Results” or “Results and Evaluation”) in your English thesis. Adjust chapter number to match your school template.

---

## X. Experimental Results

This chapter presents the outcomes of the batch evaluation comparing HiveMind and EdgePipe under controlled conditions. Section X.1 summarizes the experimental setup; Section X.2 reports quantitative results by model; Section X.3 analyzes sensitivity to device count, bandwidth, and device performance; Section X.4 synthesizes findings and discusses implications for edge deployment.

### X.1 Setup and Metrics

All experiments use the same random seed, device order, and bandwidth matrix for both algorithms. Metrics are **throughput** (batches per second, higher is better) and **inference time** (seconds per sample for one forward pass through the pipeline, lower is better). The baseline configuration for cross-model comparison is: five devices, bandwidth range 21–31 MB/s, device performance range 41–60 GFlops/s.

### X.2 Cross-Model Comparison

**Table X.1** lists throughput and inference time for each model under the baseline configuration. EdgePipe achieves higher throughput than HiveMind on all four models; HiveMind achieves lower inference time on all four models. The magnitude of the gap varies with model depth and layer profiles: on Vgg19 and YOLONet, throughput is lower for both algorithms (heavier per-layer computation), while AlexNet and SqueezeNet show higher throughput values.

**Table X.1.** Throughput and inference time under baseline settings (5 devices, bandwidth 21–31 MB/s, performance 41–60 GFlops/s). Values are from the simulation pipeline; minor rounding may apply.

| Model (baseline) | Algorithm | Throughput (batches/s) | Inference time (s) |
|------------------|-----------|------------------------|---------------------|
| AlexNet | HiveMind | 38.60 | 0.04625 |
| AlexNet | EdgePipe | 50.88 | 0.05799 |
| Vgg19 | HiveMind | 2.09 | 0.50850 |
| Vgg19 | EdgePipe | 9.80 | 0.71999 |
| YOLONet | HiveMind | 2.24 | 1.00224 |
| YOLONet | EdgePipe | 5.84 | 1.09285 |
| SqueezeNet | HiveMind | 40.74 | 0.05862 |
| SqueezeNet | EdgePipe | 53.70 | 0.07330 |

**Interpretation.** The results support a clear **throughput–latency trade-off**: EdgePipe optimizes the bottleneck stage and thus yields higher throughput when the pipeline is saturated; HiveMind minimizes end-to-end path cost for a single sample and thus yields lower inference time. There is no single “best” algorithm across both metrics; the choice depends on whether the deployment is **throughput-bound** (e.g., batch analytics) or **latency-bound** (e.g., interactive applications).

### X.3 Sensitivity Analysis

**Device count (AlexNet, fixed bandwidth and performance).** Throughput does not increase monotonically with the number of devices. In the recorded run, EdgePipe achieves the highest throughput at **three** devices (54.39 batches/s); HiveMind peaks at **four** devices (45.61 batches/s). At six devices, throughput drops for both algorithms (HiveMind 36.84, EdgePipe 43.86), indicating that **communication overhead** and finer stage splitting can outweigh the benefit of additional parallelism.

**Bandwidth (AlexNet, five devices).** For bandwidth ranges 10–20, 21–31, and 40–60 MB/s, both algorithms show higher throughput as bandwidth increases. At 40–60 MB/s, EdgePipe reaches 92.98 batches/s and HiveMind 59.02 batches/s, versus 50.88 and 38.60 at 21–31 MB/s. This confirms that **network quality** is a first-order driver of pipeline performance.

**Device performance (AlexNet, five devices).** Raising the performance range from 20–40 to 41–60 to 60–100 GFlops/s improves throughput for both algorithms. HiveMind’s inference time decreases from about 0.064 s to 0.046 s to 0.041 s across these ranges, showing that **stronger edge devices** reduce compute-bound stage time.

### X.4 Summary and Discussion

The experiments lead to three main conclusions:

1. **Objective alignment.** HiveMind is preferable when minimizing end-to-end latency per sample; EdgePipe is preferable when maximizing sustained throughput under pipeline parallelism.
2. **Configuration matters.** Device count, bandwidth, and performance ranges jointly determine whether the system is compute-bound or communication-bound; optimal device count is not always the maximum available.
3. **Simulation scope.** Results are obtained from the **layer-profile-based simulator** embedded in the platform; they are consistent with the interactive UI but do not include real hardware jitter or OS-level scheduling. Real deployment may introduce additional variance; this is acknowledged as a limitation in Chapter Y (limitations / future work).

**Figures.** The corresponding figures (throughput and inference time by model; throughput vs device count, bandwidth, and performance) should be placed in this chapter as **Figure X.1–X.4** (or renumbered to match your thesis figure list), with captions consistent with the midterm report.

---

## 与中期报告 / 第三部分的对应关系

| 论文章节 | 中期报告 | 说明 |
|----------|----------|------|
| X.1 Setup | 3.3 Experimental Design | 实验设计 |
| X.2–X.3 + 图表 | 3.4 Experimental Results | 图表与分项分析 |
| **Table X.1** | 新增 | **结果量化汇总表**（导师建议的“结果部分”核心） |
| X.4 Summary | 可单独成节 | 结论与讨论，与第四、五章衔接 |

---

## 使用说明

1. 将 **Table X.1** 复制到 Word，把 “X” 改成论文实际章号（如 5 或 6）。
2. 若重新运行 `run_batch_experiments.py`，数值可能变化，请用 `analyze_and_plot.py` 打印的 `ANALYSIS SUMMARY` 或 CSV 更新表格。
3. 图仍放在本章，与 **Table X.1** 和 **X.4** 一起构成完整的“结果章节”。
