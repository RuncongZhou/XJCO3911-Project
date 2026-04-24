# 三、毕业设计（论文）已完成的研究部分

## 3.1 Model Layer Profiling and Data Preparation

Unlike video action recognition, which relies on large-scale video datasets, multi-device DNN collaborative inference requires per-layer computational and data-transfer profiles. The primary data source for this study is pre-profiled layer-level statistics extracted from standard DNN architectures. Each supported model (AlexNet, Vgg19, YOLONet, SqueezeNet) is represented by a CSV file containing, for every layer: floating-point operations (Flops), intermediate data size (DataSize in bytes), input/output tensor shapes, and parameter count. These profiles are obtained through standard profiling tools and stored in a structured format that the partition algorithms consume directly.

To ensure consistency across models with varying layer counts and formats, a preprocessing pipeline was implemented in the backend. The pipeline handles numeric conversion for fields that may contain locale-specific formatting (e.g., comma-separated thousands), validates layer indices, and injects the profile data into the HiveMind and EdgePipe engines at runtime. This design allows the platform to support models with different architectures—for instance, SqueezeNet with 21 feature layers versus AlexNet with 13—without modifying the core algorithm logic. The layer profiles provide the essential inputs for computing per-stage computation time (Flops / device_GFlops) and transfer time (DataSize / bandwidth), which drive the partition optimization.

**Figure 1.** Schematic of layer-level profiling data flow. Each model CSV supplies Flops and DataSize per layer, which are loaded and injected into the partition engines for simulation.

**[粘贴图片：figures/layer_profiling_flow.png]**

---

## 3.2 Partition Algorithms: HiveMind and EdgePipe

Two representative partition algorithms were implemented and integrated into the platform: HiveMind and EdgePipe. They optimize different objectives and employ distinct algorithmic strategies.

**HiveMind** formulates the partition problem as a shortest-path optimization. It minimizes end-to-end inference latency—the time for a single sample to traverse the pipeline from input to output. The algorithm uses a backward dynamic programming procedure over devices and layer indices, computing the optimal split points that minimize the sum of computation time, transfer time, and downstream cost. The result is a partition that favors low per-sample latency, suitable for latency-sensitive applications such as real-time interactive systems.

**EdgePipe** uses a different dynamic programming formulation aimed at maximizing pipeline throughput—the number of batches completed per second when the pipeline is saturated. It optimizes the bottleneck stage time by balancing computation and communication across devices. The partition produced by EdgePipe typically yields higher throughput than HiveMind when the pipeline is fully utilized, at the cost of potentially higher per-sample latency. This trade-off is central to the experimental comparison.

Both algorithms accept the same inputs: layer Flops, layer DataSize, device performance (GFlops/s), and inter-device bandwidth (MB/s). They output layer partition boundaries (start and end indices per device), which the platform uses to compute throughput and inference time via a pipeline simulation model.

**Figure 2.** Conceptual comparison of HiveMind (latency-minimizing) and EdgePipe (throughput-maximizing) partition strategies. HiveMind optimizes the critical path for a single sample; EdgePipe balances stage times to maximize batch throughput.

**[粘贴图片：figures/algo_comparison_concept.png]**

---

## 3.3 Experimental Design and Batch Evaluation Framework

To systematically compare HiveMind and EdgePipe under controlled conditions, an experimental framework was designed with four varying factors and two fixed metrics. The varying factors are: (1) **Models**: AlexNet, Vgg19, YOLONet, SqueezeNet; (2) **Device count**: 3, 4, 5, 6; (3) **Bandwidth range**: 10–20, 21–31, 40–60 MB/s; (4) **Device performance range**: 20–40, 41–60, 60–100 GFlops/s. The metrics are throughput (batches per second) and inference time (seconds per sample).

Fair comparison is ensured by using the same random seed, device order, and topology for both algorithms in each configuration. A batch experiment script runs all combinations without requiring the web server, writes results to a CSV file, and supports reproducibility. The script invokes the same backend logic used by the interactive platform, ensuring consistency between batch experiments and live simulation.

**Table 2.** Experimental factors and levels used for HiveMind vs EdgePipe comparison. Each row varies one factor while holding others fixed at baseline (5 devices, bandwidth 21–31 MB/s, performance 41–60 GFlops/s).

**[粘贴图片：figures/experimental_design_table.png]**

---

## 3.4 Experimental Results and Analysis

The batch experiments were executed and analyzed using a dedicated plotting script. Four figures summarize the key findings.

**Model comparison (Figure 3).** Under five devices with fixed bandwidth and performance, EdgePipe achieves higher throughput across all four models (AlexNet, Vgg19, YOLONet, SqueezeNet), while HiveMind achieves lower inference latency. This confirms that the two algorithms optimize different objectives: throughput versus latency. There is no single best choice across all scenarios; the selection depends on whether the deployment is throughput-bound or latency-bound.

**Impact of device count (Figure 4).** For AlexNet, throughput does not increase monotonically with more devices. EdgePipe peaks around 3 devices; HiveMind around 4. Beyond these points, adding devices reduces throughput, likely due to increased communication overhead and finer-grained partitioning that introduces more transfer stages. This result highlights the importance of topology configuration in addition to algorithm choice.

**Impact of bandwidth (Figure 5).** Higher bandwidth improves throughput for both algorithms. The trend is clear: better network connectivity helps pipeline performance. When bandwidth is constrained, communication becomes the bottleneck and limits the benefit of distributed computation.

**Impact of device performance (Figure 6).** Higher device performance (GFlops/s) improves throughput for both algorithms. Stronger devices reduce computation time per stage, allowing the pipeline to run faster. The interaction between device performance and bandwidth determines the overall bottleneck.

**Figure 3.** Throughput and inference time by model (5 devices, bandwidth 21–31 MB/s, performance 41–60 GFlops/s). EdgePipe achieves higher throughput; HiveMind achieves lower inference latency. Different objectives: throughput vs. latency.

**[粘贴图片：figures/fig1_model_comparison.png]**

**Figure 4.** Throughput versus device count for AlexNet. EdgePipe outperforms HiveMind at all device counts. Throughput peaks around 3–4 devices; beyond that, communication overhead dominates.

**[粘贴图片：figures/fig2_device_count.png]**

**Figure 5.** Throughput versus bandwidth range for AlexNet (5 devices). Higher bandwidth improves throughput for both algorithms.

**[粘贴图片：figures/fig3_bandwidth.png]**

**Figure 6.** Throughput versus device performance range for AlexNet (5 devices). Higher device performance improves throughput for both algorithms.

**[粘贴图片：figures/fig4_performance.png]**

---

## 3.5 Quantitative Results Summary

This subsection consolidates the main numerical outcomes so that the “results” of the study are explicit in tabular form, not only in figures. **Table 3** reports throughput and inference time for all four models under the baseline configuration (5 devices, bandwidth 21–31 MB/s, performance 41–60 GFlops/s). These values are produced by the same batch pipeline as Figures 3–6; if experiments are re-run, update the table from `experiments_results.csv`.

**Table 3.** Throughput (batches/s) and inference time (s) by model and algorithm under baseline settings.

| Model | Algorithm | Throughput (batches/s) | Inference time (s) |
|-------|-----------|--------------------------|----------------------|
| AlexNet | HiveMind | 38.60 | 0.0463 |
| AlexNet | EdgePipe | 50.88 | 0.0580 |
| Vgg19 | HiveMind | 2.09 | 0.5085 |
| Vgg19 | EdgePipe | 9.80 | 0.7200 |
| YOLONet | HiveMind | 2.24 | 1.0022 |
| YOLONet | EdgePipe | 5.84 | 1.0929 |
| SqueezeNet | HiveMind | 40.74 | 0.0586 |
| SqueezeNet | EdgePipe | 53.70 | 0.0733 |

**Synthesis.** (1) EdgePipe achieves higher throughput than HiveMind on every model; HiveMind achieves lower inference time on every model. (2) The gap is largest in relative throughput on Vgg19 and YOLONet (compute-heavy models), where absolute throughput is lower for both algorithms. (3) For AlexNet with varying device count (see Figure 4), EdgePipe’s throughput peaks at 3 devices (54.39 batches/s in the recorded run); HiveMind peaks at 4 devices (45.61 batches/s); adding more devices can reduce throughput due to communication overhead. (4) Bandwidth and device performance ranges (Figures 5–6) monotonically improve throughput when increased, confirming that network and device capability are limiting factors alongside algorithm choice.

Together, **Table 3** and **Figures 3–6** constitute the empirical **results** of this thesis: they answer *how* HiveMind and EdgePipe behave under the same topology, and *when* each objective (latency vs throughput) is favored.

---

## 3.6 Multi-Device Inference Visualization Platform

To demonstrate the practical applicability of the research and enable interactive exploration of partition configurations, a web-based visualization platform was developed. The platform adopts a frontend-backend separated architecture: the backend is built with Flask and provides REST APIs for model loading, layer retrieval, simulation, device topology configuration, and algorithm comparison; the frontend is built with D3.js and Vis.js, providing interactive visualizations for model structure, device topology, partition scheme, and metrics.

The system provides five main functional views: (1) **Model structure**: hierarchical display of DNN layers with Flops and DataSize; (2) **Device topology**: configurable device count, bandwidth matrix, and performance; (3) **Partition scheme**: visualization of the layer assignment produced by HiveMind or EdgePipe; (4) **Metrics**: throughput and inference time for the selected configuration; (5) **Algorithm comparison**: side-by-side comparison of HiveMind and EdgePipe under the same topology. The user can select model, device count, algorithm, bandwidth range, and performance range, then click Simulate to obtain results. Custom partition input is also supported for manual evaluation.

**Figure 7.** Platform architecture. The frontend (D3.js/Vis.js) sends POST requests to the Flask backend; the backend runs HiveMind or EdgePipe and returns partition, throughput, and inference time. Layer profiles are loaded from CSV files.

**[粘贴图片：figures/platform_architecture.png]**

---

## 图片与表格清单（更新）

| 项目 | 插入位置 | 说明 |
|------|----------|------|
| **Table 3** | 3.5 节 | 量化结果汇总表（与 `experiments_results.csv` 一致时可微调小数位） |

---

## 图片粘贴清单与图注汇总

| 图片文件 | 插入位置 | 图注 |
|----------|----------|------|
| `figures/layer_profiling_flow.png` | 3.1 节 Figure 1 | Figure 1. Schematic of layer-level profiling data flow. Each model CSV supplies Flops and DataSize per layer, which are loaded and injected into the partition engines for simulation. |
| `figures/algo_comparison_concept.png` | 3.2 节 Figure 2 | Figure 2. Conceptual comparison of HiveMind (latency-minimizing) and EdgePipe (throughput-maximizing) partition strategies. HiveMind optimizes the critical path; EdgePipe balances stage times to maximize throughput. |
| `figures/experimental_design_table.png` | 3.3 节 Table 2 下方 | Table 2. Experimental factors and levels used for HiveMind vs EdgePipe comparison. Each row varies one factor while holding others fixed at baseline. |
| `figures/fig1_model_comparison.png` | 3.4 节 Figure 3 | Figure 3. Throughput and inference time by model (5 devices, bandwidth 21–31 MB/s, performance 41–60 GFlops/s). EdgePipe achieves higher throughput; HiveMind achieves lower inference latency. |
| `figures/fig2_device_count.png` | 3.4 节 Figure 4 | Figure 4. Throughput versus device count for AlexNet. EdgePipe outperforms HiveMind at all device counts. Throughput peaks around 3–4 devices; beyond that, communication overhead dominates. |
| `figures/fig3_bandwidth.png` | 3.4 节 Figure 5 | Figure 5. Throughput versus bandwidth range for AlexNet (5 devices). Higher bandwidth improves throughput for both algorithms. |
| `figures/fig4_performance.png` | 3.4 节 Figure 6 | Figure 6. Throughput versus device performance range for AlexNet (5 devices). Higher device performance improves throughput for both algorithms. |
| `figures/platform_architecture.png` | 3.6 节 Figure 7 | Figure 7. Platform architecture. The frontend (D3.js/Vis.js) sends POST requests to the Flask backend; the backend runs HiveMind or EdgePipe and returns partition, throughput, and inference time. Layer profiles are loaded from CSV files. |
| `figures/timeline.png` | **第五部分：工作安排** Figure 8 | Figure 8. Timeline of remaining work and defense preparation (experiments mid–late March; platform refinement late March–early April; thesis drafting early–mid April; revision mid–late April; defense early May). |

---

## 生成图片前的准备

若 `figures/` 目录下缺少部分图片，请先运行：

```bash
# 1. 生成实验数据（若 experiments_results.csv 不存在）
python run_batch_experiments.py

# 2. 生成实验图表 fig1–fig4
python analyze_and_plot.py

# 3. 生成实验设计表（3.3 节）
python figures/generate_experimental_design_table.py

# 4. 生成平台架构图
python figures/generate_platform_diagram.py

# 5. 生成层级数据流图（3.1 节 Figure 1）
python figures/generate_layer_profiling_diagram.py

# 6. 生成算法对比概念图（3.2 节 Figure 2）
python figures/generate_algo_comparison_diagram.py

# 7. 生成时间线图（第五部分：工作安排 Figure 8）
python figures/generate_timeline.py
```
