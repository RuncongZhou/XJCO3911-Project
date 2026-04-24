# 毕业设计整体安排及进度（中期阶段）

## Table 1. Project Schedule and Progress Overview (as of Mid-term Check, March 2026)

| Phase | Time Period | Key Tasks | Status |
|-------|-------------|-----------|--------|
| Topic Selection & Literature Survey | Sep 2025 | Finalize research direction on multi-device DNN collaborative inference; collect and review related work on edge inference, pipeline partitioning, and distributed deep learning; preliminary study of HiveMind and EdgePipe algorithms | Completed |
| Algorithm Implementation & Platform Development | Oct 2025 – Jan 2026 | Implement HiveMind (shortest-path, latency-minimizing) and EdgePipe (DP-based, throughput-maximizing) partition engines; build Flask backend with REST APIs (/layers, /simulate, /compare); develop D3.js/Vis.js frontend for model structure, device topology, partition scheme, and metrics visualization; support AlexNet, Vgg19, YOLONet, SqueezeNet | Completed |
| Experiment Design & Batch Execution | Jan – Mar 2026 | Define experimental factors (models, device count 3–6, bandwidth 10–60 MB/s, device performance 20–100 GFlops/s); implement batch experiment scripts; run HiveMind vs EdgePipe comparison under controlled topology and seed; collect throughput and inference-time metrics | Completed |
| Result Analysis & Figure Generation | Mar – Apr 2026 | Analyze experimental data; generate figures on model comparison, device-count impact, bandwidth impact, and device-performance impact; summarize conclusions and design implications (throughput-bound vs latency-bound scenarios) | In Progress |
| Thesis Writing & Defense Preparation | Apr – May 2026 | Complete dissertation draft; finalize related work and methodology; compile experimental results and discussion; prepare defense slides and Q&A | Planned |

---

**Table 1.** Project schedule and progress overview as of the mid-term stage (March 2026). Status: Completed (phases 1–3), In Progress (phase 4), Planned (phase 5).
