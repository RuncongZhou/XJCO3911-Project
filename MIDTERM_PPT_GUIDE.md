# 中期答辩 PPT 设计指南：System Design & Progress & Results

---

## 一、System Design（建议 4～5 页）

### 1. 总体架构（一页）

**建议放：** 一张**系统架构示意图**（可手绘或 PPT 画）

- 三层：用户层（浏览器）、服务层（Flask 后端）、数据与算法层（CSV 模型数据 + 分区/流水线算法）。
- 箭头：用户操作 → 前端请求 → 后端 API → 算法/数据 → 返回结果 → 前端展示。

**页上文字建议：**

- Title: System Architecture
- 要点：Web-based visualization platform; Frontend (config + visualization) and Backend (REST API + simulation); Layer metadata and partitioning algorithms (HiveMind, EdgePipe) as core components.

---

### 2. 功能模块划分（一页）

**建议放：** **无截图**，用 PPT 的框图列出 4～5 个模块即可。

**页上文字建议：**

- Title: Main Functional Modules
- 模块一：Model and layer management — load DNN models (e.g. AlexNet, Vgg19, YOLONet), display layer hierarchy and attributes (FLOPs, data size, type).
- 模块二：Device and topology configuration — set device count, bandwidth and performance range, generate device topology and order.
- 模块三：Partition design and simulation — algorithm-based partition (HiveMind / EdgePipe) or user-defined layer ranges; pipeline execution simulation and throughput/latency computation.
- 模块四：Visualization and comparison — model structure, device topology, partition scheme, performance metrics; side-by-side algorithm comparison.

---

### 3. 后端 API 与流程（一页）

**建议放：** 一张**流程图**：用户点击「执行模拟」→ 前端发 POST /api/simulate → 后端选算法、跑仿真 → 返回 partition + throughput + inferenceTime → 前端刷新展示。

**页上文字建议：**

- Title: Backend API and Simulation Flow
- 要点：REST API: /api/model/<name>/layers (layer metadata), /api/simulate (partition + metrics), /api/device-topology (topology), /api/compare (HiveMind vs EdgePipe). When user clicks Simulate, backend runs selected algorithm or evaluates custom partition and returns throughput and inference time.

---

### 4. 分区与流水线模型（一页，可选）

**建议放：** 简单示意图：一条链状 DNN（若干层），在层与层之间画 2～3 条竖线表示“切分到不同设备”，并标出“Device 1 / 2 / 3”和“pipeline stages”。

**页上文字建议：**

- Title: Partition and Pipeline Execution Model
- 要点：DNN layers are partitioned into segments; each segment is assigned to one device; execution follows a pipeline so that different devices process different layers in parallel; throughput is limited by the slowest stage (computation or communication).

---

## 二、Progress & Results（建议 4～5 页）

### 1. 平台主界面总览（必放一张截图）

**截图内容：** 浏览器**整页**：左侧配置面板（模型、设备数、算法、带宽/性能、设备顺序、分割方案设计、执行模拟/算法对比按钮）+ 右侧当前选中的主 Tab（建议先选「模型结构」或「分割方案」）。

**页上文字建议：**

- Title: Visualization Platform — Main Interface
- 要点：Left panel: model selection, device count, algorithm (HiveMind/EdgePipe), bandwidth and performance range, device order, optional custom partition. Right panel: model structure, device topology, partition scheme, performance metrics, and algorithm comparison tabs.

---

### 2. 模型层次结构展示（必放一张截图）

**截图内容：** 选择「模型结构」Tab，展示某一模型（如 AlexNet 或 Vgg19）的**层结构可视化**（D3 或列表形式的层次图）。

**页上文字建议：**

- Title: DNN Model Hierarchy
- 要点：Platform loads layer metadata (name, type, FLOPs, data size) and visualizes the layer hierarchy; supports multiple models (AlexNet, Vgg19, YOLONet) for comparative experiments.

---

### 3. 设备拓扑与分割方案（一张或两张截图）

**截图一：** 「设备拓扑」Tab — 节点为设备、边为链路的拓扑图（若当前实现是列表/表格也可截）。

**截图二：** 「分割方案」Tab — 展示某次模拟后的**分割结果**（每设备负责的起止层或类似展示）。

**页上文字建议：**

- Title: Device Topology and Partition Scheme
- 要点：Device topology is generated from configured count and bandwidth/performance; partition scheme shows which layers are assigned to which device after simulation (algorithm or custom); this supports understanding of load distribution across edge devices.

---

### 4. 性能指标与算法对比（必放一张截图）

**截图内容：** 「性能指标」Tab 中**吞吐量、推理时间、各设备起止层**等；或「算法对比」Tab 中 **HiveMind vs EdgePipe** 的对比表格/结果。

**页上文字建议：**

- Title: Performance Metrics and Algorithm Comparison
- 要点：After simulation, the platform displays throughput (e.g. samples per second), single-sample inference time, and per-device layer ranges; algorithm comparison runs both HiveMind and EdgePipe under the same configuration and compares throughput and latency to illustrate the impact of partitioning strategy.

---

### 5. 已完成工作小结（可无截图）

**建议放：** 纯文字或简短列表（不用复杂符号，可成段叙述）。

**页上文字建议：**

- Title: Completed Work Summary
- 内容：Backend: REST API and pipeline simulation with HiveMind and EdgePipe; support for custom partition input and evaluation. Frontend: configuration panel and five visualization tabs. Data: layer metadata (CSV) for multiple models. Additional scripts: PartEnum and SwarmDP for offline experiments. Current platform can demonstrate multi-device partition, manual partition design, and algorithm comparison as required by the thesis objectives.

---

## 三、截图操作建议

1. 先运行平台（python app.py），浏览器打开 http://127.0.0.1:5000 。
2. 按上述说明依次：选好模型、设备数、算法，点一次「执行模拟」或「算法对比」，再切换 Tab 截对应界面。
3. 截图尽量**全屏或主内容区域**，避免多余窗口；分辨率适中，保证投影可读。
4. 若某 Tab 无图只有表格/文字，也截一张，用于说明“有该功能”。

---

## 四、两部分的衔接话术建议

- 讲完 System Design 后：So much for the design. Next I will show the current progress and some results on the platform.
- 讲 Progress & Results 时：We have implemented the modules just described. Here are some screenshots: the main interface, the model hierarchy, the partition and topology, and the metrics and comparison. This demonstrates that the main requirements — multi-device simulation, manual partition design, and performance visualization — are already in place.

按上述结构即可把 System Design 和 Progress & Results 讲清楚，并让评委看到真实界面与结果。
