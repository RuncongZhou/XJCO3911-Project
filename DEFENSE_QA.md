# 答辩常见问题与参考答案（中英文）

---

## Q1: What is the main difference between HiveMind and EdgePipe? Why did you choose to compare these two?

**English:**  
HiveMind uses a shortest-path formulation to minimize inference latency. It optimizes for the time from input to output of a single sample. EdgePipe uses dynamic programming to maximize throughput, i.e., the number of batches processed per second. They optimize different objectives: latency versus throughput. I chose them because they represent two typical optimization goals in edge inference, and the comparison helps users understand the trade-off when designing real systems.

**中文：**  
HiveMind 采用最短路径 formulation 来最小化推理延迟，优化的是单样本从输入到输出的时间。EdgePipe 用动态规划来最大化吞吐量，即每秒处理的批次数。两者优化目标不同：延迟 vs 吞吐量。选择它们是因为它们代表了边缘推理中两种典型的优化目标，对比有助于用户在设计实际系统时理解这种权衡。

---

## Q2: Why does throughput not always increase when you add more devices?

**English:**  
When we add more devices, we split the model into more stages. Each stage needs to transfer intermediate data to the next device. More devices mean more communication hops and more data transfers. If the network bandwidth is limited or the communication overhead grows faster than the computation savings, throughput can drop. Our experiments show EdgePipe peaks around 3 devices and HiveMind around 4; beyond that, communication overhead dominates.

**中文：**  
增加设备时，模型被分成更多阶段，每个阶段都要把中间数据传给下一台设备。设备越多，通信跳数越多，传输量越大。如果网络带宽有限，或通信开销增长快于计算节省，吞吐量反而会下降。我们的实验显示 EdgePipe 在约 3 台设备时达到峰值，HiveMind 约 4 台，再增加设备时通信开销占主导。

---

## Q3: What is the main contribution of your thesis?

**English:**  
The main contribution is a visualization platform that supports multi-device DNN collaborative inference. It allows users to load different models, configure device topology and bandwidth, run HiveMind or EdgePipe or custom partitions, and compare results. We also provide experimental comparison under varying models, device count, bandwidth, and device performance, and summarize when to choose which algorithm based on throughput-bound or latency-bound scenarios.

**中文：**  
主要贡献是一个支持多设备 DNN 协同推理的可视化平台。用户可以加载不同模型、配置设备拓扑和带宽、运行 HiveMind、EdgePipe 或自定义划分，并比较结果。我们还提供了在不同模型、设备数、带宽和设备性能下的实验对比，并总结了在吞吐量受限或延迟受限场景下如何选择算法。

---

## Q4: How do you ensure fairness when comparing HiveMind and EdgePipe?

**English:**  
We use the same topology, same random seed, same device order, and same bandwidth matrix for both algorithms. The only variable is the partition strategy produced by each algorithm. This way, any difference in throughput or latency comes from the algorithm itself, not from different experimental setups.

**中文：**  
我们对两种算法使用相同的拓扑、相同的随机种子、相同的设备顺序和相同的带宽矩阵。唯一的变量是各算法产生的划分策略。这样，吞吐量或延迟的差异都来自算法本身，而不是实验设置不同。

---

## Q5: What are the limitations of your current work?

**English:**  
First, we use simulation rather than real deployment; actual runtime may differ due to network jitter and hardware variance. Second, we assume a linear pipeline topology; more complex topologies like trees or graphs are not yet supported. Third, device ordering is currently fixed or random; joint optimization of partition and device ordering is planned as future work. Fourth, we have not yet done ablation study or scalability tests on larger models.

**中文：**  
第一，我们使用仿真而非真实部署，实际运行可能因网络抖动和硬件差异而不同。第二，我们假设线性流水线拓扑，更复杂的拓扑如树或图尚未支持。第三，设备顺序目前是固定或随机的，划分与设备顺序的联合优化计划作为后续工作。第四，尚未对更大模型做消融实验或可扩展性测试。

---

## Q6: What is the difference between latency and throughput? Why does EdgePipe have higher throughput but HiveMind has lower latency?

**English:**  
Latency is the time for one sample to go through the pipeline from input to output. Throughput is how many samples complete per second when the pipeline is full. EdgePipe optimizes the bottleneck stage time, so when the pipeline is saturated, more samples finish per second. HiveMind optimizes the end-to-end path for a single sample, so one sample finishes faster. A pipeline optimized for throughput may have longer per-sample latency because it balances stage times differently.

**中文：**  
延迟是单个样本从输入到输出经过流水线的时间。吞吐量是流水线满载时每秒完成的样本数。EdgePipe 优化瓶颈阶段时间，所以流水线饱和时每秒完成更多样本。HiveMind 优化单样本的端到端路径，所以单个样本更快完成。为吞吐量优化的流水线可能单样本延迟更长，因为它以不同方式平衡各阶段时间。

---

## Q7: How does your platform obtain layer-level data (Flops, DataSize) for different models?

**English:**  
We use pre-computed CSV files for each model. The CSV contains per-layer Flops, DataSize, input/output shape, and parameters. These profiles are loaded by the backend and injected into the HiveMind or EdgePipe engine. The platform supports AlexNet, Vgg19, YOLONet, SqueezeNet, and can be extended by adding new CSV files and algorithm modules.

**中文：**  
我们为每个模型使用预计算的 CSV 文件。CSV 包含每层的 Flops、DataSize、输入输出形状和参数量。这些 profile 由后端加载并注入 HiveMind 或 EdgePipe 引擎。平台支持 AlexNet、Vgg19、YOLONet、SqueezeNet，可通过添加新 CSV 和算法模块扩展。

---

## Q8: What future work do you plan?

**English:**  
For algorithms, we plan joint optimization of partition and device ordering, and hybrid strategies that combine HiveMind and EdgePipe. For evaluation, we need ablation study and scalability tests. For the platform, we want batch export and communication-overhead visualization. For the thesis, we need to complete related work, refine methodology, and finalize discussion.

**中文：**  
算法方面，计划做划分与设备顺序的联合优化，以及结合 HiveMind 与 EdgePipe 的混合策略。评估方面，需要消融实验和可扩展性测试。平台方面，希望增加批量导出和通信开销可视化。论文方面，需要完善相关工作、细化方法论并定稿讨论部分。

---

## Q9: In which scenario would you choose HiveMind, and when would you choose EdgePipe?

**English:**  
Choose HiveMind when the system is latency-bound: for example, real-time interactive applications, autonomous driving, or any case where the user cares about how fast one request completes. Choose EdgePipe when the system is throughput-bound: for example, batch processing, video analytics, or when you need to process as many samples as possible per second.

**中文：**  
当系统是延迟受限时选 HiveMind：例如实时交互应用、自动驾驶，或任何用户关心单次请求完成速度的场景。当系统是吞吐量受限时选 EdgePipe：例如批处理、视频分析，或需要每秒处理尽可能多样本的场景。

---

## Q10: Does your platform support real DNN inference, or is it simulation only?

**English:**  
It is simulation only. We use pre-profiled layer Flops and DataSize to compute computation time and transfer time. The platform does not run actual PyTorch or TensorFlow inference. This design allows fast exploration of different configurations without deploying to real hardware. Real deployment validation is planned as future work.

**中文：**  
目前仅为仿真。我们使用预分析的层 Flops 和 DataSize 计算计算时间和传输时间。平台不运行实际的 PyTorch 或 TensorFlow 推理。这种设计可以在不部署到真实硬件的情况下快速探索不同配置。真实部署验证计划作为后续工作。
