# 实验对比 PPT 截图与文案指南

## 一、需要粘贴的截图

### 1. 平台 Compare 对比表
- **截图内容**：打开平台 → 配置模型/设备数/带宽 → 点击 Compare algorithms → Compare 标签中的对比表格
- **建议**：选 AlexNet + 5 设备，表格清晰展示 HiveMind 与 EdgePipe 的 Throughput、Inference time

### 2. 模型对比柱状图（fig1_model_comparison.png）
- **位置**：`figures/fig1_model_comparison.png`
- **内容**：4 个模型（AlexNet、Vgg19、YOLONet、SqueezeNet）下 HiveMind 与 EdgePipe 的吞吐量、推理时间柱状图

### 3. 设备数影响折线图（fig2_device_count.png）
- **位置**：`figures/fig2_device_count.png`
- **内容**：AlexNet 下，设备数 3/4/5/6 与吞吐量的关系

### 4. 带宽影响柱状图（fig3_bandwidth.png）
- **位置**：`figures/fig3_bandwidth.png`
- **内容**：不同带宽范围（10-20、21-31、40-60 MB/s）下的吞吐量对比

### 5. 设备性能影响柱状图（fig4_performance.png）
- **位置**：`figures/fig4_performance.png`
- **内容**：不同设备算力范围（20-40、41-60、60-100 GFlops/s）下的吞吐量对比

---

## 二、PPT 文字内容（简明）

### 实验设计
- 对比算法：HiveMind（最短路径） vs EdgePipe（流水线 DP）
- 变量：模型、设备数、带宽、设备算力
- 指标：吞吐量（batches/s）、推理时延（s）

### 主要结论
- **吞吐量**：EdgePipe 在多数配置下更高
- **推理时延**：HiveMind 通常更低
- **带宽**：带宽越高，两种算法吞吐量均提升
- **设备数**：设备数增加，吞吐量先升后趋于平稳

### 单页示例文案（可复制）

**实验设置**
- 模型：AlexNet / Vgg19 / YOLONet / SqueezeNet
- 设备数：3–6，带宽 21–31 MB/s，算力 41–60 GFlops/s

**结论**
- EdgePipe 吞吐量优于 HiveMind
- HiveMind 单样本推理时延更低
- 带宽与算力提升均有利于性能
