# 仿真结果 vs 单机双进程真实验证 — 对比说明

## 1. 为什么不能简单“对齐成一个数”

| 维度 | 主系统仿真（HiveMind / EdgePipe） | 单机双进程真实验证（AlexNet） |
|------|-----------------------------------|------------------------------|
| 模型与切分 | 按算法输出划分层边界；多种模型与设备数 | **固定**在 `features+avgpool` 与 `classifier` 之间切一刀 |
| 时间含义 | 用 CSV 的 Flops/DataSize + 带宽公式算 **吞吐与推理时间** | **真实**前向 + **TCP** 传张量 + 反序列化，含 PyTorch 与系统开销 |
| 是否同一套代码 | 与 `app.py` / 批量实验一致 | 独立脚本 `two_process_alexnet.py`，**不参与** HiveMind/EdgePipe 划分 |

结论：**不宜**把两边的“推理时间”当成同一物理量做 1:1 相等验证；论文里应写成 **互补**：主文用仿真做**算法与配置对比**；真实部署用 **AlexNet 双进程**证明 **划分—传输—再推理** 链路可跑通。

---

## 2. 你可填写的对比表（供论文/答辩）

### 表 A：仿真（来自 `experiments_results.csv`，示例为基线配置）

在毕设源码目录运行：

```bash
python export_results_summary_table.py
```

取 **AlexNet、5 devices、HiveMind 与 EdgePipe** 一行的 `throughput` 与 `inferenceTime`，填入下表。

| 来源 | 场景 | Throughput (batches/s) | Inference time (s) |
|------|------|------------------------|----------------------|
| 仿真 | AlexNet, 5 dev, bw 21–31, perf 41–60, **HiveMind** | （填 CSV） | （填 CSV） |
| 仿真 | 同上，**EdgePipe** | （填 CSV） | （填 CSV） |

### 表 B：真实双进程（自动生成）

在项目根目录执行：

```bash
python real_deploy/run_validation.py
```

运行结束后终端会打印 **汇总表** 与 **结论**（含与单进程整网前向的对比），无需手工抄数。若需手动分两终端运行，仍可自行从 `[server]` / `[client]` 与 `REAL_DEPLOY_TIMING:` 行整理。

**汇总表** 中「合计」为 `connect + recv + classifier`；server 的 part1/send 与 `recv` 在时间上重叠，**勿**再与 `recv` 相加。详见 `real_deploy/README.md`。

同一次运行还会在项目 `figures/` 下生成 **`real_deploy_table.md`** 与多张 **PNG**（与仿真图表同目录），便于答辩与论文插图；若需从已有 JSON 重新出图，执行 `python real_deploy/plot_real_deploy_results.py`。

说明：**仿真**里的 `inferenceTime` 是流水线模型下的标量；**真实**里是分段实测，二者**单位可比性有限**，对比时写“量级/趋势”或“真实链路额外开销”即可。

---

## 3. 论文里推荐写法（一两句）

- 主实验采用基于层画像的 **仿真**，对 HiveMind 与 EdgePipe 在不同拓扑与带宽下进行 **系统对比**。
- 另构建 **AlexNet 双进程** 原型：前半段与后半段在 **独立进程** 中执行，中间经 **本机 TCP** 传输特征张量，终端输出 **validation OK**，用于验证 **真实推理与传输链路**；**不与**仿真数值强行等同，而作为 **实现层面** 的补充。

---

## 4. 若老师坚持要“数字对比”

可做 **同一台机、同一模型、同一输入** 下：

1. **只跑 PyTorch 整网一次**（不分段）→ 记总耗时 T_full。  
2. **跑双进程分段** → 记 T_part1 + T_transfer + T_part2。  

在文中说明：T_part1+T_transfer+T_part2 通常 **≥** T_full（多进程与通信有额外开销），这是预期现象，可写进“讨论”。

（如需，可再加一个小脚本 `real_deploy/benchmark_single_vs_split.py` 做整网 vs 双进程计时；需要的话可再说。）
