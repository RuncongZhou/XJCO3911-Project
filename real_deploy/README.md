# 单机双进程真实推理验证（AlexNet）

在 **同一台电脑** 上启动两个进程：进程 A 跑 `features + avgpool + flatten`，通过 **本机 TCP** 把中间张量发给进程 B；进程 B 跑 `classifier`。用于与仿真平台对照的**简单真实部署验证**。

## 1. 安装依赖（仅需一次）

在项目根目录 `毕设源码` 下执行（**推荐用当前 Python 解释器安装**，避免装到别的环境）：

```bash
python -m pip install -r real_deploy/requirements.txt
```

若下载慢，可用 CPU 版 PyTorch（体积较小）：

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

首次运行会从网络下载 AlexNet 预训练权重（需联网）。

## 2. 一键运行（推荐）

在项目根目录执行：

```bash
python real_deploy/run_validation.py
```

会先启动 server（等模型加载完成并监听端口），再启动 client。结束后会打印：

- 两段 `[server]` / `[client]` 的原始日志（含一行 `REAL_DEPLOY_TIMING:` JSON，供程序解析）；
- **「真实双进程 AlexNet — 结果汇总表」**：主链路各阶段耗时、server 侧参考分项、**单进程整网前向** 的多次平均基线，以及 **「结论」** 三条（可直接粘贴到实验记录或论文讨论）。

若首次运行需下载 AlexNet 权重，请等待直至出现 `listening`。

**说明：** 汇总表中的「合计」为 `connect + recv + classifier`（`recv` 已含对端 part1+发送与反序列化，勿再与 server 的 part1 相加）。整网基线用于对比多进程与传输带来的额外开销。

### 论文/答辩图表与表格（与仿真 `figures/` 同目录）

成功跑完后会**自动生成**（需已安装 `matplotlib`，与 `analyze_and_plot.py` 相同）：

| 文件 | 用途 |
|------|------|
| `figures/real_deploy_last.json` | 完整数值快照，可存档或复现作图 |
| `figures/real_deploy_table.md` | **Markdown 表格**（表 1–3 + 结论要点），可直接粘贴到 Word / Typora / 转 LaTeX |
| `figures/real_deploy_pipeline_stages.png` | 主链路各阶段耗时（ms）横向柱状图 |
| `figures/real_deploy_split_vs_full.png` | 双进程合计 vs 单进程整网前向对比 |
| `figures/real_deploy_pipeline_share.png` | 端到端时间占比饼图 |
| `figures/real_deploy_summary_table.png` | 英文表格式汇总图（适合幻灯片） |

**若未安装 matplotlib：** 仍会写入 JSON 与 Markdown；安装后可在项目根目录执行：

```bash
python -m pip install matplotlib
python real_deploy/plot_real_deploy_results.py
```

或指定某次保存的 JSON：`python real_deploy/plot_real_deploy_results.py figures/real_deploy_last.json`

## 3. 手动分两终端运行

**终端 1：**

```bash
cd 毕设源码
python -m real_deploy.two_process_alexnet server --port 29500
```

**终端 2（等终端 1 出现 listening 后）：**

```bash
cd 毕设源码
python -m real_deploy.two_process_alexnet client --port 29500
```

## 4. 说明

- 默认 `127.0.0.1:29500`；若端口占用可改 `--port`。
- 输入为固定随机种子生成的 `1×3×224×224` 张量，与 ImageNet 预处理尺寸一致。
- 与毕设主系统的 **HiveMind/EdgePipe 仿真** 独立；论文中可写为「真实推理链路的补充验证」。

## 5. 毕设里怎么表述「单机」与「多设备协同」

- 本验证是 **两个独立进程**（不是同一进程里多线程抢一张卡），中间经 **TCP** 传张量，语义上对应「两台逻辑设备上的两段计算 + 链路传输」，与多设备流水线的抽象一致。
- **物理上只有一台电脑**时，属于在可控条件下的 **概念验证（proof of concept）**：证明「划分—传输—再计算」的真实链路可跑通；**不等价**于测真实 WiFi/多机带宽，但比纯仿真多一步 **真实 PyTorch 推理与序列化/反序列化**。
- 若老师问「为什么不是多台真机」：可答主工作量在 **仿真与划分算法对比**，真实多机部署受设备与环境限制；本实验作为 **补充验证** 即可。
