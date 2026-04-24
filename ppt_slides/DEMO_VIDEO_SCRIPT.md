# Demo Video Guide (~1.5 minutes)

## 1. Before recording

1. Run `python app.py`, open http://127.0.0.1:5000.
2. Close other windows/tabs; recorder ready; optional window size 1280×720.

---

## 2. Full narration script (English)

**0:00–0:06**  
This is the multi-device DNN collaborative inference platform.

**0:06–0:12**  
Model: AlexNet.

**0:12–0:16**  
Five devices.

**0:16–0:22**  
Algorithm: EdgePipe.

**0:22–0:30**  
I click Run simulation. The view jumps to Metrics — throughput, inference time, and device utilization. (The tabs were already there; after run we’re on Metrics.)

**0:30–0:38**  
This tab is model structure — the layer hierarchy.

**0:38–0:46**  
This one is device topology and bandwidth links.

**0:46–0:54**  
Partition scheme: which layers run on which device.

**0:54–1:02**  
I click Compare algorithms. Both HiveMind and EdgePipe run under the same config; the table shows throughput and inference time for each.

**1:10–1:18**  
You can see the comparison in the table.

**1:18–1:26**  
The platform also supports custom partition: tick Use custom partition and set the layer range per device, then run simulation to evaluate.

**1:26–1:30**  
Thank you.

---

## 3. One line per step (quick reference)

| Time   | You say | You do |
|--------|---------|--------|
| 0:00   | Multi-device DNN collaborative inference platform. | Show interface. |
| 0:06   | Model: AlexNet. | Click **Model** → AlexNet. |
| 0:12   | Five devices. | Point at **Devices** (5). |
| 0:16   | Algorithm: EdgePipe. | Click **Algorithm** → EdgePipe. |
| 0:22   | Run simulation. | Click **Run simulation**. Wait — view auto-jumps to **Metrics**. |
| 0:30   | Metrics: throughput, inference time, utilization. | Already on Metrics (no click). |
| 0:38   | Model structure — layer hierarchy. | Click tab **Model structure**. |
| 0:46   | Device topology and links. | Click tab **Topology**. |
| 0:54   | Partition scheme — layers per device. | Click tab **Partition**. |
| 1:02   | Compare algorithms. | Click **Compare algorithms**. Wait. |
| 1:10   | Table: HiveMind and EdgePipe, same config. | Point at table rows/columns. |
| 1:18   | Use custom partition — set layer range per device. | Check **Use custom partition**, point at inputs. |
| 1:26   | Thanks. | — |

---

## 4. Tips

- One short sentence per row; pause while the UI updates.
- If a request is slow, just wait or say “Computing.”
- Keep config panel, content area, and log in frame.

---

## 5. 中文简版口播（备用）

0:00 这是多设备 DNN 协作推理可视化平台。  
0:06 模型选 AlexNet。  
0:12 五台设备。  
0:16 算法选 EdgePipe。  
0:22 点击执行模拟，界面会跳到 Metrics，显示吞吐量、推理时间、利用率。（标签页本来就有，跑完直接到 Metrics。）  
0:30 当前就是性能指标。  
0:38 这是模型结构。  
0:46 这是设备拓扑和链路。  
0:54 这是分割方案。  
1:02 点击算法对比，表格里是 HiveMind 和 EdgePipe 的对比结果。  
1:10 表格里可看两种算法的吞吐量和推理时间。  
1:18 支持自定义分割：勾选后为每台设备设层范围，再执行模拟评估。  
1:26 谢谢。
