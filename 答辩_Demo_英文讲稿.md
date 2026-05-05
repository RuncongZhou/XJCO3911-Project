# 答辩 Demo 英文演讲稿（约 3 到 4 分钟）

每段上面是英文演讲词，下面一行是中文，写这一步要点哪里、指哪里，照着练。

---

**开场（约十秒）**

This is the multi-device DNN collaborative inference platform in the browser. The back end is Flask, layer profiles come from CSV, and the stack supports both HiveMind and EdgePipe. Parameters sit on the left, the center has five tabs from model structure to compare, and the log is on the right.

开页后手从左到右划一下左栏、中间 Tab 条、右栏 Log。不必多窗。

**实验参数（约半分钟）**

I pick AlexNet, five devices, and HiveMind. I press Thesis baseline so the bandwidth range is 21 to 31 MB/s, the performance range is 41 to 60 GFlop/s, and the random seed is one. That matches the factor setting in the main batch table rows. I leave link efficiency at a hundred per cent for this run. In partition design, the form supports a custom layer range for each device if I want a manual plan instead of the policy; I keep that off in this pass.

左栏 Model 选 AlexNet，Devices 5，Algorithm HiveMind。点 Thesis baseline 按钮。看 Random seed 为 1。此时带宽 21 到 31 MB/s、算力 41 到 60 GFlop/s。Link efficiency 满格 100% 不拖。Partition design 可勾选 Use custom partition 在各行填每设备起止层号、覆盖算法切分。本段演示不勾，用算法自动分区。手指可在 Partition design 标题或那一块停半秒，不必展开填数。

**拓扑（约半分钟）**

I generate topology to sample the link graph, then I shuffle device order once so the pipeline order changes for the next run. I switch to the topology view and use one link label to read off bandwidth in MB/s. The list on the left is the order the chain will use.

点 Generate topology。在 Device order 点 Shuffle。点上方 Tab 里的 Topology。圆环上指一条边，读边上 MB。左侧设备顺序指一下，说明后面链顺序。

**模型结构（约二十秒）**

I open the model structure tab, pause on the layer graph, and then move on. Nothing to change on this pass.

点 Tab Model structure，在层图上看一下就行。不额外动别的，后面照流程点。

**仿真、分区与指标（约一分多钟）**

I run the simulation with HiveMind. The engine uses the device order I set earlier, including the shuffle in the topology step. Then I open the partition view to show the layer to device layout the engine just decided, which is the automatic plan under custom partition off. After that I open metrics. The values are from the same linear simulation model as in the thesis, not real hardware stopwatch time. I walk through throughput, inference time, utilization, comm volume in MB, and the energy proxy, then the per device GFlop bars, the bandwidth matrix, and the pipeline stage times to see the bottleneck. I export simulation json and open the file to show the structured result in one place, for example the config echo, partition fields, and the metrics you would align with a table or a figure.

左栏点 Run simulation。等跑完。可随口带一句管线顺序就是前面 shuffle 过的。点 Tab Partition，看每设备上层的分块。说一句这是没开自定义、算法给的分法。再点 Tab Metrics。从上往下指五个数字块。再指 Device performance 条、Bandwidth matrix、Pipeline stage time。在 Metrics 页点 Export simulation JSON。下载后本地用记事本或 VS 打开，对着屏幕过一眼字段。

**算法对比（约半分钟）**

I run compare algorithms, open the compare view for HiveMind and EdgePipe on the same settings, and read the chart and the table. I export compare json and open it to show how both algorithms sit in one object next to the evaluation chapter. That is enough to close the demo and return to the slides.

左栏点 Compare algorithms。等完。点 Tab Compare。看图和表。点 Export compare JSON。打开导出的文件举例。关浏览器或回 PPT。
