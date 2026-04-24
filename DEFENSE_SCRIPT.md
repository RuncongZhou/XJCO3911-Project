# Defense Script (~8 min: ~2 min video + ~6 min speech)

**按幻灯片顺序撰写，视频在第三部分中间（Slide 12）**

---

## Slide 1 — Title (~15 s)

Good morning/afternoon. I am Runcong Zhou. My thesis is on multi-device DNN collaborative inference at the edge. I will present background, system design, results, and remaining work.

---

## Slide 2 — Contents (~5 s)

The presentation has four parts: research background, system design, progress and results, and remaining work.

---

## Slide 3–5 — Part 1: Research Background & Objectives (~45 s)

**Slide 3:** Research background and objectives.

**Slide 4:** Single devices have limited resources. Pipeline inference across multiple devices can improve throughput. This diagram shows devices processing layers in parallel.

**Slide 5:** Four objectives: DNN visualization, device configuration, partition simulation, and performance evaluation. We compare HiveMind and EdgePipe.

---

## Slide 6 — Part 2: System Design (section) (~5 s)

Next, system design and methodology.

---

## Slide 7 — Main Function Modules (~25 s)

The platform has four main modules. The first manages models and layers. The second configures devices and topology. The third handles partition and simulation, supporting HiveMind, EdgePipe, or custom partitions. The fourth provides visualization and comparison.

---

## Slide 8 — Backend API & Simulation Flow (~25 s)

When the user clicks Simulate, the frontend sends a POST request to the backend. The backend runs the selected algorithm or evaluates the custom partition, then returns partition, throughput, and inference time. The main APIs are /layers, /simulate, /device-topology, and /compare.

---

## Slide 9 — Part 3: Progress & Preliminary Results (section) (~5 s)

Now I will show the experiment design, a short platform demo, and the experimental results.

---

## Slide 10 — Experiment Design (~35 s)

We compare HiveMind and EdgePipe under the same topology. HiveMind minimizes inference latency while EdgePipe maximizes throughput. We vary four factors: models, device count from 3 to 6, bandwidth from 10 to 60 MB/s, and device performance from 20 to 100 GFlops/s. The metrics are throughput and inference time.

---

## Slide 11 — Platform Overview (~20 s)

On the left you see model, device count, algorithm, bandwidth, performance, and optional custom partition. On the right there are five tabs. You run Simulate, then switch tabs to view results. Here is a short video demo.

---

## Slide 12 — Video Presentation (~2 min)

**[播放视频]**

That was the platform demo. Next I present the experimental results.

---

## Slide 13 — Model Comparison (Figure 1) (~40 s)

Figure 1 shows throughput and inference time across four models with five devices. EdgePipe achieves higher throughput while HiveMind achieves lower latency. So the two algorithms optimize different objectives.

---

## Slide 14 — Impact of Device Count (Figure 2) (~35 s)

Figure 2 shows throughput versus device count for AlexNet. EdgePipe outperforms HiveMind at all device counts. Throughput does not increase monotonically with more devices. EdgePipe peaks around 3 devices and HiveMind around 4. Beyond that, more devices reduce throughput, likely due to communication overhead.

---

## Slide 15 — Impact of Bandwidth (Figure 3) (~25 s)

Figure 3 shows the impact of bandwidth. Higher bandwidth improves throughput for both algorithms.

---

## Slide 16 — Impact of Device Performance (Figure 4) (~25 s)

Figure 4 shows the impact of device performance. Higher device performance improves throughput for both algorithms.

---

## Slide 17 — Conclusions (~45 s)

Three main points. First, the two algorithms optimize different objectives, throughput versus latency, so there is no single best choice across all scenarios. Second, configuration matters: performance depends on topology, bandwidth, and device count, not only on the algorithm. Third, we should choose the algorithm and configuration based on whether the system is throughput-bound or latency-bound.

---

## Slide 18 — Part 4: Remaining Work and Timeline (section) (~5 s)

Finally, remaining work and timeline.

---

## Slide 19 — Remaining Work (~30 s)

Remaining work includes several aspects. For the algorithm, we plan joint optimization of partition and device ordering, and hybrid strategies. For evaluation, we need ablation study and scalability tests. For the platform, we want batch export and communication-overhead visualization. For the thesis, we need to complete related work, refine methodology, and finalize discussion.

---

## Slide 20 — Timeline (~25 s)

The timeline is as follows. Experiments in mid- to late March, platform refinement in late March to early April, thesis drafting in early to mid-April, revision in mid- to late April, and defense preparation in late April to early May. Defense is in early May.

---

## Slide 21–22 — Reference & Thanks (~15 s)

These are the references. That is my presentation. Thank you for listening. I am happy to answer questions.

---

## Time Summary

| Slides | Content | Duration |
|--------|---------|----------|
| 1–2 | Title, Contents | ~20 s |
| 3–5 | Part 1: Background | ~45 s |
| 6–8 | Part 2: System Design | ~55 s |
| 9–11 | Part 3 intro, Experiment Design, Platform | ~65 s |
| 12 | Video | ~2 min |
| 13–17 | Part 3: Fig 1–4, Conclusions | ~2.5 min |
| 18–20 | Part 4: Remaining Work, Timeline | ~60 s |
| 21–22 | Reference, Thanks | ~15 s |
| **Total** | | **~8 min** |
