# Defense script English

Slide 1. I am presenting my project, a web platform to simulate multi-device collaborative DNN inference at the edge and to compare two planning policies, HiveMind and EdgePipe, under the same random seeds, link draws, and per-device compute, using shared CSV layer profiles.

Slide 2. One device often cannot run the full model, and splitting the network across devices adds communication, so we need a reproducible way to see throughput versus latency and how K, bandwidth, and compute change the outcome, without treating every case as a physical hardware stopwatch.

Slide 3. The work sits in the edge inference and partitioning literature, and the thesis contribution is a unified implementation with batch evaluation.

Slide 4. The goal is a fair paired comparison, same model, same K, same order, same bandwidth and performance ranges, same seed, and only the algorithm changes.

Slide 5. The stack is Flask JSON APIs, browser D3 views, two Python engines with injected layer data, and batch scripts that write experiments results CSV and Matplotlib figures.

Slide 6. The workflow is configure, optional topology shuffle, simulate or compare, then metrics and exports, and the Chapter 4 figures come from the same CSV and scripts.

Slide 7. The design is a thin stateless client and server, and every run records seed and ranges for traceability.

Slide 8. Next I will show how the experiment is set up, then the main results, then limits and future work.

Slide 9. Section 3 is results. Unless a figure says otherwise, I use the thesis defaults, typically five devices, 21 to 31 megabytes per second bandwidth, 41 to 60 gigaflops per second per device, and seed one.

Slide 10. I compare HiveMind and EdgePipe on AlexNet, Vgg19, YOLONet, and SqueezeNet, with K from three to six, and low, default, and high bandwidth and performance sweeps. I report throughput and inference time, and optionally utilisation, activation traffic, and an illustrative energy-style proxy. Batch CSV and UI JSON follow the same field definitions as the report.

Slide 11. Figure 1 on five devices shows EdgePipe higher in throughput on all four models and HiveMind lower in inference time on all four, so the pattern is a quality trade-off, not one algorithm winning every metric on the same run.

Slide 12. Figure 2 plots the same runs in throughput versus latency space, the two policies do not sit on one point that is best on both axes, and the scatter shows a Pareto-style trade-off under fixed seeds.

Slide 13. Figure 3 sweeps K with other factors fixed. Throughput is not monotonic in K, for example on AlexNet the peak is not at the largest K, and on Vgg19 the curves differ, but the message is that more devices is not always better because of pipeline and communication.

Slide 14. Figure 4 sweeps inter-device link bandwidth from 10 to 20, then 21 to 31, then 40 to 60 megabytes per second with performance held at 41 to 60. Both methods improve with bandwidth, EdgePipe keeps a throughput lead, and the gap is largest at the highest range.

Slide 15. Figure 5 sweeps nominal gigaflops per second from 20 to 40, 41 to 60, and 60 to 100 with bandwidth fixed at 21 to 31. HiveMind throughput rises with faster devices while EdgePipe is nearly flat in this run, so policy still matters besides raw FLOP per second.

Slide 16. Figure 6 is the ratio of EdgePipe to HiveMind throughput. The gain is not constant by model, and Vgg19 shows the largest multiplicative gain because the same policy gap interacts with a heavier layer profile.

Slide 17. Figure 7 is inference time on AlexNet with five devices. Better links and stronger devices both reduce latency, and HiveMind stays the lower-latency curve on this grid, which matches the throughput story.

Slide 18. Figure 8 is a three by three grid of bandwidth by performance for AlexNet with K equals five, low bandwidth at the bottom row and high at the top. HiveMind shows stronger interaction between the two resources, and EdgePipe is more row-dominated, with larger steps when bandwidth changes.

Slide 19. Figure 9 shows per-stage time from the shared pipeline evaluation after each engine’s final partition, highlights the bottleneck, and end-to-end time follows the slowest stage. The numbers are simulation outputs aligned with the table, not a physical trace.

Slide 20. This section wraps up conclusions and outlook.

Slide 21. The platform and paired evaluation show EdgePipe favouring throughput and HiveMind favouring lower latency depending on model, K, link, and compute, and all results follow the report’s static compute and link assumptions.

Slide 22. The work is an end-to-end path from layer data to engines and UI with fair paired comparison, and next steps are validation on real hardware or traces and richer models if we need them.

Slide 23. The references are listed in the report and I will not read them aloud.

Slide 24. Thank you, I will now run the live demo and then take questions.
