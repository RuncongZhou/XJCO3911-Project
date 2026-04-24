# Graduation Design (Thesis) Progress Report — English (Paragraph Form)

---

## 1. Overview of the Graduation Design (Thesis)

This project is entitled Design and Implementation of a Multi-Device DNN Collaborative Inference System at the Edge. With the continuous growth of deep neural network models, a single edge device often cannot meet the computational and storage requirements for real-time inference. This thesis aims to investigate a multi-device collaborative inference mechanism for edge environments and to develop a visualization platform for demonstration. By simulating multi-device model partitioning and pipeline execution of DNNs in a virtual environment, the platform can intuitively display the hierarchical structure of the models, support users in manually selecting or designing partitioning schemes, and compute and present relevant performance metrics such as throughput and inference latency. This study helps deepen the understanding of DNN collaborative inference mechanisms in edge computing environments and provides an experimental foundation for research on distributed inference.

---

## 2. Overall Arrangement and Schedule

The work is arranged in several phases. In the first phase, literature review and requirement analysis are carried out, including a survey of edge computing, DNN partitioning and pipeline scheduling, and the definition of system requirements and evaluation metrics. In the second phase, the focus is on algorithm design and implementation, including the implementation of partitioning algorithms such as HiveMind and EdgePipe and the pipeline execution simulation logic. In the third phase, the visualization platform is developed, including the backend APIs for simulation, layer metadata and device topology, and the frontend for displaying model structure, device topology, partition scheme and performance metrics. In the fourth phase, experiments and evaluation are conducted with different models, device counts and algorithms, and the results are analysed. In the final phase, the thesis is drafted and revised, and defense materials and a demonstration are prepared.

---

## 3. Completed Research Parts

The completed work includes the following. The system architecture and requirements have been defined, including the functional requirements for multi-device simulation, manual partition design and performance metrics, and the overall design of the platform. The partitioning and pipeline simulation have been implemented and integrated, with support for strategies such as HiveMind and EdgePipe and for computing throughput and single-sample inference time under a given device topology and partition. A web-based visualization platform has been developed, which provides the display of DNN model hierarchy, the visualization of device network topology, the visualization of partition scheme, and the presentation of performance metrics including throughput, latency and per-device load, as well as algorithm comparison under the same configuration. The manual partition design feature has been implemented, allowing users to specify the layer range executed by each device and to evaluate the corresponding performance when custom partition is enabled. Support for multiple models and algorithms has been completed, with several DNN models such as AlexNet, Vgg19 and YOLONet integrated, and multiple partitioning or scheduling algorithms available; in addition, enumeration-based and PSO-based methods have been implemented as standalone scripts for further experimentation and comparison. The data and API design have been completed, with layer metadata prepared in CSV form and REST-style APIs implemented for models, layers, simulation, topology and algorithm comparison.

---

## 4. Next Part of Work Arrangement

The next part of the work will focus on the following. Systematic experiments will be carried out with different device counts, bandwidth and performance ranges, models and algorithms; the results on throughput, latency and scalability will be recorded and analysed, and tables and figures will be produced for the thesis. The thesis will be written in full, including the introduction, related work, system design, implementation, experiments and conclusion, and the description will be aligned with the implemented system. Documentation and code cleanup will be done, including updating the README and user guide and adding or refining comments in key modules for clarity and reproducibility. Defense preparation will be carried out, including the preparation of slides and a short live demonstration of the platform covering model selection, custom partition, simulation and algorithm comparison, and rehearsing answers to likely questions on design choices and limitations.

---

## 5. Problems Existing in the Work

At present, the following issues remain. The current system relies on simulated device performance and bandwidth, and the results have not been validated against real edge devices or networks; future work could incorporate lightweight measurements or link to a testbed. For very large models or a large number of devices, the interface and simulation may need further tuning in terms of loading states, error handling and optional simplification of visualizations to keep the demonstration smooth. The advanced methods such as full enumeration and PSO-based partitioning are implemented as standalone scripts and are not yet integrated into the web platform as optional algorithms; integrating them would strengthen the algorithm comparison part of the thesis if time permits. Part of the code and comments are still in Chinese; for consistency with an English thesis and possible open-source release, key modules and the README could be documented or translated into English where appropriate.
