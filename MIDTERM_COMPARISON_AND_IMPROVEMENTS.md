# 中期报告对比与完善建议

## 一、抄袭风险判断：**无抄袭痕迹**

| 维度 | 你的报告 | 同学报告 | 结论 |
|------|----------|----------|------|
| **研究主题** | 多设备 DNN 协同推理、HiveMind/EdgePipe、边缘计算 | 视频动作识别、TimeSformer、UCF101/SSv2 | 完全不同 |
| **技术路线** | 模型划分、流水线仿真、Flask+D3.js 平台 | ResNet+LSTM 基线、注意力机制消融、FastAPI+Streamlit | 完全不同 |
| **数据来源** | 层级 Flops/DataSize CSV | UCF101、Kinetics-400、SSv2 视频 | 完全不同 |
| **核心算法** | HiveMind（最短路径）、EdgePipe（DP 吞吐量） | TimeSformer（时空注意力）、CNN+RNN | 完全不同 |
| **实验指标** | 吞吐量、推理延迟、设备数/带宽/性能影响 | Top-1 准确率、FLOPs、混淆矩阵 | 完全不同 |

**结论**：两份报告在选题、方法、实验和结论上均不同，仅有的大致相似之处是中期报告的结构（一概述、二进度、三已完成研究、四工作安排、五存在问题），这是学校统一模板，不属于抄袭。

---

## 二、优劣对比

### 你的报告优势

1. **技术表述更集中**：围绕 HiveMind/EdgePipe 的优化目标（延迟 vs 吞吐量）展开，逻辑清楚。
2. **实验设计更规范**：有明确的实验因素、基线、控制变量，可复现性强。
3. **图表编号统一**：Figure 1–7、Table 2 编号清晰，无错位。
4. **问题与对策对应**：第五部分列出的问题与第四部分的工作安排一一对应，形成闭环。
5. **引用导师工作**：References [3] 引用戴鹏林老师相关论文，体现与导师课题的衔接。

### 同学报告优势

1. **概述部分更直观**：有 Figure 1（Transformer 结构）、Figure 2（五种注意力）、Figure 3（研究流程），开篇即展示方法。
2. **“存在的问题”更精炼**：采用“问题 + 下一步计划”的短段式写法，便于快速阅读。
3. **原型系统有量化数据**：如 “47 次推理、69.7% 平均置信度、1525 ms 延迟”，增强说服力。
4. **跨数据集验证**：UCF101 + SSv2 + Kinetics-400，体现泛化能力。

### 同学报告的问题（供你避免）

1. **图表编号混乱**：正文中 Figure 1–13 与图注不对应（如 3.1 写 Figure 1 但应为 UCF101 样本图；3.6 写 Figure 8、9 但图注为 Figure 11、12）。
2. **第二部分标题笔误**：“体安排”应为“整体安排”。
3. **第五部分有拼写错误**：“xfocus”应为“focus”。

---

## 三、你可完善之处

### 1. 第一部分：增加一张“研究流程/方法概览图”

同学在概述中用了 Figure 3 展示整体研究流程。你可增加一张类似图，展示：

```
数据准备(CSV) → 算法实现(HiveMind/EdgePipe) → 平台开发(Flask+D3.js) → 批量实验 → 结果分析与可视化
```

便于读者快速把握整体技术路线。

### 2. 第三部分：补充平台界面的量化数据（若有）

若平台已有运行记录，可仿照同学写法，加入具体数字，例如：

- “平台累计完成 XX 次仿真”
- “支持 4 种模型、设备数 3–6 可配置”
- “批量实验脚本在 XX 秒内完成 XX 组配置”

可增强“已完成研究”的说服力。

### 3. 第五部分：采用“问题 + 应对”的简洁结构

同学写法示例：

> **Fine-Grained Action Discrimination**: A primary challenge is... Our next step, as outlined in the future work, is to explore multi-scale feature fusion...

你可将现有五段式改为类似结构，每段包含：

- 问题简述（1–2 句）
- 计划中的应对措施（1 句，可与第四部分呼应）

例如：

> **Simulation vs. Real Deployment**: The evaluation is based entirely on simulation... Validation on real edge devices or a more detailed simulation incorporating network jitter and scheduling overhead is planned in the next stage (Section 4).

### 4. 第四部分：补充 Figure 8（时间线图）

第四部分末尾应插入 `figures/timeline.png`，并配图注，与同学报告中的时间线展示方式一致。

### 5. 第二部分：增加一句“进度自评”

同学在第二部分结尾写了：

> "The project has not only adhered to the intended schedule but has in several respects exceeded the original scope."

你可在第二部分末尾加一句类似总结，例如：

> "From the perspective of the current mid-term stage, the project has progressed largely in accordance with the original plan. The core system implementation and preliminary experiments have been completed, and the project remains on track for the final-stage analysis and thesis completion."

### 6. 检查第四、五部分与学校表格的对应关系

学校表格顺序为：

- **四** = 下一部分的工作安排  
- **五** = 毕业设计（论文）工作中存在的问题  

你当前 `MIDTERM_SECTION4_PROBLEMS.md` 与 `MIDTERM_SECTION5_WORK_ARRANGEMENT.md` 的命名与学校顺序相反，最终成稿时需按学校顺序放置内容。

---

## 四、总结

| 项目 | 评价 |
|------|------|
| **抄袭风险** | 无，内容与同学报告完全不同 |
| **整体质量** | 技术路线清晰，实验设计规范，问题与对策对应 |
| **可改进点** | 增加研究流程图、平台量化数据、精简第五部分结构、补充 Figure 8、第二部分收尾句 |

你的报告在技术深度和逻辑完整性上表现良好，按上述建议做小幅修改即可进一步提升。
