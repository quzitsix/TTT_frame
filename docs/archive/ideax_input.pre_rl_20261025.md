# 项目研究任务

> 快照日期：2026-10-20
> 项目根目录：`/home/quzitsix/TTT_frame`
> 研究类型：多模态 LLM 架构、流式视频记忆、Test-Time Training（TTT）、第一人称视频预测
> 证据约定：本文将内容标为 **实际结果**、**已实现**、**机制推断**、**计划** 或 **待确认**。只有代码、运行产物或原始数据检查支持的内容才写成事实。外部论文的阅读范围逐项列出；仅阅读摘要或项目页时不将其写成已复现结论。

## 1. 研究目标

### 1.1 最终问题

项目希望研究一种适用于第一人称家庭视频流的参数化记忆架构：模型按时间顺序观察视频，在不长期保留全部历史帧、caption 或 KV cache 的条件下，把与人物、物体、位置、状态变化和行为规律有关的信息写入固定或缓慢增长的内部状态，并在之后回答问题或预测未来事件。

当前最值得继续验证的版本不是笼统的“TTT 能否记住长视频”，而是：

> **在严格因果、相同视觉证据和相近计算预算下，以未来预测作为外层训练信号，能否学到比 frozen VLM、长上下文 ICL、显式检索、固定状态 recurrent/SSM 和现有 Spatial-TTT 更有效、更稳定的 test-time parameter memory？**

这里的“未来预测”至少可分为：

1. **短期预测**：预测数秒内将交互的物体、动作、接触时间或状态变化；
2. **中长期状态预测**：根据此前观察预测物体最后位置、下一次交互、人物行为模式或事件结果；
3. **预测辅助记忆**：未来标签本身不一定是最终任务，而是用来训练“哪些信息值得写、写到哪里、何时巩固或忘记”的机制；
4. **预测误差驱动更新**：在线阶段只能利用当前已观察到的 token 及由它们产生的自监督误差，不能偷看查询时刻之后的视频。

### 1.2 主要研究对象

主要研究对象是 **LLM/VLM 中可在测试时更新的 fast-weight 层及其离线学习规则**，而不是单独提高 caption teacher 的语言质量。要研究的组成部分包括：

- 视频 token 到 fast state 的写入目标；
- fast state、短窗口注意力和慢参数之间的信息分工；
- 对象、事件、时间和场景变化的绑定与遗忘；
- 未来预测目标如何训练写入率、路由、巩固和读取；
- 离线 slow-weight 训练与在线无标签更新的因果边界；
- 固定状态容量、计算量、显存和推理延迟之间的权衡。

### 1.3 预期贡献的最低标准

一个成立的架构贡献至少需要同时做到：

1. 相对当前 Spatial-TTT 或其他直接基线，有可定位的结构或学习规则增量，而不只是调学习率、chunk 大小或换 backbone；
2. 在参数量、训练 token、可见帧、分辨率和总计算量受控的比较中改善目标任务；
3. 通过 slow-only、禁用在线更新、打乱时间、遮蔽未来、等预算 ICL、显式检索和 recurrent/SSM 对照，证明收益来自所声称的记忆机制；
4. 给出能推翻机制假设的实验，而不是只报告平均分上涨；
5. 先在本项目资源内完成小规模机制验证，再讨论大规模预训练。

本文不预设 TTT 一定优于显式检索或 recurrent state。如果匹配预算后优势消失，合理的研究结论可以是：某类预测/记忆问题更适合结构化外部状态、短期 KV 或选择性 SSM，而非在线参数更新。

## 2. 项目现状

### 2.1 已实现的三条主要路线

#### 2.1.1 LoRA-TTT：由临时教师文本写入参数

**代码路径**：[ttt_frame/videoqa.py](ttt_frame/videoqa.py)、[ttt_frame/video.py](ttt_frame/video.py)、[docs/ARCHITECTURE_LORA_TTT.md](docs/ARCHITECTURE_LORA_TTT.md)。

**信息流**：

```text
按时间切分的视频帧
  -> 冻结 VLM（禁用当前 LoRA）生成临时 observation / pseudo-QA
  -> 丢弃原图前，用 teacher-forced causal CE 更新语言解码器 q_proj/v_proj 上的 LoRA
  -> 丢弃帧、临时 caption、pseudo-QA 和 optimizer state
  -> 之后仅加载 LoRA 参数回答问题
```

**相对基础模型的改动**：

- 默认基础模型为 Qwen3-VL-2B-Instruct；视觉编码器和原始权重冻结；
- 在 28 个语言层的 `q_proj`、`v_proj` 上加入 LoRA，共 56 个投影、112 个 LoRA 张量；
- 当前默认 `rank=16`、`alpha=32`，可训练参数为 3,211,264 个 FP32 值，约 12.25 MiB；
- teacher 与 student 默认是同一个 VLM 实例，teacher 生成时禁用 adapter；项目也实现过外部 Codex/OpenAI-compatible teacher 接口；
- 每个视频块顺序优化，问答阶段冻结记忆；不同 household/session 之间显式 reset。

**训练目标**：student 对临时 caption/QA 的标准 causal language-model cross entropy。该目标训练的是“复述 teacher 产生的文本”，并不直接预测未来视觉状态。

**关键边界**：历史视觉首先被压缩为 teacher 文本，再压缩到 LoRA。teacher 漏掉、识别错或编造的信息没有可靠恢复路径；连续 AdamW 更新也没有显式 replay、重要性保护或多时间尺度巩固。

#### 2.1.2 Spatial-TTT：视觉 token 直接写入分层 fast weights

**代码路径**：[ttt_frame/spatial_memory.py](ttt_frame/spatial_memory.py)、[ttt_frame/spatial_model.py](ttt_frame/spatial_model.py)、[ttt_frame/spatial_videoqa.py](ttt_frame/spatial_videoqa.py)、[docs/SPATIAL_TTT.md](docs/SPATIAL_TTT.md)。

**信息流**：

```text
视频块 -> Qwen3-VL visual tokens -> 3D depthwise conv / QKV preparation
      -> 多个 language layer 中的 [local SWA || nonlinear TTT branch]
      -> 逐 chunk 更新每层 SwiGLU fast weights W0/W1/W2
      -> 清除帧、历史 token 和历史 KV，仅保留 fast state
文本问题 -> 同一组层以 read 模式读取 fast state -> LM head 生成答案
```

**相对基础模型的改动**：

- 在 Qwen3-VL-2B 的 28 个语言层中，当前实现选择每四层中的前三层，共 21 个 TTT 层；
- 每层加入局部滑动窗口注意力与 TTT 分支，视觉 token 还经过 3D 深度可分离卷积；
- 每个 TTT 层维护按 head 分解的 SwiGLU 参数 `W0/W1/W2`；query 读取旧状态，key/value 根据关联目标写入；
- 写入目标在代码中为 `sum(f_W(K) * V)` 的关联 ascent，不是 MSE 重构；
- 可使用 token-dependent learning rate、momentum、Muon 型矩阵更新和行范数归一化；
- 运行时 fast state 与 checkpoint 中的 slow/initial weights 分离，可以单独序列化；
- 当前实现为纯 PyTorch SDPA/matmul，没有复现官方 fused CUDA/Triton kernel。

抽象地看，单层在一个 chunk 内执行：

```text
read:    y_i = f_{W_t}(q_i)
update:  ΔW_t ∝ Muon(∇_W Σ_i <f_{W_t}(k_i), v_i>)
         W_{t+1} = row_normalize(W_t + gated_rate * ΔW_t)
```

实际顺序、pending chunk、momentum 和 residual 融合应以代码为准。

**状态规模**：当前 2B 配置的 21 层 fast state 共 66,060,288 个 FP32 值，约 252 MiB；启用 momentum 后再增加约 252 MiB。运行时还需基础模型、slow weights、激活、视觉编码和生成 KV，因此总显存远高于约 505 MiB 的 fast-state 与 momentum 本身。

**当前在线目标**：测试时的关联写入只使用当前 token 的 K/V。它没有“下一帧”“下一动作”或“未来 latent”监督，因此不能把现有实现称为 forecasting TTT。

**`without_memory` 的含义**：Spatial 的 slow-only 条件仍会加载官方 Spatial-TTT slow checkpoint，只清空或禁用本视频产生的 fast state。它不是原始 Qwen 基座。这解释了为什么有/无视频记忆可能输出相同答案：官方 slow weights 和语言先验仍然存在，而当前视频 fast update 可能没有改变最终 argmax。正式因果比较应优先使用同 checkpoint 下的 `spatial` 对 `spatial_slow_only`。

#### 2.1.3 Spatial slow-weight 离线训练器

**代码路径**：[ttt_frame/spatial_trainer.py](ttt_frame/spatial_trainer.py)。

**流程**：一个 episode 从空 fast state 开始；按时间写入已经预处理好的视频 batch；随后对文本 QA completion 计算 teacher-forced causal CE；梯度穿过 fast writes，更新选择的 Spatial slow parameters；optimizer step 后清空 episode state。

**已实现边界**：

- 基础 Qwen attention 默认冻结；可训练新增 Spatial 参数，或只训练 readout 子集；
- 有梯度裁剪、非有限检查和 episode reset；
- 调用者必须提供 processed write batches 和 QA batch；
- 目前没有完整 dataset loader、DDP 训练入口、统一 checkpoint schema 或官方规模训练 recipe；
- 没有 gradient checkpointing 与 stateful write 的完整兼容方案；
- **没有未来 latent、下一动作、下一状态或 prediction-error loss**。

因此，该训练器证明“slow weights 可以通过最终 QA loss 学习如何写入”，但尚未完成用户期望的 forecasting-oriented training framework。

### 2.2 其他已实现的诊断路线

1. **Standalone LaCT symbolic probe**：64 维合成 semantic code 上的 fast-weight 关联记忆，见 [ttt_frame/lact.py](ttt_frame/lact.py) 和 [ttt_frame/_lact_upstream.py](ttt_frame/_lact_upstream.py)。它用于隔离机制，不等于官方 Spatial-TTT，也不含真实视觉编码。
2. **Frozen V-JEPA 2 probe**：用官方 ViT-L/16 checkpoint 提取冻结特征，再用 leave-one-content-group-out kernel ridge 读出。它是监督表征对照，不是 TTT，也没有跨 episode 持久状态。见 [docs/JEPA_BASELINE.md](docs/JEPA_BASELINE.md)。
3. **多种评测 runner**：HomeSentinel、SVCBench、C-PNG、FHO-STA 协议和比较脚本已存在，但不同实验的监督条件并不一致，不能把所有绝对分数放在同一排行榜中。

### 2.3 已有实验与结果

下表只列项目中能由结果文件或明确文档核查的内容。

| 实验 | 数据/设置 | 实际结果 | 可以得出的结论 | 不能得出的结论 |
|---|---|---|---|---|
| HomeSentinel Asuka 正式 Spatial 运行 | 52 个有序室内视频，约 33,297 秒；138 个 open-ended 问题；60 秒/4 帧；Qwen3-VL-2B + 官方 Spatial nano | memory mean token-F1 0.1554；slow/no-memory 0.1624。event 为 0.1224 对 0.1186，home 为 0.1951 对 0.2172，owner 为 0.1514 对 0.1552。两边 exact 均为 0；memory 与 control 有 96/138 个答案不同，memory 赢 33、输 41、平 64 | fast state 确实会影响生成，但没有稳定提高该词法指标；event 子类有很小正差，其他类为负 | 不能凭 token-F1 断定语义完全错误；也不能断定更密采样或经过任务训练的 TTT 一定无效 |
| HomeSentinel 快速 ICL 诊断 | 15 题、历史帧预算不完全匹配 | full-video ICL 0.2001，Spatial 0.1660，frozen 0.1730 | ICL 是必须纳入的强基线 | 样本太小、预算不完全匹配，不能形成主结论 |
| SVCBench Natural9 matched-prefix | 9 个真实来源视频、42 条轨迹问题、125 query points、625 个五臂结果；相同 cutoff 和帧选择 | query exact：blind 0.240、ICL 0.264、Spatial 0.240、slow-only 0.232、base bypass 0.240；question macro：0.3060、0.2889、0.3131、0.2980、0.3060。fast 相对 slow 净多 1/125；Spatial 122/125 输出为 1 或 2 | 当前 Spatial 有明显数值输出塌缩；fast state 没有产生稳健 counting 增益 | counting 混合识别、re-ID、去重、累积和数字 readout，不能单独代表一般视频记忆能力 |
| C1 symbolic mechanism probe | 12 episodes × 3 conditions × 5 question types × 10 trials；64 维 LaCT | oracle/retrieval 1.0000；LaCT relocated 0.6383、same-scene 0.7333、filler 0.4967；location 0.8111、mover 0.8528、owner 0.3028、state 0.6833、next-action 0.4639 | 关联 fast weights 能保存部分结构化绑定，但场景漂移、owner 与 next-action 绑定较弱 | 合成 semantic code 结果不能外推到真实视频或语言生成 |
| C2 C-PNG strict visual MCQ | 12 episodes × 3 conditions × 5 问题，180 题/arm；未来卡隐藏 | ICL 89/180=0.4944；blind 0.2667；Spatial fast 0.2667；slow-only 0.2833；base bypass 0.2722。Spatial 预测 A 达 130/180 | 同一官方 checkpoint 下，当前 fast write 没有可检测收益；readout 有强选项偏置 | fixture 有文字卡、周期规律和 control 字段捷径，不能作为真实视频有效性证据 |
| C3 frozen V-JEPA 2 probe | 同 C-PNG，冻结 ViT-L 特征 + group-aware kernel ridge | V-JEPA 126/180=0.7000；匹配低分辨率 pixel feature 132/180=0.7333 | 表征 probe 未显示 JEPA 特征超过简单 pixel control；fixture 可被模板捷径利用 | 这是有监督 probe，不能与零样本 Spatial/ICL 的绝对分数直接排名，也不是在线记忆 |
| Q9 长视频密集写入 pilot | 1080 秒，4 秒/8 帧，共 2160 帧、约 211,680 visual tokens | Spatial ingest 约 3843 秒；正式 MCQ 中 memory 与 control 都答 C、gold 为 A。Codex 派生 19 题中 dense Spatial 14/19、sparse 12/19、no-memory 12/19；文本模型可达 16/19 且 shuffle 后 17/19 | 密集写入成本高；该题集语言/选项泄漏很强，不能证明视觉记忆 | 不能把 14/19 归因于 TTT；题目由 teacher 内容派生，独立性不足 |
| LoRA teacher pilots | SuperMemory 5 题、EPIC 小片段、local/Codex teacher | SuperMemory：blind 1/5、LoRA 12-step 3/5、LoRA 3-step 3/5、notes 2/5、直接视觉 1/5；EPIC teacher target 出现重复和幻觉 | LoRA 参数确实能写入 teacher 文本；更高 optimizer steps 未在小样本上带来额外收益；teacher 质量是瓶颈 | 5 题不能证明总体提升；loss 下降不能证明形成正确视觉记忆 |
| FHO-STA 实验 A | 已有协议、脚本和文档 | **没有可用结果文件；本地 Ego4D FHO 数据不可用** | 目前只能把它列为计划/受阻实验 | 不得写成已失败或已有 noun/verb accuracy |

主要结果文件：

- [runs/homesentinel_asuka_spatial_60s4f_run6/summary.json](runs/homesentinel_asuka_spatial_60s4f_run6/summary.json)
- [runs/svcbench_real_natural9_matched_v1/summary.json](runs/svcbench_real_natural9_matched_v1/summary.json)
- [runs/experiment_c_v2_final.summary.json](runs/experiment_c_v2_final.summary.json)
- [runs/experiment_c_visual_v2_ep12.summary.json](runs/experiment_c_visual_v2_ep12.summary.json)
- [docs/PREDICTIVE_TTT_EXPERIMENTS_RESULTS.md](docs/PREDICTIVE_TTT_EXPERIMENTS_RESULTS.md)
- [docs/FHO_STA_EXPERIMENT_A_RESULTS.md](docs/FHO_STA_EXPERIMENT_A_RESULTS.md)
- [docs/PROJECT_EXPERIMENT_SUMMARY.md](docs/PROJECT_EXPERIMENT_SUMMARY.md)

### 2.4 数据实际情况

#### HomeSentinel

- `/data/HomeSentinel/asuka` 是当前唯一媒体完整的 creator：52 个有序视频；raw video 总长约 38,682 秒，indoor video 总长约 33,297 秒；整个目录约 40 GiB；
- `benchmark_queries` 有 138 个 open-ended 问题：owner 52、home 42、event 44；event 和部分 home 带 cutoff；
- `merged_captions.json` 有大量自动聚合文本，但不是人工视觉真值；不得把 caption-only 分数等同于原视频记忆；
- 当前没有正式 train/val/test split；若用同一 creator 训练和测试，必须按视频/时间/实体身份避免泄漏；
- 其余 116 个列出的 creator 视频当前没有本地媒体。

数据说明见 `/home/quzitsix/meowbench/docs/HOMESENTINEL_README.md`。过去文档记录过 Qwen3.5-35B-A3B caption-only 的词法结果，但本次检查未发现 `/data/hf_models/Qwen/Qwen3.5-35B-A3B`，因此该模型当前是否仍可用标为 **待确认**。

#### SVCBench Natural9

- 本地子集 `/data/quzitsix/svcbench_real_natural9_v1` 约 619 MiB；
- 9 个视频来自 Ego4D、ARKitScenes、ScanNet、ScanNet++、RoomTour3D、CODa 和 YouTube 等 7 类来源；部分是第一人称或扫描序列，不能统称为家庭第一人称视频；
- 当前主要问题为 streaming counting，适合作为 readout/状态维护诊断，不应成为项目唯一主任务。

#### C-PNG

- 项目生成的文字 PNG-card 机制 fixture；便于严格控制未来泄漏、场景变换和 observation budget；
- 存在模板、标签周期、控制字段和低级像素捷径；已经证明继续扩大同类卡片的研究价值有限；
- legacy schema v1 曾包含 forecast target 泄漏，旧结果只保留作协议错误案例，不得作为有效证据。

### 2.5 当前已经尝试与仍未完成的方向

| 方向 | 状态 | 结论或缺口 |
|---|---|---|
| 视觉 -> teacher caption/QA -> LoRA 写入 | 已实现、做过小规模 pilot | teacher omission/hallucination 与顺序遗忘明显；尚无大规模独立 benchmark |
| 官方 Spatial slow weights + 本地流式 fast state | 已实现、多数据诊断 | fast state 改变输出，但目前没有稳定准确率收益 |
| dense/sparse 视频采样 | 已尝试 | dense 可能覆盖更多短事件，但成本很高；尚未做严格 accuracy-compute Pareto 曲线 |
| 同 checkpoint fast vs slow-only | 已实现 | 是当前隔离 fast update 的主要消融；现有结果未支持 fast 的稳定收益 |
| Spatial slow-weight QA 外层训练 | 最小训练器已实现 | 缺数据流水线、DDP、完整 recipe 与预测目标；尚未形成正式训练结果 |
| forecasting loss / future-latent objective | **未实现** | 是下一阶段核心研究空白 |
| prediction-error/surprise update gate | **未实现** | 已有外部工作覆盖相近概念，不能直接当新意 |
| 多时间尺度 fast/anchor/consolidation | **未实现** | 外部 Fast Spatial Memory 已覆盖 Fisher/EMA elastic anchor 的一种实现 |
| StreamTTT 风格短 KV + fast weights | **未实现** | 外部 StreamTTT 已是最直接竞争方案，简单照搬不构成新方法 |
| 显式视觉检索、对象轨迹或事件图 | symbolic 对照存在，真实视频未接入 | 是必须补的强基线，也可能比参数记忆更合适 |
| Mamba/SSM/recurrent fixed-state baseline | 未实现 | 缺少能区分“在线优化”与“一般递归状态”的关键控制组 |
| V-JEPA 2 frozen encoder | 已在 C-PNG 做监督 probe | 还未与真实视频上的共同 readout、相同监督和同预算比较 |
| FHO-STA / Ego4D anticipation | 协议有、数据无 | 尚无预测任务的实际主结果 |
| HomeSentinel forecasting labels | 未构建 | 原问题偏回忆；可从事件时间线构造，但必须人工/规则审计并切断未来泄漏 |

## 3. 已观察到的问题

### 3.1 有实验支持的观察

1. **当前 Spatial fast state 没有稳定带来准确率收益。** HomeSentinel 总体低于 slow/no-memory；C-PNG 低于 slow-only；Natural9 fast 相对 slow 只净多 1/125。不同任务、指标和输入形式上方向不一致。
2. **写入发生不等于记忆可读。** HomeSentinel 中 96/138 个回答因 fast state 改变，但总体分数未改善；Natural9 输出集中在 1/2；C-PNG 强烈偏 A。当前瓶颈可能位于写入、保持、问题到 key 的寻址、语言 readout，或这些环节的组合。
3. **稀疏采样会错过短交互，密集采样代价又很高。** Q9 1080 秒以 4 秒/8 帧写入需约 64 分钟；HomeSentinel 60 秒/4 帧仍运行约 89 分钟。采样率和更新成本是架构问题，不只是数据预处理问题。
4. **现有合成卡和 teacher-derived 问题存在捷径。** C-PNG 的 pixel control 高于 V-JEPA；Q9 文本模型在无视频情况下得分很高；legacy fixture 出现未来目标泄漏。因此必须把 blind、shuffle、cutoff 和低级特征 control 作为正式协议的一部分。
5. **counting 不是当前最合适的唯一任务。** Natural9 暴露了状态累积与 readout 问题，但同时依赖检测、实体去重、重识别和数字生成，不能明确定位 TTT 的优势来源。
6. **teacher 质量限制 LoRA-TTT。** EPIC pilot 的临时 supervision 有重复和幻觉；LoRA 只能压缩 teacher 说出的内容。
7. **现有 JEPA 结果不能作为 JEPA 与 TTT 的公平胜负。** JEPA 使用其他 episode 的监督 readout；Spatial 是零样本生成；pixel 模板本身已很强。

### 3.2 尚未验证的机制假设

以下是合理但尚未由项目实验支持的假设：

1. 未来 latent / next-state 目标会迫使 fast state 保存可预测、任务相关的因果信息，因此优于当前无语义约束的 K/V 关联更新；
2. prediction error 或 surprise 可以作为写入门控，减少冗余帧更新与长期漂移；
3. 物体状态、事件顺序和人物习惯需要不同时间尺度或不同参数子空间，单一共享 fast matrix 会相互干扰；
4. fast state 可能已经存有信息，但纯文本 query 的 key 与视觉写入 key 不对齐，造成读取失败；
5. 官方 Spatial slow weights主要为空间重建/空间智能训练，与家庭事件和行为预测域不匹配；任务化 slow-weight 训练可能显著改变结果；
6. 预测训练的收益可能全部来自更好的 frozen visual representation，而不需要测试时更新；
7. 显式对象/事件表、retrieval memory 或 selective SSM 可能在当前数据规模和监督条件下更强、更稳定。

这些假设必须通过互相区分的消融验证。例如，若加入 future loss 后 frozen encoder 同样提升而 online update 没有额外收益，就不能把收益归因于 TTT。

### 3.3 当前缺少的证据

1. 没有完成任何标准真实视频 anticipation benchmark 的主实验；FHO-STA 仍无本地数据结果。
2. 没有在相同 backbone、相同训练 token、相同可见帧、相近 FLOPs 下比较 TTT、ICL、retrieval、Mamba/SSM 和普通 recurrent memory。
3. 没有证据说明信息是在写入时丢失、随时间漂移，还是读取时失败；缺少可探测 fast state 的 linear probe、controlled recall 和 overwrite 曲线。
4. 没有 fast state 容量随对象数、事件数、时间跨度、场景切换次数变化的 scaling law。
5. 没有按 prediction horizon、evidence sparsity、scene shift、entity recurrence 和遮挡程度分层的结果。
6. 没有严格证明未来帧、聚合 caption、teacher 答案或同实体跨 split 信息未泄漏到训练和查询。
7. 没有 end-to-end forecasting slow-weight checkpoint，也没有训练稳定性曲线、多个 seed 或统计置信区间。
8. 没有完整的吞吐、训练 FLOPs、峰值显存、query latency、状态序列化大小与 KV cache 对照。
9. HomeSentinel 没有正式数据 split；其 open-ended 词法指标与语义正确性的关系未人工校准。
10. 尚未确认可接受的总训练时长、外部数据下载许可、Qwen3.5 模型是否仍在本机，以及是否允许为实验占用全部 8 张 GPU。

### 3.4 目前最具体的瓶颈定位

| 层面 | 当前瓶颈 | 证据/来源 |
|---|---|---|
| 感知 | 低采样率漏掉短交互；caption teacher 会遗漏或幻觉 | Q9/HomeSentinel 采样配置；EPIC teacher pilot |
| 写入目标 | 当前关联 ascent 不要求保存未来有用的语义 | `spatial_memory.py`；无 forecast loss |
| 时间表达 | 顺序主要来自 causal order、RoPE、SWA、3D conv 和 chunk 次序；没有显式 clock/event slot | `spatial_model.py` 和当前 state schema |
| 稳定性 | 连续更新可能漂移、覆盖；无 replay、importance consolidation 或状态分区 | 当前实现；长期结果未提升 |
| 寻址 | 视觉写入与文本问题读取可能 key-space 不对齐 | 机制推断；需 probe 验证 |
| 读取 | MCQ 字母偏置、count 输出塌缩、生成先验强 | C-PNG、Natural9 实际结果 |
| 外层训练 | 只有最终 QA CE 的最小 trainer；无 prediction curriculum | `spatial_trainer.py` |
| 计算 | 序列内 fast state 有严格依赖，不能把一个 episode 的 chunk 随意并行到多卡 | 架构依赖；可并行 episode/arm |
| 评测 | 真实预测数据缺失；旧 fixture/teacher-derived 问题有捷径 | FHO-STA 无结果；C/Q9 审计 |

## 4. 相关工作与方法关系

### 4.1 阅读说明

本节优先记录作者论文页、arXiv、会议页和官方代码。由于本轮目标是准备 IdeaX 输入而不是完成系统综述，多数来源阅读到 **摘要、方法概览、官方项目页或 README**；未逐页阅读附录和全部实验表的条目会明确标记。发布时间按 arXiv 首次提交、会议年份或官方项目页标注记录，不把搜索引擎摘要当作原始证据。

### 4.2 TTT 与 fast-weight 主线

#### 1. Learning to (Learn at Test Time): RNNs with Expressive Hidden States

- 论文：[arXiv:2407.04620](https://arxiv.org/abs/2407.04620)，首次提交 2024-07-05；当前检索版本 v4 为 2025-08-31。
- 代码：[官方 PyTorch](https://github.com/test-time-training/ttt-lm-pytorch)。
- 阅读范围：摘要、论文方法概览、官方仓库 README；未逐项复核全部语言模型实验。
- 核心机制：把序列模型 hidden state 改成通过自监督损失在线更新的 TTT-Linear/TTT-MLP 参数；外层训练学习初始参数和更新规则。
- 适用条件与局限：需要可在线计算的自监督目标；参数状态更有表达力，但每 token 优化与内存 I/O 成本高。原始验证以语言序列为主，不直接解决视觉实体绑定和视频预测。
- 与本项目关系：这是“参数即状态”的基础。任何候选若只把 RNN state 换成 MLP fast weights，已被该工作覆盖；新增点必须落在多模态写入目标、时间结构、训练方式或稳定性上。

#### 2. Test-Time Training Done Right / Large Chunk TTT（LaCT）

- 论文：[arXiv:2505.23884](https://arxiv.org/abs/2505.23884)，2025-05-29。
- 项目与代码：[项目页](https://tianyuanzhang.com/projects/ttt-done-right)、[代码](https://github.com/a1600012888/LaCT)。
- 阅读范围：摘要、项目页、与本地实现对应的官方算子说明；未通读所有任务附录。
- 核心机制：用大 chunk 近似逐 token TTT，引入非线性状态、Muon 等更新以提高表达力和硬件效率；覆盖语言、NVS 和视频生成等任务。
- 局限：chunk-start 近似可能削弱 chunk 内精细顺序；fast state 仍面临跨 chunk 漂移与激活/内存压力。
- 与本项目关系：本地 Spatial fast-weight 算子直接继承其非线性关联更新思想。仅更改 chunk size、Muon 或 MLP 维度通常属于复现/调参，而非新的架构贡献。

#### 3. Spatial-TTT: Streaming Visual-based Spatial Intelligence with Test-Time Training

- 论文：[arXiv:2603.12255](https://arxiv.org/abs/2603.12255)，2026-03-12。
- 代码：[THU-SI/Spatial-TTT](https://github.com/THU-SI/Spatial-TTT)。
- 阅读范围：摘要、官方 README、主要结构说明，并与本地 wrapper/fast-weight 代码核对；未复现官方训练。
- 核心机制：把 large-chunk TTT、sliding-window attention 和 3D 时空卷积组合进视觉语言骨干，用 dense 3D description 等数据训练流式空间状态。
- 局限：目标偏空间智能与重建式表征，不等于对象事件记忆或显式未来预测；官方 slow weights 的训练域与家庭行为预测不一致。
- 与本项目关系：它是当前实现最直接的架构来源。IdeaX 不能把“在 Qwen-VL 中加入 Spatial-TTT block”作为新 idea；必须明确改写哪一个目标、状态结构或写读规则。

#### 4. StreamTTT

- 论文：[arXiv:2608.13416](https://arxiv.org/abs/2608.13416)，首次提交 2026-08-13，检索版本 v4 为 2026-09-10。
- 代码：[zeyun-zhong/StreamTTT](https://github.com/zeyun-zhong/StreamTTT)。
- 阅读范围：摘要、官方 README 和公开系统配置；尚未在本机部署或复现实验。
- 核心机制：Qwen3-VL 中用 fast weights 保存长历史，同时保留约 4K 的短期 KV，上层联合训练离线 long-video QA 与实时 streaming QA。
- 适用条件与局限：最贴近“长视频流 + 固定状态 + VLM QA”，但需要其专门训练数据与训练 recipe；公开结果不自动转移到本项目家庭预测域。
- 与本项目关系：这是最直接的外部竞争系统。简单的“Spatial fast state + 短 KV”“同时训 offline/online QA”已被覆盖；新 idea 必须说明预测目标、状态组织或因果学习规则上的实质差异。

#### 5. Rethinking Expressivity and Efficiency in TTT / E²-TTT

- 论文：[arXiv:2608.21308](https://arxiv.org/abs/2608.21308)，2026-08-21，v2 2026-08-26。
- 代码：[zeyun-zhong/E2-TTT](https://github.com/zeyun-zhong/E2-TTT)。
- 阅读范围：摘要和官方 README；未复现。
- 核心机制：针对 chunk-start 近似给出闭式 chunk state transition，试图保留逐 token 更新动态并提高长度外推与检索效率。
- 局限：验证主要是语言建模/检索；没有直接解决视觉 grounding、未来监督和长期干扰。
- 与本项目关系：可作为更新算子的候选强基线。若 idea 只声称“更精确地并行 LaCT 更新”，需要与 E²-TTT 逐式比较。

#### 6. End-to-End Test-Time Training for Long Context

- 论文：[arXiv:2512.23675](https://arxiv.org/abs/2512.23675)，2025-12-29。
- 代码：[test-time-training/e2e](https://github.com/test-time-training/e2e)。
- 阅读范围：摘要、官方 README；未复现大规模训练。
- 核心机制：标准短窗口 Transformer 在上下文上继续 next-token test-time training，外层 meta-learn 初始化，使在线学习本身成为长上下文机制。
- 局限：语言预训练规模大，直接复现超出本项目当前预算；next-token loss 也未必保留稀疏视觉实体状态。
- 与本项目关系：支持“必须离线学习如何在线学习”的原则。项目最小 trainer 是雏形，但尚未有等价规模的 meta-training 或 forecasting objective。

#### 7. Fast Spatial Memory with Scalable Elastic Test-Time Training

- 论文：[arXiv:2604.07350](https://arxiv.org/abs/2604.07350)，2026-04-08；项目页标注 ECCV 2026。
- 项目/代码：[项目页](https://fast-spatial-memory.github.io/)；代码链接由项目页提供。
- 阅读范围：项目页摘要、方法和主要结果；未下载模型、未通读附录。
- 核心机制：LaCET 为 LaCT fast weights 加入 Fisher-style 重要性与动态 EMA anchor，弹性拉回重要参数，减少灾难遗忘和相机插值捷径；用于 3D/4D novel-view rendering。
- 局限：依赖 posed images、ray/time embedding 和大规模 3D/4D 预训练；目标不是语言 QA 或行为预测。
- 与本项目关系：直接覆盖“fast/slow anchor + 重要性巩固”的一般想法。若提出多时间尺度记忆，必须超过简单 EMA/Fisher elastic consolidation，并在事件记忆上给出新的信息流。

### 4.3 预测、在线适应与世界模型主线

#### 8. Forget, Anticipate and Adapt: Test-Time Training for Long Videos / Frame Forgetting Network

- 论文：[arXiv:2606.26515](https://arxiv.org/abs/2606.26515)，2026-06-25；官方仓库标注 ECCV 2026。
- 代码：[rajatmodi62/ffn](https://github.com/rajatmodi62/ffn)。
- 阅读范围：摘要与官方 README；未复现。
- 核心机制：只维持 exiting/current/anticipated-next frame 表征，用预测 surprise 调整时间窗口和 test-time adaptation，并提供 memory restoration；用于长视频 segmentation/classification/depth。
- 局限：任务和输出不是开放式 VLM QA；长期离散实体/事件绑定能力仍需另测。
- 与本项目关系：这是最接近“以 forecasting error 控制 TTT 写入”的工作。仅增加 surprise gate 或 next-frame loss 很可能已被覆盖；IdeaX 必须提出与参数状态、实体绑定或多任务读出的更深结合。

#### 9. Test-Time Training on Video Streams

- 论文：[arXiv:2307.05014](https://arxiv.org/abs/2307.05014)，2023-07-11，v3 2025-01-04。
- 代码：[test-time-training/ttt-video](https://github.com/test-time-training/ttt-video)。
- 阅读范围：摘要和代码 README；未复现。
- 核心机制：在视频流上用当前/近期帧的自监督重构目标在线适应，用于 segmentation 和 depth；强调时间 locality。
- 局限：性能提升可能来自 domain adaptation 或邻近帧相似性，而非长期记忆；不处理长程问答。
- 与本项目关系：这是重要的反例解释。任何 forecasting-TTT 增益都必须用 reset、远距证据和 domain-shift controls 区分“短期适应”与“长期记忆”。

#### 10. Anticipative Video Transformer（AVT）

- 论文：[arXiv:2106.02036](https://arxiv.org/abs/2106.02036)，2021-06-03；ICCV 2021。
- 代码/模型：[官方项目](https://facebookresearch.github.io/AVT/)；原始代码仓库已归档。
- 阅读范围：摘要与官方项目说明；未复现。
- 核心机制：causal attention 同时学习 next-action classification、未来 frame feature prediction 和中间 action supervision。
- 局限：历史主要保存在普通 attention/context，不进行测试时参数更新。
- 与本项目关系：必须作为“预测训练本身已足够”的非 TTT 强基线。若 future loss 同时改善 AVT/frozen recurrent baseline，不能把收益全部归于 fast weights。

#### 11. StillFast

- 论文：[arXiv:2304.03959](https://arxiv.org/abs/2304.03959)，2023；CVPR Workshops 2023。
- 代码/权重：[fpv-iplab/StillFast](https://github.com/fpv-iplab/StillFast)。
- 阅读范围：摘要、官方 README 与公开权重说明；未在本机运行。
- 核心机制：融合低帧率的 slow video 与高分辨率 recent still frame，预测下一交互的 object、verb 和 time-to-contact。
- 局限：主要解决短期 hand-object anticipation，不是长时固定容量记忆。
- 与本项目关系：是 FHO-STA 最小实验的直接强基线，也能检验 TTT 是否只是弥补近期高分辨率视觉不足。

#### 12. V-JEPA 2 与 V-JEPA 2.1

- V-JEPA 2：[arXiv:2506.09985](https://arxiv.org/abs/2506.09985)，2025-06-11。
- V-JEPA 2.1：[arXiv:2603.14482](https://arxiv.org/abs/2603.14482)，2026-03-15，检索版本 v3 为 2026-06-11。
- 代码/模型：[facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2)。
- 阅读范围：两篇摘要、官方仓库与项目说明；本项目只复现了 V-JEPA 2 ViT-L frozen feature probe，没有复现 2.1 训练。
- 核心机制：V-JEPA 2 在 latent space 做 masked prediction，并包含 action anticipation / action-conditioned world model 方向；2.1 增加对可见与 masked token 的 dense predictive loss 和深层自监督，以得到更密集的时空表征。
- 局限：强表征不等于 episode-persistent memory；预训练数据和算力远超本项目从头复现能力。
- 与本项目关系：future-latent loss 的关键来源与非 TTT 对照。简单“把 V-JEPA loss 加到 Spatial”不足以证明架构创新，必须解释该 loss 如何学习写入、路由或巩固规则。

#### 13. V-JEPA 2 with Action-Conditioned World Models

- 来源同上 V-JEPA 2 论文和仓库。
- 阅读范围：摘要/项目页中 action-conditioned 部分；未复现机器人模型。
- 关系说明：若项目未来引入 ego-motion、hand action 或控制信号，action-conditioned predictor 是合理基线；但 HomeSentinel 当前没有统一 action/control 标签，数据条件 **待确认**。

### 4.4 非 TTT 记忆架构与反事实基线

#### 14. MA-LMM: Memory-Augmented Large Multimodal Model for Long-Term Video Understanding

- 论文：[arXiv:2404.05726](https://arxiv.org/abs/2404.05726)，2024-04-08；CVPR 2024。
- 代码/项目：[作者项目页](https://boheumd.github.io/MA-LMM/)。
- 阅读范围：摘要和官方项目页；CVF HTML 本轮访问返回 403，未逐页阅读正文。
- 核心机制：在线处理视频，把历史视觉信息写入显式 memory bank，避免一次把全部帧送进 LLM context。
- 局限：memory bank 会随设计增长或需要压缩/淘汰；它不把状态存进参数，也不自动解决身份去重。
- 与本项目关系：是更简单、可解释的长视频基线。TTT 若不能在固定状态、延迟或精度上超过显式 bank，就缺少采用参数记忆的理由。

#### 15. Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- 论文：[arXiv:2312.00752](https://arxiv.org/abs/2312.00752)，2023-12-01，v2 2024-05-31。
- 代码：[state-spaces/mamba](https://github.com/state-spaces/mamba)。
- 阅读范围：摘要、方法概览和官方 README；未训练视频版本。
- 核心机制：input-dependent selective SSM，以固定递归状态线性扫描长序列。
- 局限：普通 Mamba state 由前向动力学更新，不做在线梯度/参数优化；多模态实体绑定仍需训练。
- 与本项目关系：这是区分“TTT 更新是否必要”与“一般固定状态递归模型是否足够”的核心架构控制。

#### 16. Titans: Learning to Memorize at Test Time

- 论文/官方介绍：[Google Research 页面](https://research.google/pubs/titans-learning-to-memorize-at-test-time/)，2025。
- 代码：本轮未确认官方完整训练代码，标为 **待确认**。
- 阅读范围：官方摘要页；未逐页通读论文。
- 核心机制：将短期 attention 与可在线学习的深层 neural long-term memory 组合，以 surprise/learned objective 写入。
- 局限：主要是语言长上下文证据；公开可复现性和视觉迁移条件需核实。
- 与本项目关系：覆盖“attention + neural long-term memory + surprise”的大框架。IdeaX 提案必须检查是否只是 Titans 的多模态移植。

#### 17. SAM 2

- 论文：[arXiv:2408.00714](https://arxiv.org/abs/2408.00714)，2024-08-01，v2 2024-10-28。
- 代码：[facebookresearch/sam2](https://github.com/facebookresearch/sam2)。
- 阅读范围：摘要和官方 README；未作为本项目 baseline 运行。
- 核心机制：streaming memory transformer 保存对象分割所需的历史特征，用提示、memory attention 和选择性帧记忆维持对象身份。
- 局限：针对 promptable segmentation 和对象 identity，不回答一般事件/行为问题。
- 与本项目关系：提供非参数化对象持续跟踪的强结构参考。若目标主要是物体位置/状态，先比较对象 memory bank，避免把 tracking 问题错误归因于 TTT。

### 4.5 补充的视觉 TTT 与记忆设计来源

#### 18. One-Minute Video Generation with Test-Time Training

- 论文：[arXiv:2504.05298](https://arxiv.org/abs/2504.05298)，2025-04-07；CVPR 2025。
- 代码：[test-time-training/ttt-video-dit](https://github.com/test-time-training/ttt-video-dit)。
- 阅读范围：摘要与官方 README；未复现。
- 核心机制：在 CogVideoX-5B 的局部 attention 外加入正向/反向全局 TTT layers，通过分阶段训练把生成长度扩到 63 秒。
- 局限：扩散生成可利用双向结构，和本项目严格因果的观察-预测不同；官方 README 说明 TTT-MLP 训练 kernel 只支持 H100，超出当前 4090 MVP。
- 与本项目关系：证明视觉生成中的 TTT state 可以承担跨场景一致性，但不能直接作为视频问答记忆证据。任何双向写入机制都必须改成因果版本后才可比较。

#### 19. MIRAS: It's All Connected

- 论文：[arXiv:2504.13173](https://arxiv.org/abs/2504.13173)，2025-04-17。
- 代码：本轮未从一手来源确认完整官方代码，标为 **待确认**。
- 阅读范围：摘要与 arXiv 方法概述；未逐页通读实验。
- 核心机制：把 sequence memory 统一拆为 memory architecture、attentional-bias objective、retention gate 和 memory learning algorithm 四个选择，并据此构造 Moneta、Yaad、Memora。
- 局限：主要验证语言与 recall 任务，没有现成视觉预测 checkpoint。
- 与本项目关系：提醒 IdeaX 分清“状态形式、写入目标、保留机制、优化算法”四个变量，避免把多个变化捆在一次实验中。仅从该四元组挑组件组合不自动构成新意。

#### 20. ViT³ / Vision Test-Time Training

- 代码与论文入口：[LeapLabTHU/ViTTT](https://github.com/LeapLabTHU/ViTTT)，官方仓库标注 CVPR 2026。
- 阅读范围：官方 README 和仓库结构；未下载权重或复现实验。
- 核心机制：系统研究视觉 TTT 的 inner model 与 inner-training 设计，覆盖分类、生成、检测和分割。
- 局限：没有本项目所需的长视频 VLM 问答与 episode-persistent readout。
- 与本项目关系：可作为“视觉 TTT block 如何设计”的来源；若 IdeaX 改 inner model，必须查明 ViT³ 是否已覆盖，而不能只与语言 TTT 比较。

#### 21. World-T³

- 项目/论文入口：[World-T³](https://adreamwu.github.io/World-T3/)，2026 项目页。
- 代码：项目页所示公开状态需在 IdeaX 执行时再次核验。
- 阅读范围：项目页摘要与方法图；未复现。
- 核心机制：将历史视觉特征通过 recursive least squares 写入 pose-keyed fast-weight scene memory，根据相机位姿读取，用于长程 revisit 一致的世界生成。
- 局限：依赖相机控制/位姿，目标是高分辨率实时生成；项目页性能基于 B200，不能外推到当前服务器。
- 与本项目关系：RLS、pose-keyed 寻址和 `leave room -> return` 条件值得作为结构参考；但家庭语义、所有权和行为预测不能由 pose memory 直接解决。

### 4.6 评测与数据来源

#### 22. Ego4D Forecasting

- 官方任务：[Ego4D Forecasting benchmark](https://ego4d-data.org/docs/benchmarks/forecasting/)。
- 代码：[EGO4D/forecasting](https://github.com/EGO4D/forecasting)。
- 阅读范围：官方 task schema 与 evaluator 说明；本地数据未获得。
- 任务：FHO Short-Term Anticipation 预测 noun、verb、box 和 time-to-contact；Long-Term Anticipation 预测未来 action sequence。
- 与本项目关系：最标准的第一人称短/长期预测评测，但数据访问是当前 blocker。

#### 23. EgoTempo

- 代码/数据说明：[google-research-datasets/egotempo](https://github.com/google-research-datasets/egotempo)，CVPR 2025，仓库 release 记录为 2025-03-19。
- 阅读范围：官方 README；未下载视频。
- 核心：500 个测试样例、10 类时间推理问题；作者声称过滤了单帧或常识即可回答的问题。
- 局限：依赖 Ego4D 媒体访问，且 open QA 常需 LLM judge。
- 与本项目关系：适合作为时间理解补充，不能替代可控预测标签。

#### 24. OVO-Bench

- 代码/数据：[JoeLeelyf/OVO-Bench](https://github.com/JoeLeelyf/OVO-Bench)，CVPR 2025。
- 阅读范围：官方 README 与当前 edition 数据说明；未下载。
- 核心：644 个视频、3,100 个查询，覆盖 backward tracing、real-time perception 和 forward active responding；官方页报告平均 query timestamp 约 263.42 秒。
- 局限：完整数据体积较大，任务同时混合回忆、实时感知和前向响应。
- 与本项目关系：适合 StreamTTT/ICL/显式检索的外部验证；应先做小子集并记录版本。

#### 25. EgoMemory / EgoMemReason

- 项目页：[EgoMemory](https://egomemory.github.io/)；对应 arXiv 检索标题为 [EgoMemReason, arXiv:2605.09874](https://arxiv.org/abs/2605.09874)，2026-05-11。项目页与论文当前命名不一致，引用时应核对最新版本。
- 代码/benchmark：项目页链接；本轮未下载。
- 阅读范围：项目页摘要、任务构成和主要分析；未逐页通读论文。
- 核心：week-long egocentric memory，500 个问题，entity/event/behavior 三类，平均每题 5.1 个证据片段、25.9 小时 backtracking；包含 cumulative state、event ordering/linking 和 behavior/activity pattern。
- 局限：规模和数据准备远大于本项目 MVP；行为题可能依赖高级 caption/agent pipeline。
- 与本项目关系：长期目标高度匹配，特别是 activity-pattern prediction；可作为后期有效性验证，不适合第一个训练 smoke test。

### 4.7 方法之间的继承、替代、组合与冲突

| 关系 | 说明 | 对 IdeaX 的约束 |
|---|---|---|
| TTT-LM -> LaCT -> Spatial-TTT | 从逐 token 参数状态到 large-chunk 高效更新，再到视觉/空间结构 | 不能把已有三步组合当新意 |
| LaCT -> E²-TTT | E²-TTT 针对 chunk-start 近似给出更精确状态转移 | 新 update rule 必须与它比较精度和吞吐 |
| LaCT -> Fast Spatial Memory | 加 Fisher/EMA elastic anchor 解决跨 chunk 漂移 | 简单 anchor/consolidation 已被覆盖 |
| Spatial-TTT -> StreamTTT | 面向 Qwen3-VL 流式长视频，把 fast state 与短 KV 协同 | “短 KV + fast state”不是空白 |
| Predictive TTT 与 FFN | FFN 已用 anticipated frame / surprise 调节更新窗口 | “预测误差门控”本身不是足够增量 |
| Predictive representation 与 TTT | AVT、V-JEPA 可在不更新测试时参数的情况下从 future loss 获益 | 必须有 matched frozen/recurrent control |
| TTT 与 Mamba/SSM | 两者都是固定状态长序列方案，只有 TTT 显式在线优化参数 | 需证明在线优化比学习好的递归动力学更有价值 |
| TTT 与显式 memory | MA-LMM/SAM2 保存可检索 token/对象状态，容量与解释性不同 | 需比较相同 memory bytes 和 latency，而不只比准确率 |
| 项目当前 Spatial 与 official slow weights | 本视频 fast state建立在已训练 slow weights 上 | `without_memory` 不是“无官方记忆/无先验”；必须使用 slow-only 术语 |

### 4.8 最接近本项目的工作与已被覆盖的设计

最接近当前代码的是 **Spatial-TTT**；最接近最终系统目标的是 **StreamTTT**；最接近“forecasting 控制在线适应”的是 **Frame Forgetting Network**；最接近“稳定长期 fast weights”的是 **Fast Spatial Memory**；最强的非 TTT 预测反事实包括 **AVT、StillFast、V-JEPA 2/2.1**；最重要的非参数记忆/固定状态反事实是 **MA-LMM、SAM 2、Mamba**。

IdeaX 在构思前必须把以下方案视为“已有覆盖，除非有明确新增机制”：

- 普通 TTT-MLP/TTT-Linear 替代 attention；
- Spatial-TTT + sliding window；
- fast weights + 4K 左右短期 KV；
- next-frame/future-latent auxiliary loss；
- prediction surprise 作为单一写入 gate；
- EMA anchor 或 Fisher/EWC 式 elastic consolidation；
- 直接把 V-JEPA encoder 接到现有 fast weights；
- 仅把 chunk size、rank、层数、learning rate 或采样率改成可学习超参数；
- 普通显式 memory bank、对象 token bank 或 selective SSM 的直接拼接。

这些组件仍可作为子模块，但 idea 必须说明新的耦合关系、训练信号、状态语义、因果更新或可验证理论预期，且需与最接近工作逐点查重。

## 5. 可探索的研究空间

### 5.1 必须满足的约束

1. **严格因果**：时刻 `t` 的 write 只能用 `<=t` 的帧、动作和自监督目标；未来片段只能作为离线 outer loss target，不能进入在线 fast-state 输入。
2. **公平计算**：至少报告参数量、可训练参数、训练 token/帧、视觉编码次数、训练 FLOPs 或可靠 proxy、峰值显存、fast-state bytes、KV bytes、ingest 吞吐和 query latency。
3. **同证据比较**：所有主 arm 使用相同视频 cutoff、帧选择、分辨率、问题和选项；额外检索或 caption 必须单独计入预算。
4. **隔离 slow 与 fast**：包含 full、slow-only/online-update-disabled、reset-every-chunk、random-update 或 shuffled-order 等消融。
5. **区分预测表征与参数记忆**：future loss 必须同时用于 matched frozen/recurrent baseline，才能判断 TTT 的边际贡献。
6. **防泄漏**：blind 文本、选项 shuffle、future masking、同实体/同模板 group split、caption access audit 和 query cutoff lint 必须自动化。
7. **先机制后规模**：第一阶段用 2B backbone、少量视频和封闭标签确定因果趋势；只有预注册的成功条件满足后才扩展到 8B 或长时数据。
8. **允许负结论**：如果 explicit/recurrent baseline 更强，应报告适用边界，而不是通过额外数据、更多帧或更大模型掩盖。

### 5.2 尚待回答的核心研究问题

#### A. 预测什么，才会形成可读的记忆？

- 下一帧像素、masked latent、未来 object-centric latent、下一动作、状态转移、time-to-contact 和最终 QA loss 各保留什么信息？
- 低层 future prediction 是否只学习相机运动/纹理连续性，忽略稀疏事件？
- 多 horizon loss 是否需要分别写入短期与长期 state？
- 能否用 information bottleneck 或 intervention target 让状态优先保留“改变未来分布”的事件？

#### B. 什么时候写、写多少、什么时候重置或巩固？

- prediction error、novelty、uncertainty、entity state change 和 scene boundary 哪一种 gate 真正相关？
- gate 的节省来自减少视觉编码，还是仅减少 fast update？两者应分开计费。
- 长期稳定需要 replay、anchor、低秩子空间、分层时间尺度还是显式稀疏路由？
- 新事件应覆盖旧状态还是保留多个版本？对位置、身份、习惯三类问题答案不同。

#### C. fast state 应表示什么？

- 通用关联矩阵、object slot、event transition operator、predictive sufficient statistic 或可查询 world model 哪个更适合家庭视频？
- 对象身份、位置、状态、人物关系、事件顺序是否需要结构化子空间？
- 固定容量下，如何度量每个对象/事件占用的有效 bits，而不是只统计参数量？

#### D. 写入与读取如何对齐？

- 视觉 key 与语言 query 是否需要显式 cross-modal alignment loss？
- 能否设计 counterfactual readout：改变问题但保持历史，测试 state 中是否存在对应事实？
- 生成错误来自 memory absence 还是 LM decoder bias？是否需要结构化 probe 与生成两套 head？

#### E. TTT 的优势来自在线优化还是一般递归计算？

- 在相同 state bytes、FLOPs 和训练 token 下，gradient-based TTT 是否超过 Mamba/GRU/recurrent transformer？
- 闭式/RLS/associative update 是否比 backprop-based update更稳定？
- online optimization 是否只在 test distribution shift 下有用，而在同分布记忆任务上无优势？

#### F. forecasting 与 memory 的任务定义是否需要重新拆分？

- 短期 anticipation 可能只需最近一帧；长期 behavior prediction 才需要累积记忆。
- “预测正确”可能来自数据先验，不代表使用了历史。需要 history-ablation、counterfactual history 与 matched blind prior。
- “记住过去”与“预测未来”可能冲突：精确 episodic trace 与抽象行为模式需要不同压缩策略。

### 5.3 可供 IdeaX 参考但不预设答案的方向

以下只是研究空间，不是已认可的候选 idea：

- 将 fast state 分成可快速覆盖的 episodic state 与缓慢更新的 predictive schema，并用不同 horizon 的 loss 学习路由；
- 让 future target 监督的不是内容重构，而是“哪一部分 fast state 对未来预测有因果贡献”的 credit assignment；
- 在 object/event change point 处触发稀疏参数写入，并用可逆或版本化更新保存状态历史；
- 训练 cross-modal query compiler，把语言问题映射到与视觉写入一致的 key/update 子空间；
- 将 TTT 作为局部可塑层嵌入 selective SSM，但必须说明超出直接串联的共同状态方程；
- 研究 fixed-memory Pareto：相同 bytes 下比较 fast weights、KV、object slots、compressed retrieval 和 SSM state；
- 用预测不确定性选择哪些帧值得高分辨率视觉编码，而不只控制 fast update；
- 把长视频目标重定义为 learned predictive sufficient state，而不是追求保存所有细节。

### 5.4 推荐的最小实验任务形态

第一阶段不建议继续扩大文字 PNG-card，也不建议以 counting 作为唯一主任务。更合适的最小机制实验应具备：

1. 真实或接近真实的 10–60 秒第一人称片段；
2. 明确 `observed_end_sec` 和 0.5/1/2/5 秒 prediction horizon；
3. 封闭集合的 next object、verb、state transition 或 destination 标签；
4. 同一历史的 counterfactual 或 block-swap 版本，使语言先验不能单独答对；
5. 每题标出关键证据帧，能测短期、远距和 scene-shift；
6. 2B 模型先跑 blind、recent-frame、equal-frame ICL、explicit retrieval、Mamba/recurrent、Spatial slow、Spatial fast 和 proposed；
7. 将 prediction accuracy 与 delayed recall/readout probe 同时报告，区分“会预测但没保存”和“保存了但不会生成”。

若 Ego4D 暂时不可得，可从 HomeSentinel 的有效 scene 时间线抽取小规模事件前缀，但标签必须由原视频和 cutoff 人工抽查，不能直接把自动 caption 当真值。也可优先下载公开且许可清楚的短 anticipation 子集。数据选择与许可 **待确认**。

## 6. 实验资源与限制

### 6.1 当前硬件快照

2026-10-20 在当前项目服务器执行 `nvidia-smi` 得到：

- 8 × NVIDIA GeForce RTX 4090 D；每卡显示 49,140 MiB 显存；
- 检查时 GPU 0–1 各占用约 38,471 MiB，GPU 2–7 基本空闲；占用是瞬时状态，不能视为长期独占保证；
- `/data` 总 7.0 TiB，已用 6.2 TiB，空闲约 481 GiB，使用率 93%；
- `/home` 所在盘空闲约 199 GiB。

因此可以并行不同 episode、不同 baseline、不同 seed 或不同数据 shard，但单个 episode 内的 fast state 存在时间依赖，不能简单把连续 chunk 数据并行而保持相同语义。可考虑 pipeline/tensor parallel 或 sequence-aware scan，但当前代码没有实现。

若另有 H20 96GB 机器可供本项目使用，当前服务器没有检测到它，故标为 **待确认资源**，不纳入 MVP 预算。

### 6.2 已确认模型与 checkpoint

| 模型/权重 | 路径 | 大小/状态 | 用途 |
|---|---|---|---|
| Qwen3-VL-2B-Instruct | `/data/quzitsix/models/Qwen3-VL-2B-Instruct` | 约 4.0 GiB，可运行 | 主 MVP backbone |
| Qwen3-VL-8B-Instruct | `/data/quzitsix/models/Qwen3-VL-8B-Instruct` | 约 17 GiB，可作为扩大验证 | 较大 backbone；训练预算需评估 |
| Spatial-TTT nano | `/data/quzitsix/models/Spatial-TTT-nano` | 目录约 4.7 GiB，正式结果记录 checkpoint hash | 当前 slow-weight 基线 |
| V-JEPA 2 ViT-L | `/data/leiyj23/ICRA2026/models/vjepa2/vitl.pt` | 项目文档记录约 303.9M 参数、5.13GB；官方源码另在 `/data/leiyj23/ICRA2026/assets/vjepa2-main-full` | frozen predictive representation 对照 |
| Qwen3.5-35B-A3B | 过去文档路径 `/data/hf_models/Qwen/Qwen3.5-35B-A3B` | 本次未发现该路径 | **待确认**，不能写入必备计划 |

### 6.3 软件与实现边界

- 项目 Python 包要求 Python >=3.11；核心 fast memory 仅依赖 PyTorch；视频/Spatial 额外使用 Transformers 4.57–<5、Accelerate、safetensors、PyAV、Pillow 等，见 [pyproject.toml](pyproject.toml)；
- 已有项目实验使用 `meowbench` conda 环境；本文件未启动训练或安装依赖；
- 当前 fast-weight 实现是纯 PyTorch，适合快速改公式和调试，但吞吐不代表官方 fused kernel 上限；
- 可以修改：TTT 层位置与维度、Q/K/V preparation、写入 loss/update、gate、state schema、readout、slow-weight trainer、LoRA targets、frame sampler、评测 runner；
- 可以冻结基础 VLM，仅训练新增层/adapter；2B 规模的模块训练、LoRA、probe 和多 episode 并行是合理 MVP；
- 从头训练基础 VLM、复现 V-JEPA 规模预训练、或按 StreamTTT/Spatial 原论文规模继续预训练，目前没有已确认的数据与训练预算；
- 底层 CUDA/Triton kernel 理论上可改，但当前项目没有相应实现和性能工程验证，是否投入标为 **待确认**。

### 6.4 已知时间与显存样例

- HomeSentinel 52 视频、60 秒/4 帧的 Spatial 运行约 5,339 秒；
- Q9 1080 秒、4 秒/8 帧的 dense Spatial ingest 约 3,843 秒；
- LoRA pilot 的 12-step suite 约 27.52 分钟，3-step 约 23.39 分钟，峰值 allocated 约 6.26 GiB；teacher/视觉处理占据大量固定时间；
- Spatial 2B fast state + momentum 约 505 MiB，但端到端显存还包括 2B VLM、视觉 tokens、slow layers、训练图和生成 KV；
- 当前没有 forecasting slow-weight 训练的实测时长、最大 batch 或 8 卡扩展效率，均为 **待确认**。

### 6.5 数据与训练预算边界

- **已本地可用**：HomeSentinel Asuka 约 9.25 小时 indoor 视频、Natural9 小子集、项目合成 fixture、Q9/SuperMemory pilots；
- **当前不可用**：Ego4D FHO 媒体，因此 FHO-STA/LTA 不能立即成为训练数据；
- **可候选下载但未确认**：EgoTempo、OVO-Bench 小子集、其他公开 anticipation 数据；需核实许可、下载大小和视频可得性；
- HomeSentinel caption、profile、object 文件含自动生成内容，不能当人工真值；
- 没有已确认的总 GPU 小时上限、实验截止日期、可下载数据上限或人工标注预算；IdeaX 应为每个 idea 给出明确 MVP 预算，并把这些参数标为待用户确认；
- 磁盘仅余约 481 GiB，不适合无筛选下载超大数据集或保留大量重复帧 cache。

### 6.6 现实可行的两阶段预算

这不是强制配置，而是 IdeaX 应优先满足的资源尺度：

1. **小规模机制验证**：Qwen3-VL-2B，冻结大部分 backbone；训练新增 TTT/predictor/readout，使用数百到数千个短 clip；1–2 张卡调试，最多 8 卡按 episode/shard 并行；完成 3 个以上 seed、强消融和精度-成本表。
2. **大规模有效性验证**：只有小实验达到预注册条件后，扩展到 Qwen3-VL-8B、更多真实长视频和更长 horizon；需要单独估算训练 token、GPU 小时、checkpoint/feature cache 空间与通信开销。

从头训练一个基础 VLM 或复现百万小时级 JEPA 不属于当前可行 MVP。

## 7. 请 IdeaX 完成的任务

### 7.1 总任务

基于以上项目事实，先进行中立的补充调研，再生成 **5 条有区分度、具体、可检验的 LLM/VLM 架构研究 idea**。项目偏好是“TTT 架构 + forecasting 视角的训练框架 + 合适数据”，但这只是优先方向，不是必须维护的假设。若调研表明某个问题更适合非 TTT 方案，允许重新定义问题或提出能证伪 TTT 必要性的架构。

只生成研究提案，不修改代码、不启动训练、不虚构实验结果，也不得宣称创新性已被证明。

### 7.2 工作流：调研 -> 构思 -> 查重 -> 自查 -> 卡片

#### 阶段一：补充调研

1. 先核验本文件列出的最接近工作，尤其是 Spatial-TTT、StreamTTT、FFN、Fast Spatial Memory、E²-TTT、Titans、V-JEPA 2/2.1、AVT、StillFast、MA-LMM 和 Mamba；
2. 搜索截至执行当日的作者论文、会议页面、官方代码和技术报告；搜索结果只能作为线索，必须打开原始来源；
3. 对每一条候选 idea，至少给出 3 篇最接近来源和 2 个反事实/替代架构来源；
4. 每篇记录标题、URL、发布时间/版本、作者、阅读范围、代码/权重地址、核心机制、适用条件、局限和与候选的关系；
5. 只读到摘要时写“仅摘要”；无法访问正文或代码时写明；未检索到某设计不能写成“从未有人做过”；
6. 优先寻找否定当前假设的证据，例如 frozen predictive model、显式检索或 SSM 在等预算下已经足够。

#### 阶段二：构思

生成至少 8 个内部候选，再基于重叠、可证伪性和资源可行性筛选为最终 5 个。最终候选之间不得只是：

- 同一 future loss 换不同层；
- 同一 gate 换输入特征；
- 同一 memory block 换数据集；
- 同一架构只改 rank、chunk、层数或 horizon；
- 把已有 V-JEPA/Spatial/Mamba 模块直接串起来。

五条应覆盖不同的核心机制，例如更新目标、状态结构、写读对齐、稳定性/巩固、固定预算选择或对 TTT 假设的反方向检验；具体分类由调研决定。

#### 阶段三：逐项查重

对每条 idea 建立“最近工作 -> 相同部分 -> 新增部分 -> 若删除新增部分会退化成什么已有方法”的查重表。至少回答：

- 是否退化成 Spatial-TTT、StreamTTT、FFN、Fast Spatial Memory、Titans、AVT/V-JEPA 或普通 retrieval/SSM？
- 新增部分是结构和信息流改变，还是普通损失/超参数拼接？
- 最近工作是否已经在附录、代码分支或后续版本实现？
- 若实质增量不能成立，淘汰该候选并从内部候选补一个。

#### 阶段四：五项自查

每张卡必须逐项给出 `通过 / 有条件通过 / 不通过` 与理由。自查通过只表示候选自评，不代表创新性或效果已验证。

1. **增量自查**：是否明确超出已有方法和普通模块拼接？
2. **因果与公平自查**：是否控制可见帧、参数量、训练 token、FLOPs、数据量和 future leakage？
3. **可证伪自查**：是否存在结果会明确推翻机制假设，而不是任何结果都能解释？
4. **资源自查**：最小实验是否能在已确认的 2B/8×4090D/有限磁盘条件内完成？
5. **评测有效性自查**：是否排除语言先验、模板、选项、同实体和 caption 泄漏，并选择能定位机制的指标？

任一候选在“增量”或“可证伪”上为不通过时，不得进入最终五条。

### 7.3 每条 idea 必须回答的九项内容

1. 要解释或解决的具体问题，以及方法成立所需条件；
2. 最近相关工作和实质差异，包含最接近方法的逐点查重；
3. 架构改动的位置、模块、参数状态和完整信息流；
4. 训练目标、数据构造、离线训练流程、在线写入流程和查询/推理流程；
5. 方法为何可能有效的机制假设，以及与替代解释的区别；
6. 最小可行实验、可复用代码/模型/数据、预计运行规模和成功门槛；
7. 强基线、关键消融、负对照和能够证伪方法的实验；
8. 参数量、可训练参数、state bytes、FLOPs、训练成本、ingest/query 延迟、显存、KV cache 和精度之间的权衡；
9. 主要失败条件、停止条件，以及从小规模机制验证扩展到大规模有效性验证所需的证据。

比较时必须控制参数量、训练 token、视觉帧、分辨率、数据量和总计算量。不能把更多计算、更多数据、更强 teacher 或更大 backbone 带来的收益直接归因于架构。

### 7.4 `idea_card.md` 的 IdeaX 六节结构

每个最终候选输出一份 `idea_card.md`，采用以下六节：

1. **问题、成立条件与核心假设**
   明确目标、任务边界、预测 horizon、为什么需要记忆，以及何种观察会推翻假设。
2. **最近工作、查重与实质增量**
   列最近工作对照表，解释新增机制为何不是调参或直接拼接。
3. **架构与信息流**
   给出模块位置、张量/状态语义、读写方程或伪代码、参数量和因果边界。
4. **训练与推理流程**
   列数据、loss、outer/inner loop、online update、reset/consolidation 和 query 流程。
5. **实验、基线、资源与权衡**
   分开写小规模机制验证与大规模有效性验证，含成功阈值、强基线、消融、成本表。
6. **失败条件、扩大条件与五项自查**
   列主要风险、停止规则、扩展前置证据，并完成五项自查。

卡片中不得使用“预计会显著提升”这类无依据措辞。应写成可检验预测，例如：“若机制成立，随着 evidence gap 从 2 秒增至 30 秒，proposed 相对 matched recurrent baseline 的增益应增加；若只在同分布短 horizon 上提高，则更支持表征/局部适应解释。”

### 7.5 每条 `literature.md` 的要求

每个候选同时输出独立的 `literature.md`：

- 检索日期、数据库/关键词和环境限制；
- 一手来源清单，每篇含标题、作者、时间、URL、代码、阅读范围；
- 核心机制、实验条件、局限；
- 与候选的相同/不同点；
- 支持与反对候选机制的证据；
- 尚未获得或尚未通读的材料；
- 明确写出“查重结论是当前检索范围内的判断，不代表创新性认证”。

### 7.6 最终 `overview.md` 的要求

最后生成 `overview.md`，至少包含：

1. 五条 idea 的一行核心机制；
2. 候选之间的实质差异，说明为何不是同一机制的变体；
3. 每条最接近工作与新增部分；
4. 最小实验、首选数据、主指标和能够证伪的结果；
5. 可复用的本项目代码/模型/数据路径；
6. 参数、state/KV、FLOPs、显存、训练 GPU 小时和延迟的估算表；
7. 主要失败条件与是否值得扩大；
8. 推荐优先级，但同时给出选择依据和不确定性。

建议输出目录：

```text
ideax_output/
  idea_01/
    idea_card.md
    literature.md
  idea_02/
    idea_card.md
    literature.md
  idea_03/
    idea_card.md
    literature.md
  idea_04/
    idea_card.md
    literature.md
  idea_05/
    idea_card.md
    literature.md
  overview.md
```

### 7.7 IdeaX 应优先解决的决策

最终五条提案应帮助项目回答以下决策，而不只是提出新模块：

1. 应先训练当前 Spatial write rule，还是改写为 forecasting-conditioned update？
2. forecasting 目标应该监督 representation、write gate、fast-state transition 还是 readout？
3. fast weights 相对短 KV、显式对象/事件 memory 和 selective SSM 的独特收益是什么？
4. 如何用最小真实视频实验区分长期记忆、短期适应、视觉识别和语言先验？
5. 哪一条方案能在 Qwen3-VL-2B 与当前 8×4090D 条件下最快被证伪，并为后续 8B/更大数据实验提供明确门槛？

最终检查标准：产物应让不了解本项目的研究 agent 清楚理解当前系统做什么、哪些结果是真实测得、哪些只是机制猜测、哪些设计已被近期工作覆盖、项目现在最缺什么证据，以及每个候选在现有资源下具体如何验证。
