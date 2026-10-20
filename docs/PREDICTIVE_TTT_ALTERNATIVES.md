# 预测视角下的 TTT、JEPA 与长期视觉记忆路线

整理日期：2026-10-20

> 本文保留各算法的详细调研。统一架构矩阵、当前全部证据、可比性边界，以及 counting
> 降级后的新实验 C 定义见
> [`VIDEO_MEMORY_ARCHITECTURE_COMPARISON.md`](VIDEO_MEMORY_ARCHITECTURE_COMPARISON.md)。
> 下文历史计划中的 `C-real` 现在统一称为 `C-temporal`。

本文回答三个问题：

1. 在 Spatial-TTT 之外，还有哪些真正适合长视频、在线视觉状态和未来预测的 TTT 方法；
2. JEPA 应当怎样加入实验，才不会把强视觉表征误写成长期记忆能力；
3. 针对实验 C 的 `location / mover / owner / state / next_action`，下一轮怎样做公平、可证伪的比较。

这里把“预测”分成三层。第一层是预测未来视觉表征，例如 V-JEPA；第二层是用预测误差决定何时写入，例如 FFN；第三层是让一个参数状态持续吸收历史，例如 Spatial-TTT、StreamTTT、E²-TTT 和 Titans。三者可以组合，但解决的问题不同。

## 1. 现有实现处在什么位置

当前仓库的 Spatial-TTT 路线是：

```text
视觉/问题 token -> 共享 Q/K/V
                 ├─ local sliding-window attention ───────────┐
                 └─ 视觉 span 做 3D depthwise convolution     │
                         -> LaCT SwiGLU fast-weight 分支 ──────┤
                                      两支相加 -> output projection

视频阶段：local attention + fast-weight 写入
问题阶段：local attention + 已有 fast-weight 只读，不更新 episode state
```

它与官方 [Spatial-TTT](https://github.com/THU-SI/Spatial-TTT) 的核心结构一致：滑窗注意力负责局部证据，非线性 fast weights 压缩长程视觉信息，空间预测卷积加入几何与时间连续性。仓库中的写入目标是 LaCT 的关联目标 `sum(f_W(K) * V)`，并不是“预测下一帧”或“重建当前帧”。这解释了一个重要边界：它可以学会把视觉 token 写进参数，但更新信号本身没有直接要求它保存“某人把某物从 A 搬到 B”这一可读状态，也没有直接要求它预测下一动作。

[LaCT](https://arxiv.org/abs/2505.23884) 的大块更新让非线性 fast weights 能高效运行，并允许 Muon、动量和权重归一化；代价是同一个大 chunk 内的 token 被近似当作集合，细粒度先后顺序主要交给局部注意力和位置编码。实验 C 恰好需要“旧位置被新位置覆盖”“所有者不随移动改变”“状态更新发生在动作之后”，所以更新顺序、覆盖规则和长期稳定性都必须单独测。

实验 C 的 runner 已经补齐 fixture schema v2 控制协议。视觉 runner 会分别渲染 `relocated`、`same_scene_control` 和 `filler_only_control`，三者使用相同 event-card 数量；`future_event` 只作为 sidecar target 保存，不属于观测流，runner 还会过滤 legacy fixture schema v1 manifest 中遗留的 `forecast` 事件。它包含 `blind`、Spatial slow-only 和 Spatial base-attention bypass，默认使用答案位置严格均衡的四选一题，并按 condition 与 query type 汇总；开放生成及其语义判分只保留为辅助诊断。symbolic runner 根据记忆中的 `moved_status + owner + final location` 和公开策略组合 `next_action`；三条件现已匹配事件数和 key-value 写入预算。因此 C-PNG 已能测场景迁移、普通 filler retention、未来标签泄漏和同 checkpoint 的 fast-memory paired gain。当前剩余限制是卡片仍以文字承载视觉信息，尚未接入 StreamTTT 和视觉 explicit table，也不能代替 C-temporal 的无文字场景或真实家庭视频。

## 2. 候选路线总表

| 方法 | 测试时更新什么 | 自监督/写入信号 | 跨窗口状态 | 对实验 C 的直接适用性 | 8×4090 的现实性 | 建议优先级 |
|---|---|---|---|---|---|---|
| **StreamTTT** | 每层 TTT 分支的 fast weights；短 KV cache 单独保留 | E²-TTT 式在线关联更新；离线同时训练实时 QA 与长视频 QA | 固定大小 fast weights + 4K 短 KV | 很高；原生是流式 VLM，可直接回答问题 | 已发布 Qwen3-VL 代码和权重；先做推理，完整再训练成本较高 | **P0：最直接架构对照** |
| **V-JEPA 2 / 2.1** | 推理时通常不改参数 | action-free 模型预测被遮挡的时空 latent；2-AC 预测动作条件下的未来 latent | 原版没有跨任意长视频的持久状态 | 中；冻结 probe 已完成，下一步必须另接显式或递归状态 | ViT-L 300M 可单卡；8 卡可按 episode 并行 | **P0：补 explicit/recurrent state** |
| **FFN** | 视觉 backbone/SSL head，按 surprise 有选择地更新 | 下一帧预测误差 + latent 差异；退出窗口时恢复旧适配 | 持续权重 + 小缓冲区 | 中高；很适合验证“预测误差 gate 是否改善写入” | 完整方法是任务型视觉模型；移植 gate 很轻 | P1：在外部基线后移植 gate |
| **E²-TTT** | fast weights、动量、衰减状态 | 保留逐 token 动态的闭式 chunk 更新 | 固定大小 fast weights | 高；能改善覆盖顺序和长程检索，但需接到 VLM | 官方有 340M/1.3B LM 代码；移植 update operator 可行 | **P1：替换 LaCT 更新器** |
| **LaCET / Fast Spatial Memory** | fast weights + EMA anchor + Fisher 重要度 | LaCT 更新后做弹性巩固 | 多 chunk fast state 和稳定 anchor | 高；直接针对长序列漂移和遗忘 | 需要移植 consolidate；完整 4D 模型要求相机位姿 | **P1：稳定性消融** |
| **Online Video TTT / FFN 完整版** | backbone 与 SSL head | MAE 重建，或 FFN 的下一帧预测 | 近期窗口和连续模型权重 | 中；强项是域适应、分割/深度，不是结构化 QA | 有官方 PyTorch；需重新做 QA readout | P1 |
| **TTT-Linear / TTT-MLP** | 作为 hidden state 的线性模型或两层 MLP | 每 token 自监督关联更新 | 固定大小 fast weights | 中；是 LaCT/E² 的原始轻量控制 | PyTorch 定义易运行，优化训练实现以 JAX/CUDA kernel 为主 | P1：更新基线 |
| **TTT-E2E** | 标准滑窗 Transformer 的普通权重 | 测试上下文的 next-token prediction；训练阶段 meta-learn 初始点 | 权重承载上下文 | 概念上高；目前是语言模型，没有视觉版本 | 官方 JAX checkpoint；直接迁移代价大 | P2：训练原则 |
| **Titans / MIRAS** | 深 MLP 神经记忆 | surprise 梯度、动量、可学习遗忘 | 长期神经记忆 + 短期注意力 | 高；适合选择性写入和稳定/可塑性研究 | 一手论文完整，未见官方视觉 checkpoint | P2：设计来源 |
| **ViT³** | 视觉 token 内的 TTT inner model | 视觉关联学习 | 序列内固定状态 | 中；验证视觉 TTT block 设计，但没有 VLM 问答头 | 官方 PyTorch与预训练模型已发布 | P2 |
| **World-T³** | pose-keyed fast-weight scene memory | 递归最小二乘吸收视觉特征 | 固定存储的 pose-keyed 参数记忆 | location/revisit 很强；owner/语义关系需另接头 | 当前系统报告单张 B200；不适合作为首轮 4090 baseline | P3：方法参考 |
| **Mamba/SSM、DeltaNet 类** | 固定递归隐藏状态，不做梯度更新 | 训练时学好固定 recurrence | 固定大小向量/矩阵状态 | 中；是必要的“无测试时改参”控制组 | 小模型容易部署 | P1：机制对照 |

## 3. 最值得先做的三个对照

### 3.1 StreamTTT：和当前项目最接近的外部架构

[StreamTTT 论文](https://arxiv.org/abs/2608.13416) 和[官方实现](https://github.com/zeyun-zhong/StreamTTT)把 Qwen3-VL 每个 decoder layer 分成两条并行路径：

```text
recent tokens -> 4K sliding KV cache -> 当前感知
older stream  -> E²-TTT fast weights -> 长期记忆
                         \             /
                      tanh(alpha) gate
```

这比 Spatial-TTT 更直接对应项目目标。Spatial-TTT 的训练重点是长程空间证据；StreamTTT 同时用离线长视频 QA 和“答案第一次变得可用时提问”的实时 QA 训练，因此 readout 目标本来就是视频问答。它也明确把近期感知和长期记忆分开，可以检查我们目前遇到的“有记忆和无记忆回答相似”究竟来自 fast weights 没写进去，还是读取时当前语言先验淹没了记忆。

官方仓库已经发布代码与模型权重。其最小 toy recipe 是 2B backbone、16 帧、rank-4 LoRA，在 RTX 3090 上约占 12 GB；这个数字只能证明小型 smoke test 可在 4090 上运行，不能外推完整 4B 长视频训练的显存。合理顺序是：

1. 用发布权重跑 C-PNG 的固定 12 个 episode；
2. 每张 4090 跑独立 episode，做 8 路数据并行；
3. 只有推理结果表明 fast memory 有读出能力，再考虑 8 卡 LoRA/SFT；
4. 训练时保持与 Spatial-TTT 相同帧、分辨率、问题和答案集合。

公平比较必须包含 StreamTTT 的 fast-memory bypass；否则“发布模型比 nano checkpoint 强”无法归因到架构。

### 3.2 V-JEPA 2：强预测表征，但不是长期记忆

[V-JEPA 2](https://arxiv.org/abs/2506.09985) 的 action-free 预训练在 latent space 预测被遮挡的时空目标表征；[官方仓库](https://github.com/facebookresearch/vjepa2)提供 ViT-L 300M、ViT-H 600M、ViT-g 1B 等模型，并提供 EPIC-KITCHENS action anticipation probe。V-JEPA 2-AC 进一步用动作条件预测未来 latent，用于图像目标规划。[V-JEPA 2.1](https://arxiv.org/abs/2603.14482)增强了 dense predictive loss、深层自监督和时空一致特征。

这条路线应该回答：

> Spatial-TTT 的问题是记忆机制失败，还是 Qwen3-VL-2B 的视觉动态表征本来就不够好？

原版 V-JEPA 的 encoder/predictor在推理时通常冻结，也没有无限长的 episode state。直接把它叫作“JEPA memory”会混淆变量。实验应拆成三臂：

| 条件 | 作用 |
|---|---|
| `V-JEPA frozen + linear/MLP probe` | 测短期状态与动作预测的表征上限 |
| `V-JEPA + explicit last-write table` | 测强视觉表征配合可解释状态记忆 |
| `V-JEPA + fixed-size recurrent/fast-weight state` | 测同一视觉表征下隐式压缩记忆 |

当前 C-PNG 已完成官方冻结 V-JEPA 2 ViT-L probe。正式协议会对规范化后的完整观测事件流做哈希，并在测试每个 episode 时排除其整个 exact-content group；也就是 **leave-one-content-group-out** 监督 readout，而不是只排除同名 episode。三个 condition 分别有 10、10、9 个 content group；跨 condition 将任一条件下内容相同的 episode 合并后，12 个 episode 形成 **9 个 exact-content clusters**。这些 cluster 仍共享 `%3` 标签周期，不能视作 9 个独立随机样本。正式结果为 V-JEPA 2 **126/180 = 0.7000**，相同分组和 readout 下的低分辨率 pixel control 为 **132/180 = 0.7333**。此前普通 leave-one-episode-out 得到的 142/180 和 148/180 会让 exact duplicate 同时进入训练与测试，现只作为泄漏诊断保留。

这个 probe 使用跨 episode 标签监督，没有持续在线状态，因此既不是 TTT，也没有证明 JEPA 具备长期记忆，更不能与 ICL 或 Spatial 的 zero-shot MCQ choice accuracy 直接排名。pixel control 更高还说明文字、颜色、固定布局和生成器规律已经提供强低层线索。下一步应把同一个 encoder 接到 explicit last-write table 与固定大小 recurrent/fast-weight state，并迁移到 C-temporal 的无文字、照片级或真实视频。ViT-L/256 可单卡运行；ViT-g/1B 只适合作为后续表征上限。

### 3.3 FFN：把“未来预测误差”变成写入门控

[Forget, Anticipate and Adapt / FFN](https://arxiv.org/abs/2606.26515)及其[作者代码](https://github.com/rajatmodi62/ffn)是当前最贴近“预测驱动 TTT”的方法。它用下一帧预测误差和 latent 相似度构成 surprise，只在 surprise 超过动态阈值时更新；同时通过 Memory Restoration Mechanism 处理滑窗退出帧的旧适配。作者将每步计算限制到退出帧、当前帧和预测帧三帧，并在多小时视频上评测。

对本项目，最先需要的不是把整个 FFN 分割系统搬进来，而是只移植写入控制：

```text
z_t = visual_encoder(x_t)
z_hat_t = predictor(memory_{t-1})
s_t = latent_distance(z_t, z_hat_t)

if s_t >= running_mean(s) + running_std(s):
    Spatial/E² fast-weight update
else:
    read only
```

搬家事件、状态改变、开关容器等应该产生高 surprise；静止画面和重复走动应被跳过。这个消融可以直接检验：密集 4s/8f 输入的收益是否只是更多重复写入，以及减少更新能否同时提高 C 准确率和速度。

不过 surprise 不能只用像素差。第一人称相机快速转动会造成巨大像素变化，却不一定改变世界状态。建议同时记录：

```text
visual surprise   = 1 - cos(z_t, z_hat_t)
object surprise   = object/relation head 的预测误差
write gate        = max(visual surprise, object surprise) 超阈值
```

## 4. 值得移植进 Spatial-TTT 的更新规则

### 4.1 E²-TTT：保留顺序的高效 chunk 更新

[E²-TTT](https://arxiv.org/abs/2608.21308)推导了闭式状态转移：在“梯度都在 chunk 起始 fast weights 上计算”的共同近似下，闭式更新可以精确复现逐 token recurrence 的 chunk-end 权重和动量，同时保留 token-dependent 学习率、动量和衰减。[官方代码](https://github.com/zeyun-zhong/E2-TTT)发布了 340M 和 1.3B 的 MLP/SwiGLU checkpoints，并提供数值等价性验证程序。

这比单纯减小 LaCT chunk 更适合实验 C：位置覆盖和状态改变依赖事件顺序，而大块 LaCT 的一次聚合更新容易弱化“最后一次写入”的优先级。建议先保持 Spatial-TTT 的 Q/K/V、3D convolution、SWA 和 readout，只替换 `FastWeightState` 的 update operator。这样对照只改变 fast-weight dynamics。

需要报告：

* 同一 episode 的 write count、chunk size 和视觉 token 数；
* location/state 的 latest-value accuracy；
* owner 的 retention accuracy；
* 交换 move 前后顺序后的 prediction flip rate；
* 每个 chunk 的 fast-weight norm、更新量和 GPU 时间。

### 4.2 LaCET：给 fast weights 增加稳定锚点

[Fast Spatial Memory / Elastic TTT](https://fast-spatial-memory.github.io/)指出，LaCT 连续多 chunk 更新会发生 fast-weight drift、过拟合和遗忘。LaCET 为每层维护 EMA anchor 和在线 Fisher-style importance，在每次 LaCT 更新后把重要参数软拉回 anchor，使重要旧知识更稳定、非关键参数继续适配。

这正好对应 C 的两类冲突目标：

* `owner` 应跨搬家事件保持；
* `location` 和 `state` 应被新事件覆盖。

可以先在现有 `SwiGLUFastWeightMemory` 上实现一个小型消融：

```text
W'      = lact_update(W, chunk)
F_t     = beta_F * F_{t-1} + (1-beta_F) * grad(chunk)^2
anchor  = beta_A * anchor + (1-beta_A) * W'
W_t     = W' - lambda * F_t * (W' - anchor)
```

完整 FSM 以 posed images、Plücker rays 和时间戳做 4D novel-view reconstruction，不是任意视频 QA 的即插即用模型；当前家庭视频如果没有可靠相机位姿，不应把其完整模型作为第一轮 baseline。LaCET 的 consolidate operator 可以单独公平移植。

### 4.3 TTT-E2E：把写入目标改成真实预测目标

[TTT-E2E](https://arxiv.org/abs/2512.23675)不再依赖一个预设的关联记忆层，而是让普通 sliding-window Transformer 在测试时用 next-token prediction 继续学习，并在训练阶段通过 meta-learning 学到一个适合测试时继续学习的初始化；[官方 JAX 实现](https://github.com/test-time-training/e2e)发布了 125M、1B、3B checkpoints。

对视觉项目最有价值的是训练原则：fast memory 的初始化和更新目标都应该为未来预测服务。视觉版本可以把 next-token loss 替换成：

```text
L_future = 1 - cos(predictor(m_t), stopgrad(z_{t+1}))
L_state  = CE(readout(m_t), structured_state_{t+1})
L_keep   = old-query retention loss
```

只用 `L_future` 可能保存运动趋势，却丢掉低频但重要的所有权；只用 QA `L_state` 又容易依赖 teacher caption。三项联合，并通过 episode 外层梯度学习初始化，更符合 C 的目标。

## 5. 更远的设计参考

### 原始 TTT layers

[Learning to (Learn at Test Time)](https://arxiv.org/abs/2407.04620)把 RNN hidden state 本身换成一个线性模型或两层 MLP，并通过每个 token 的自监督目标更新这个模型；[官方 PyTorch 定义](https://github.com/test-time-training/ttt-lm-pytorch)适合核对算法，[官方 JAX 代码](https://github.com/test-time-training/ttt-lm-jax)用于论文训练。它建立了“参数就是递归状态”的基本对照，但逐 token 小更新的 GPU 利用率和非线性状态规模后来才由 LaCT、E²-TTT 改进。对实验 C，原始 TTT-Linear 可以作为与同维度 Mamba/DeltaNet 匹配的轻量基线；TTT-MLP 则检验非线性 state capacity 是否必要。

### TTT-Video generation

[One-Minute Video Generation with TTT](https://arxiv.org/abs/2504.05298)及其[官方实现](https://github.com/test-time-training/ttt-video-dit)在预训练 CogVideoX-5B 的局部 attention 之外加入双向 TTT layers，逐步把训练长度从 3 秒扩到 63 秒。它证明 TTT state 能支撑跨场景生成一致性，但目标、因果方向和硬件都与 C 不同：生成模型可访问双向全局结构，官方 TTT-MLP 训练路径只支持 H100，且没有问答 readout。它适合作为“预测生成中的 TTT”证据，不应进入首轮 4090 端到端 baseline。

### Titans / MIRAS

[Titans](https://research.google/pubs/titans-learning-to-memorize-at-test-time/)使用深 MLP 作为长程神经记忆，并用 gradient surprise、momentum 和自适应 forgetting 决定信息如何进入记忆。[MIRAS](https://arxiv.org/abs/2504.13173)把设计拆成四个选择：memory architecture、attentional-bias objective、retention gate 和 memory learning algorithm。

它们提示我们不要把“TTT”绑定为一种更新公式。C 的五类关系可采用不同 retention：所有权使用慢衰减，位置/状态使用强覆盖，mover 使用事件型记忆，next-action 使用预测头。当前缺少官方视觉 checkpoint，因此适合作为设计与消融来源，不适合作为近期端到端分数对照。

### ViT³

[ViT³](https://github.com/LeapLabTHU/ViTTT)系统比较视觉 TTT inner-module 和 inner-training 设计，并给出纯 TTT 的视觉架构，覆盖分类、生成、检测和分割。它能作为“Qwen attention 里嵌 LaCT”之外的视觉 block 对照，但没有长期视频 QA readout；若使用它，应固定同一个 C 状态头，而不是比较其原始分类分数。

### Online TTT on Video Streams

[Test-Time Training on Video Streams](https://video-ttt.github.io/)让当前模型从上一帧适配后的模型继续出发，在当前帧和近期小窗口上用 MAE 等自监督任务训练。它证明时间局部性对真实视频在线适配很重要，但目标是分割、深度或颜色等逐帧任务，没有单独的长期关系 memory。它适合充当“普通在线适配”控制：如果它改善当前视觉识别却不能回答搬家后的 owner/location，说明 C 测到的确是关系记忆，而不是域适应。

### World-T³

[World-T³](https://adreamwu.github.io/World-T3/)把历史视觉特征用递归最小二乘写入 pose-keyed fast weights，并根据相机位姿读取，用于长程 revisit 一致的视频世界模型。它对“回到旧房间还能否恢复场景”非常相关，但当前目标是 1280×704 实时生成，项目页报告的速度基于单张 B200；它还依赖相机控制/位姿。近期可借用两点：RLS 更新比普通 SGD 更适合精确键值写入；C 后续应加入 `leave room -> return to room` 的 revisit 条件。

### 固定递归状态：Mamba/SSM

[Mamba 官方实现](https://github.com/state-spaces/mamba)代表固定递归状态路线。它不在测试时做梯度更新，因此可以作为关键机制控制：相同视觉 encoder、相同状态宽度、相同监督下，如果 Mamba/DeltaNet 已经和 TTT 一样好，收益可能来自“有一个流式状态”，而不是测试时训练本身。

## 6. 实验 C 各能力需要什么

| 查询 | 真正需要的状态 | JEPA 单独 | Spatial/LaCT | StreamTTT | 显式 memory |
|---|---|---:|---:|---:|---:|
| `location` | object identity + latest location，允许覆盖 | 弱 | 中 | 中高 | 强 |
| `mover` | actor-object-event 的时间绑定 | 弱 | 中 | 中高 | 强 |
| `owner` | 稳定关系，不能被搬家动作覆盖 | 弱 | 中，易漂移 | 中高 | 强 |
| `state` | latest state 与动作结果 | 中 | 中 | 中高 | 强 |
| `next_action` | 当前状态 + learned dynamics | 强，尤其 2-AC | 低到中 | 中 | 取决于 predictor |

“JEPA 单独”指冻结 encoder/predictor，不附加跨视频 memory。表中的强弱是结构能力判断，不是已有 C 实验结果。JEPA 对 `next_action` 的优势只有在动作词表或 action-conditioned predictor 与数据匹配时才成立。

实验 C 的核心不是让任一方法在所有列都赢，而是把失败拆开：

```text
看不清对象          -> encoder 问题
知道变化但忘了旧关系 -> retention 问题
写入了但问不出来     -> addressing/readout 问题
只会复述、不会预测   -> dynamics objective 问题
```

## 7. 下一轮公平实验矩阵

### 7.1 C-PNG：PNG-card 控制协议已实现（fixture schema v2）

当前 runner 已实现三条件、默认隐藏 future card、严格均衡 MCQ、`blind`、Spatial slow-only 和 base-attention bypass，并保留 open-answer 语义判分作为诊断。正式 zero-shot VLM 结果统一使用 12 个 episode、三个 condition、每个 episode 五类问题和逐题 `separate` MCQ，因此每个完整 arm 有 180 道题：

| arm | layout / visual state | correct / 180 | choice accuracy | parse rate | 解释 |
|---|---|---:|---:|---:|---|
| ICL | separate；16 张卡随每道题重放 | **89** | **0.4944** | 0.9833 | 相对同一 base model 的 blind 为 +41/180；三次解析失败按错误计入 |
| blind | separate；无视觉输入 | 48 | 0.2667 | 1.0000 | 接近严格均衡随机线 0.25 |
| Spatial fast | separate；官方 slow checkpoint + episode fast state | 48 | 0.2667 | 1.0000 | 当前 fast-memory arm |
| Spatial slow-only | separate；同 checkpoint，写入后丢弃 episode fast state | **51** | **0.2833** | 1.0000 | 与 fast arm 的直接配对基线 |
| Spatial base-attention bypass | separate；绕过 Spatial branch | 49 | 0.2722 | 1.0000 | 同 Spatial checkpoint 的结构分支对照 |

五臂正式汇总见 `runs/experiment_c_visual_v2_ep12.summary.json`。ICL−blind 为 **+0.2278**；按 9 个 exact-content clusters 枚举的 sign-flip `p=0.00781250`，支持“同一个 base model 能利用卡片”这一 fixture 内结论。由于 cluster 之间仍共享确定性生成规律，这个 p 值只作机制诊断，不代表独立同分布总体推断。Spatial fast 比同 checkpoint 的 slow-only 少答对 3/180，差值为 **-0.0167**、content-group `p=0.5`；相对 base-attention bypass 也少 1/180，差值为 **-0.0056**、content-group `p=1.0`。因此完整 12-episode 结果没有显示正的 episode fast-state 增益。

这些 arm 仍有两个重要协议差异。每个 episode-condition 都只有 16 张唯一观测卡，但 separate ICL 为五道题各自重新编码一遍，共 **80 次视觉帧呈现**；Spatial 只 ingest 一次 16 张卡，随后进行五次纯文本读取。ICL/blind 使用原始 Qwen3-VL base model，三个 Spatial arm 使用加载官方 Spatial-TTT slow checkpoint 的模型。因此 ICL 与 Spatial 的绝对分数不是严格的 memory 因果对照；可直接解释的是 ICL−blind，以及同 Spatial checkpoint 的 fast−slow-only。旧 batched Spatial 调用在生成一个选项字母后立即 EOS，只是输出格式诊断，不属于性能表。

冻结 V-JEPA 2 ViT-L 也已在完整 C-PNG 上运行。正式的 leave-one-content-group-out 监督 readout 为 **126/180 = 0.7000**，相同分组和 readout 的低分辨率 pixel control 为 **132/180 = 0.7333**。按 condition 分开看，V-JEPA 2 在 `relocated / same_scene_control / filler_only_control` 上分别为 **38/60、35/60、53/60**，pixel control 分别为 **40/60、40/60、52/60**。普通 leave-one-episode-out 的旧值 142/180 和 148/180 存在 exact-duplicate train/test 泄漏，现只用于解释协议修正。两个正式 arm 测的仍是监督表征可读性，没有持续在线状态；不能与上表 zero-shot MCQ 直接排名，也不能把分数解释为长期记忆。

各臂的实现状态如下：

| arm | memory/readout | 状态与目的 |
|---|---|---|
| blind | 只给问题 | **12-episode separate MCQ 已完成**；测语言先验 |
| ICL-all | 当前 condition 的全部卡片 | **12-episode separate MCQ 已完成**；视觉上下文上限，但视觉呈现预算更高 |
| visual explicit last-write table | 结构化更新 | symbolic 已实现，视觉尚未接入；oracle-like 状态基线 |
| Spatial fast | 官方 slow checkpoint + episode fast state | **12-episode separate MCQ 已完成**；现有 fast-memory 主线 |
| Spatial slow-only | 同 checkpoint，丢弃 episode fast state | **12-episode separate MCQ 已完成**；fast-state 直接对照 |
| Spatial base-attention bypass | 同 checkpoint，绕过 Spatial branch | **12-episode separate MCQ 已完成**；结构分支对照 |
| V-JEPA frozen | 冻结 encoder + 监督 readout | **12-episode probe 已完成**；表征对照，不是 TTT |
| V-JEPA + explicit/recurrent state | 显式表或固定大小递归状态 | 尚未接入；下一轮状态记忆对照 |
| StreamTTT + memory/bypass | 发布模型 | 尚未接入；P0 外部 TTT 对照 |

三个 condition 现在分别生成等长的实际卡片流。`same_scene_control` 把场景切换改为 same-scene marker；`filler_only_control` 用同预算的无关绑定替换目标 move，所以不会把更少写入误当成更好 retention。`next_action` 不读取 forecast，而是从已发生的状态和公开策略预测。

fixture schema v2 仍有必须在 C-temporal/schema v3 消除的捷径和混杂因素：`moved_object`、最终位置和最终状态按固定三周期共同变化；filler 卡会显式出现 `control distractor`、`actor: control` 和 `control_records` 等内部控制标记；`relocated` 与 `same_scene_control` 除场景条件外还存在 `living room shelf`/`shelf`、`side table`/`table` 的位置措辞差异。这些规律可被监督 readout 或 OCR 利用，所以当前 C-PNG 适合做管线与机制诊断，不能作为真实家庭视频长期记忆能力的最终结论。

C-PNG 的主指标是严格均衡的 choice accuracy。semantic accuracy、token-F1、exact 与 substring 只用于开放生成兼容和错误分析。当前方法层面的优先缺口是 StreamTTT memory/bypass、V-JEPA 显式或递归状态，以及能摆脱文字卡 OCR 捷径的 C-temporal 无文本视频；visual explicit table 仍是需要保留的机制上限控制。

### 7.2 C-temporal：无文字可控视觉场景

把相同 manifest 渲染成无文字图像或短视频：不同颜色/形状的物品、人物、房间和容器，以视觉动作表达搬动、开合、使用。每个 episode 生成三条条件，除唯一自变量外保持帧数和动作数相同。

推荐的第一版矩阵：

| 编号 | encoder | long state | update/gate | readout |
|---|---|---|---|---|
| C0 | Qwen3-VL | 无 | 无 | closed-choice VLM |
| C1 | Qwen3-VL | ICL | 无 | closed-choice VLM |
| C2 | Qwen3-VL | explicit table/retrieval | last-write | 同一 choice scorer |
| C3 | Qwen3-VL | Spatial-LaCT | write-all | 同一 choice scorer |
| C4 | Qwen3-VL | Spatial-LaCT | FFN surprise gate | 同一 choice scorer |
| C5 | Qwen3-VL | E²-TTT | write-all | 同一 choice scorer |
| C6 | Qwen3-VL | LaCET | write-all + consolidate | 同一 choice scorer |
| C7 | V-JEPA 2 ViT-L | explicit table | last-write | 固定 MLP heads |
| C8 | V-JEPA 2 ViT-L | GRU/Mamba state | learned recurrence | 固定 MLP heads |
| C9 | StreamTTT | E² fast weights + recent KV | 官方协议 | closed-choice VLM |

所有臂必须满足：

* 相同 episode、时间边界、帧索引和分辨率；唯一观测帧数与重复视觉呈现次数分别报告并尽量匹配；
* ingest 阶段没有问题、答案、caption teacher 或未来帧；
* 每个 episode 前重置所有 KV、fast weights、optimizer/momentum 和显式表；
* `next_action` 只在 query 时刻之前的 prefix 上评测；
* VLM 方法用同一选项打分协议，feature 方法用同一训练 split 的固定 probe；
* 分别报告同 checkpoint 的 fast-state、slow-only 和 branch-bypass paired difference，不只报告不同 checkpoint 的总分。

### 7.3 主要指标

主表按 condition × query type 报 accuracy，并另外记录：

```text
relocation penalty = acc(relocated) - acc(same_scene)
filler retention   = acc(filler_only) - acc(no_filler)
fast-state gain    = acc(fast) - acc(slow-only)
branch gain        = acc(slow-only) - acc(base-attention bypass)
latest-write gain  = acc(after_move) - acc(before_move_control)
```

对预测部分增加：

* `next_action` top-1 / top-k；
* future latent cosine error；
* prediction-error calibration：surprise 是否在真正状态变化时升高；
* write precision/recall：发生状态变化的帧是否触发写入；
* 每 episode 的 update 数、耗时、峰值显存和 memory bytes。

必须加入两种反事实：交换 move 前后事件顺序，以及保持画面相同但替换 owner/mover 身份。模型若答案不随反事实翻转，说明它仍在使用语言/视觉先验。

## 8. 8 张 4090 的执行安排

下一轮不需要模型并行，按 episode/seed 做数据并行更简单，也更容易复现。优先把 GPU 用在尚缺的外部架构、状态头和无文字数据轨道：

```text
GPU 0-1: StreamTTT memory / bypass
GPU 2-3: C-temporal Qwen ICL / Spatial paired runs
GPU 4: V-JEPA ViT-L + explicit state
GPU 5: V-JEPA ViT-L + recurrent state
GPU 6: C-temporal frozen V-JEPA / pixel controls
GPU 7: visual explicit table / Spatial control completion / failed-run retry
```

每个 worker 都写独立 JSONL，最后按 `episode_id/query_id` 做 paired join。不要让多进程同时追加同一个结果文件。V-JEPA 2 ViT-L 只有 300M 参数，适合单卡；本机每张 4090 D 由 `nvidia-smi` 报告 **49,140 MiB** 显存，但 StreamTTT 完整模型能否在该预算内保持目标帧数仍应先记录实测峰值，再决定单卡、两卡分片或减小输入。官方明确给出的硬件事实只有 2B toy 在 3090 上约 12 GB。

如果需要训练，小规模 probe、gate 和 update-operator 消融可各自单卡；StreamTTT/Spatial 的 LoRA 或 slow-weight SFT 再使用 8 卡 DeepSpeed。没有必要在确认 C-temporal 有可学习信号前重训完整 JEPA 或 4B VLM。

## 9. 推荐顺序与停止条件

建议按以下顺序推进：

1. **P0：部署 StreamTTT 发布权重并加入 memory/bypass paired run**。这是最直接、变量最少的外部长期视频 TTT 对照。
2. **并行完成 C-temporal 无文本视频轨道和 V-JEPA 状态接口**。C-temporal 去掉文字卡 OCR 捷径；V-JEPA 保留 frozen encoder，分别接 explicit last-write table 与固定大小 recurrent/fast-weight state。
3. **在 C-temporal 上复用已经验证的 frozen V-JEPA/pixel probe，并加入第 2 步的两个状态臂**。所有 feature arm 使用相同训练 split 和 readout，所有 VLM arm 使用相同 closed-choice scorer。
4. **给 C-PNG 和 C-temporal 保留 visual explicit last-write table**。它不是主要新架构，但能给出视觉识别后的可解释状态上限。
5. **在外部基线稳定后给 Spatial 写入加入 FFN surprise gate，再分别移植 E² update 和 LaCET consolidate**。三者解决的问题不同，应逐项消融。
6. 只有 `next_action` 仍明显落后、而 state 查询已经可靠时，再训练 future-latent/action-conditioned predictor。

可以使用这些停止条件避免无效大训练：

* 若 `memory gain <= 0` 且 fast-state readout probe 也接近随机，停止扩大数据，先修写入/寻址；
* 若 explicit memory 很强、所有隐式 state 都弱，结论是压缩记忆容量或写入目标不足；
* 若 V-JEPA + explicit state 强于 Qwen + explicit state，优先修视觉表征；
* 若两种 encoder + explicit state 都弱，优先修数据、动作可见性和答案协议；
* 若 surprise gate 减少至少一半更新且准确率不降，再扩展到真实长视频；
* 只有相同 slow checkpoint 的 `fast - slow-only` 显著为正，才把提升归因于 episode fast state；`slow-only - base-attention bypass` 另行衡量 Spatial branch 本身。

## 10. 结论

近期最有价值的外部架构是 StreamTTT，因为它与当前 Qwen3-VL、流式视频 QA 和 fast-weight 长期记忆最对齐。JEPA 应作为预测视觉表征对照，并配相同的显式或递归状态头；原版 JEPA 不能单独证明长期记忆。算法上，FFN 的 prediction-surprise gate、E²-TTT 的顺序保持更新和 LaCET 的弹性巩固，分别对应当前系统最明显的三类风险：重复无效写入、chunk 内顺序弱化和多 chunk 漂移。

最值得尝试的新主线可以写成：

```text
V-JEPA/Qwen visual state
    -> future-latent predictor
    -> calibrated surprise gate
    -> E²-TTT fast weights
    -> LaCET-style consolidation
    -> recent KV + structured state/readout heads
```

实验时必须逐项增加。当前 Spatial fast/slow-only 与 frozen V-JEPA/pixel 已经给出第一批锚点；下一步先补 StreamTTT memory/bypass、V-JEPA explicit/recurrent state 和 C-temporal，再判断哪一个部件真正改善长期视觉状态记忆。
