# 面向未来预测的 TTT baseline 调研与实验计划

整理日期：2026-10-11

本文把项目问题从“TTT 能不能记住长视频”收窄为一个更容易证伪、也更适合小模型的问题：

> 在相同视觉输入预算下，TTT 的 fast weights 是否比 in-context learning 更擅长预测未来的物品状态、人物动作和事件结果？它在哪些时间跨度、场景变化和交互类型上开始失效？

这里的重点是**预测**，不是把已经发生的事情复述出来。评测时只能把查询时刻之前的视频交给模型，未来片段、未来标签和答案不能进入写入过程。

## 1. 先固定研究假设

我们要验证的不是“TTT 在所有 VideoQA 上都更好”，而是下面三个更具体的假设：

1. **短期预测**：给定动作发生前的第一人称观察，预测即将交互的物品、动作动词或物品状态变化。
2. **长程预测**：给定较长历史，预测未来动作序列、事件计数、物品的最后状态或下一次交互。
3. **场景变化下的预测**：人物、相机视角或房间发生变化后，模型仍能利用之前形成的对象—人物—位置关系。

每个假设都要同时比较：

* `blind`：只看当前问题或当前帧，不读历史；
* `ICL`：把历史帧放进当前上下文，但不更新参数；
* `explicit memory`：检索帧、文本、轨迹或场景图；
* `TTT`：把历史压进 fast weights，再只用当前查询读取；
* `TTT + gate`：允许模型跳过无信息帧、在场景变化时更新或重置。

真正支持 TTT 的证据应当同时满足：

* TTT 相比 ICL 的收益随历史长度或场景变化增加而变大；
* 在相同帧数、分辨率和模型大小下成立；
* 当前识别能力没有明显下降；
* 通过 permutation、未来帧泄漏和文本先验对照后仍然成立。

## 2. 数据集与任务优先级

### P0：建议先部署的四组

| 优先级 | 数据集/任务 | 输入与预测目标 | 为什么适合当前项目 | 主要指标/注意事项 |
|---|---|---|---|---|
| P0 | [Ego4D FHO Short-Term Object Interaction Anticipation](https://ego4d-data.org/docs/benchmarks/forecasting/) | 观察动作发生前的第一人称视频，预测将要交互的物品、verb 和 time-to-contact | 最直接的人—物关系预测；有 hand/object box 和接触时间；有官方代码 | verb、noun、action 的 top-k 或官方检测指标；按 0.5/1/1.5/2 秒预测距离分层 |
| P0 | [Ego4D FHO Long-Term Action Anticipation](https://github.com/EGO4D/forecasting/blob/main/LONG_TERM_ANTICIPATION.md) | 根据已观察动作历史，预测后续动作序列 | 适合测 TTT 的长程状态；可直接比较短上下文、长 ICL 和 fast-weight memory | 使用官方 forecasting evaluator；报告 action sequence 的整体、seen/unseen wearer 和不同历史长度 |
| P0 | [Ego4D-OSCA](https://arxiv.org/abs/2405.12789) | 预测下一步未观察动作会导致的 object state change，包括 no-change | 比普通动作分类更接近“物品会变成什么状态”；显式考察动作历史和物品状态历史 | 类别不均衡，不能只报 accuracy；应报告 macro-F1、balanced accuracy 和按对象/状态类别分层结果 |
| P0 | [SVCBench](https://github.com/buaa-colalab/SVCBench) | 在视频播放过程中多次提问，预测 object/event count、delta、identity 和 state transition | 是流式状态维护的直接诊断；406 个视频、1,000 个问题、4,576 个时间查询点；输出是一条预测轨迹而不是单一答案 | GPA、MoC、UDA；尤其适合画性能随时间、更新次数和场景变化的曲线 |

FHO-STA 适合先证明模型能否预测下一次人—物交互，FHO-LTA/OSCA 适合测试历史是否有用，SVCBench 适合检查状态是否随着视频持续更新。四者组合后，可以把“短期动作预测”“物品状态预测”“长程流式记忆”分开。

### P1：补充事件因果、空间动态和物理预测

| 数据集/任务 | 适合回答的问题 | 局限 |
|---|---|---|
| [VLEP](https://github.com/VALUE-Leaderboard/DataRelease/blob/main/DATA.md) | 给定一个事件和视频片段，在两个候选事件中选择更可能发生的后续事件；适合测 event causality 和 future event prediction | 主要是 TV/YouTube 视频，不是第一人称；不能代替家庭视频评测 |
| [NExT-QA](https://arxiv.org/abs/2105.08276) | causal、temporal、previous/next、why/how 问题；适合做因果预测的语言对照 | 多数视频较短，更多是视频理解控制组，不是长期记忆数据 |
| [EPIC-KITCHENS-100 Action Anticipation](https://epic-kitchens.github.io/2021) | 预测下一步厨房动作的 verb、noun、action；用户已经有 EPIC 视频，部署成本低 | 场景集中在厨房，主要是局部动作预测；应使用官方 top-5 recall 和 unseen/tail 划分 |
| [Ego-Exo4D](https://docs.ego-exo4d-data.org/) | 使用第一人称视频、第三人称视频、3D 手/体姿态、物体 mask、keystep 和 procedural dependency，预测下一步技能动作或人—物关系 | 数据、标注和多视角处理较复杂；适合第二阶段，不适合第一个 smoke test |
| [Dyn-Bench / VLM4D / DSI-Bench](https://arxiv.org/abs/2607.12503) | 测物体动态、相机运动与物体运动的区分、observer-centric 4D reasoning | 更偏动态空间理解，需确认具体 split 和视频访问方式；可以作为 Spatial-TTT 的动态对照 |
| [VSI-Super-Wild](https://github.com/THUSI-Lab/VSI-Super-Wild) | 442 个真实世界视频、6,980 个问题，含长达 4 小时以上的视频；测 agent/object/environment 三类世界状态 | 目前主要是状态回忆和空间关系，不是纯粹的下一事件预测；适合长程 memory stress test |
| [Physion](https://physion-benchmark.github.io/) | 预测碰撞、滚动、滑动、弹射、软体等物理结果；可以判断 TTT 是否真的学到可预测的动力学 | 是受控仿真，不是第一人称现实视频；应作为“物理边界”而不是主数据集 |

### P2：受控的场景变化与 RL 数据

| 平台/数据 | 用法 |
|---|---|
| [VirtualHome Action Genome](https://openaccess.thecvf.com/content/WACV2023/papers/Qiu_VirtualHome_Action_Genome_A_Simulated_Spatio-Temporal_Scene_Graph_Dataset_With_WACV2023_paper.pdf) | 在合成家庭场景里得到一致的时空 scene graph，直接生成“人物把物品从 A 搬到 B 后，下一问物品在哪里”的可控样本 |
| [Habitat 3.0](https://ai.meta.com/research/publications/habitat-3-0-a-co-habitat-for-humans-avatars-and-robots/) | 支持 Social Navigation 和 Social Rearrangement；适合模拟人物和机器人同时改变场景，测记忆与规划的闭环关系 |
| [BEHAVIOR-1K / OmniGibson](https://github.com/StanfordVL/BEHAVIOR-1K) | 1,000 个家庭活动、可执行的物理任务、场景状态和演示轨迹；适合做状态变化、动作预测和后续 RL |
| [MindCube](https://github.com/mll-lab-nu/MindCube) | 由有限视角推断空间 cognitive map，并包含 dynamics/what-if 问题；官方仓库已经放出 RL recipe | 主要是多视图图像和受控空间推理，不是长第一人称视频；适合测试 map-then-reason 与 TTT 的边界 |

MindCube 已经报告了用 cognitive map 监督再加 RL 的路线；这对我们有参考价值，但不能把它的 RL 分数直接解释成视频记忆收益。它更适合作为“结构化空间状态是否有利于后续预测”的独立对照。

## 3. 当前仓库与这些任务的对应关系

当前 `TTT_frame` 已经有三条路线，必须分开记录：

1. `ttt_frame/videoqa.py`：视频到 LoRA 的 self-distillation baseline。它通过 teacher 生成 caption/QA，然后用 AdamW 更新 LoRA。它是参数记忆 baseline，不是 Spatial-TTT。
2. `ttt_frame/spatial_memory.py` + `ttt_frame/spatial_model.py`：Spatial-TTT 风格的 SwiGLU fast weights、关联梯度、momentum、Muon、row normalization、空间分支和 sliding-window attention。
3. `ttt_frame/spatial_videoqa.py`：把视频视觉 token 写入 Spatial fast weights，丢弃视频和历史 KV，再用文本问题读取。问题阶段不写 memory，且没有 caption teacher 或在线 AdamW。

当前 Spatial 视频入口的默认配置是：

```text
chunk_seconds   = 4
frames_per_chunk = 8
max_side        = 448
```

也就是最多约 2 FPS。对短 clip 这很合理，对小时级视频则非常昂贵。每个 chunk 仍要经过 Qwen3-VL 的视觉编码、语言模型前向和选定层的 fast-weight 更新，因此耗时不只是 PyAV 抽帧时间。

官方 [Spatial-TTT README](https://github.com/THU-SI/Spatial-TTT) 的主训练脚本使用 `video_max_frames=128`、`video_fps=30`，再由 processor 把整段输入限制到最多 128 帧，并使用 `lact_chunk_size=2648` 和 `window_size=2648`。因此官方的 `video_fps=30` 不是长视频固定每秒取 30 帧；长视频会被上限截断/均匀采样。当前仓库的 4 秒/8 帧是为了流式连续写入而设置的项目配置，不能直接称为官方采样配置。

第一个实验不建议使用 4 秒/8 帧作为唯一配置。建议把它作为 dense upper-bound，再加入：

```text
fast    : 4 s / 8 frames   # 约 2 FPS，测密集视觉证据
medium  : 8 s / 4 frames   # 约 0.5 FPS，建议默认
sparse  : 30 s / 4 frames  # 约 0.13 FPS，测长视频压缩
```

三种配置必须在 ICL 和 TTT 两边同时使用。否则密集 TTT 对稀疏 ICL 的优势不能被解释为 fast weights 的优势。

## 4. 建议的统一数据格式

为了让 FHO、EPIC、SVCBench、自建搬家数据共用一套 runner，建议把每个评测项整理成 JSONL：

```json
{
  "id": "ego4d_sta_val_000001",
  "task": "next_object_interaction",
  "video": "/data/videos/clip.mp4",
  "observed_end_sec": 12.50,
  "horizon_sec": 1.00,
  "question": "What is most likely to be interacted with next?",
  "choices": ["cup", "drawer", "knife", "nothing"],
  "answer": 1,
  "metadata": {
    "verb": "open",
    "noun": "drawer",
    "subject": "camera_wearer",
    "scene_id": "kitchen_03",
    "split": "val"
  }
}
```

关键约束：

* `observed_end_sec` 之前的视频才进入 `ingest`；后面的未来片段只能用于 scorer。
* 所有条件使用相同的 `observed_end_sec`、帧数、分辨率、问题和选项。
* 主指标优先使用封闭集合选择题或结构化标签，避免自由生成中的语言先验。
* 对开放题可以保存生成答案，但只作为 secondary metric，并进行规范化后的 exact match/F1。
* `metadata` 中保存 horizon、场景、人物、物品类别和是否发生视角变化，用来画分层曲线。

当前 `spatial_videoqa ingest` 接口按整个视频路径读取，没有内置 `observed_end_sec` 参数。若原始数据不是已经裁好的 prefix clip，正式 runner 需要增加时间裁切接口，或在评测前生成只包含 prefix 的临时视频。不能把整段视频交给 ingest 后再假装是预测任务。

## 5. 第一版 baseline 矩阵

### 5.1 最小模型

建议先固定 Qwen3-VL-2B 和一张 GPU，先不引入大模型或复杂 scene graph：

| 编号 | 条件 | 历史如何使用 |
|---|---|---|
| B0 | blind/current | 只给当前查询需要的当前帧或最短 clip |
| B1 | ICL-short | 把最近 4/8 帧放进上下文，不更新权重 |
| B2 | ICL-long | 把相同总帧数的历史均匀放进上下文 |
| B3 | spatial-TTT | 按时间顺序写入 fast weights，问题阶段只读 fast weights |
| B4 | spatial-TTT + no-write control | 用同一 Spatial checkpoint，但跳过视频写入 |
| B5 | explicit retrieval | 用 CLIP/视觉 embedding 检索最近或最相似帧，作为 memory control |

LoRA-TTT (`videoqa.py`) 可以作为 B6，但不能把它和 B3 的结果合并成“TTT 总体结果”。它们的写入目标、损失和更新规则不同。

### 5.2 首轮数据顺序

1. **EPIC-KITCHENS-100 action anticipation**：先利用现有视频，验证接口和 action/verb/noun 解析。
2. **Ego4D FHO-STA**：验证第一人称 human-object 预测，按 horizon 分层。
3. **Ego4D-OSCA**：验证状态变化预测，重点是“打开/关闭、空/满、切开/未切开、移动/未移动”等。
4. **SVCBench**：验证长期 streaming state maintenance，记录每个 query point 的结果。
5. **VLEP/NExT-QA causal subset**：验证事件后果推断，作为非第一人称控制组。

每个数据集先抽 50--200 个 validation items 做 smoke run；跑通后再扩大到完整 validation。第一轮的目的不是争取 SOTA，而是确认 TTT 的收益是否集中在预测任务上。

## 6. 必须记录的实验曲线

只报一个总准确率很容易掩盖 TTT 的真实边界。每次 run 至少保存：

* accuracy/F1 或官方 metric 随 `history length` 的曲线；
* 随 `prediction horizon` 的曲线；
* 随 `number of scene changes` 的曲线；
* seen/unseen person、object、scene 的结果；
* `frames`, `visual_tokens`, `TTT updates`, GPU memory 和 wall-clock time；
* TTT 写入前后当前识别题的分数；
* 每个 query 的 prediction、confidence 和 error type。

推荐的主要统计量是 paired delta：

```text
delta_TTT_ICL = score(Spatial-TTT) - score(ICL-long)
```

再按 horizon、scene-change 和 task type 分组。若整体 delta 接近 0，但在“物品被搬动后的位置预测”上显著为正，这仍然是有价值的边界结果。

## 7. 预测框架的选择

### 7.1 第一主线：Spatial-TTT / LaCT

[Spatial-TTT](https://arxiv.org/abs/2603.12255) 是当前最贴近项目的问题设定的架构：SWA 保留局部视觉证据，TTT 分支把长程信息写入 fast weights，3D depthwise convolution 提供空间和时间连续性。官方代码和 Spatial-TTT-nano checkpoint 已经发布。

[LaCT](https://arxiv.org/abs/2505.23884) 是它的快权重基础：用大 chunk 做关联学习更新，而不是每 16/64 token 做小更新。它适合作为数学和效率 baseline，但 LaCT 本身不会自动保证“写进去的信息未来可被正确寻址”。因此在预测任务上，必须单独测 storage、addressing 和 readout。

### 7.2 第二主线：预测表征而不是预测像素

[V-JEPA 2](https://github.com/facebookresearch/vjepa2) 是很好的非 TTT 对照。它在 latent space 预测未来表示，V-JEPA 2-AC 再加入 action-conditioned world model。对于本项目，可以把它理解为：

```text
video prefix -> latent state z_t
(candidate action/event) -> predicted future state z_{t+1}
```

它可以作为“真正的预测模型”对照，帮助判断 Spatial-TTT 的收益来自记忆，还是只是来自更强的视觉编码器。

### 7.3 第三主线：显式动态轨迹和场景图

[DynTrace](https://arxiv.org/abs/2607.12503) 用动态轨迹可视化和 Dynamic Trace Graph 保存物体动态证据，是非常合适的 explicit-memory baseline。它可以回答一个重要问题：如果把 object trace 和 pairwise relation 显式写出来，是否比隐式 fast weights 更稳定？

[Cambrian-S](https://github.com/cambrian-mllm/cambrian-s) 的 predictive sensing 用预测误差寻找视频中的自然断点，再对流式视频做分段。它启发我们给 Spatial-TTT 加一个 gate：只有 prediction error 高、对象状态变化或场景切换时才进行高成本写入。

### 7.4 与控制相连的 TTT

[TTT-VLA](https://arxiv.org/abs/2606.03127) 不直接更新整个 VLA，而是只优化一个 latent prompt，并用环境交互的 proxy task 做 test-time adaptation。这对我们的设计有直接启发：先让 TTT 只负责更新一个小的、可读的预测状态或 latent prompt，再考虑更新大规模 attention/MLP 参数。

## 8. 强化学习的应用路径

RL 值得做，但不建议第一步就让 RL 同时学习视觉、记忆和机器人策略。更可行的顺序是从“学习什么时候写入”开始。

### Level 0：监督/自监督预测，不使用 RL

先用 FHO、OSCA、SVCBench 建立固定的 TTT writer 和 readout。此阶段的 loss 可以是：

* next object / verb / noun 分类损失；
* object state change 分类损失；
* 下一事件二选一或多选一损失；
* future latent state prediction loss；
* 旧问题保持不变的 retention loss。

这个阶段的作用是确认 fast weights 有可读信息，再决定 RL 是否值得引入。

### Level 1：RL 学 memory controller

把 memory controller 的动作定义为：

```text
write(frame/chunk)
skip(chunk)
flush()
reset()
选择采样率、chunk size、fast-weight learning rate 或 decay
```

奖励可以写成：

```text
r_t = λ_pred * future_prediction_score
    + λ_state * state_consistency
    - λ_compute * update_cost
    - λ_forget * old_query_regression
```

这样 RL 学的是“哪些观察值得进入隐式记忆”，而不是直接从稀疏视频中学习所有动作。训练环境可以先用 VirtualHome 或 BEHAVIOR-1K，因为物品位置、状态变化和事件时间都已知。

### Level 2：行动条件的未来状态预测

在 fast-weight state `m_t` 之后增加一个小的 action-conditioned predictor：

```text
m_t, candidate_action -> predicted object/person/event state at t+1
```

在 Habitat 3.0 的 Social Rearrangement 或 BEHAVIOR-1K 中，使用仿真器返回的状态变化作为 reward/label。此时可以比较：

* 无记忆的 policy；
* ICL/显式 scene graph policy；
* TTT memory policy；
* TTT memory + learned gate。

### Level 3：闭环 RL / VLA adaptation

最后才考虑把预测状态用于动作规划。可以参考：

* [TTT-VLA](https://arxiv.org/abs/2606.03127)：只优化 latent prompt；
* [Test-Time Training for Visual Foresight VLA](https://arxiv.org/abs/2605.08215)：把 TTT 与视觉前瞻结合；
* [On-the-Fly VLA Adaptation via Test-Time RL](https://arxiv.org/abs/2601.06748)：在交互过程中用 RL 适配 VLA；
* [TTRV](https://openaccess.thecvf.com/content/CVPR2026/html/Singh_TTRV_Test-Time_Reinforcement_Learning_for_Vision_Language_Models_CVPR_2026_paper.html)：在无标签测试样本上用 test-time RL 改进 VLM 输出。

这些工作说明 test-time RL 可行，但它们的 reward、动作接口和模型更新对象都不同。它们不能直接证明“RL 能让 Spatial-TTT 记住家庭视频”，只能作为 memory controller 和 latent adaptation 的方法来源。

## 9. 当前最值得做的三个实验

### 实验 A：FHO-STA 的短期预测边界

固定 Qwen3-VL-2B，分别跑 `blind`、`ICL-4f`、`Spatial-TTT-4s8f` 和 `Spatial-TTT-8s4f`。按 0.5、1.0、1.5、2.0 秒预测距离分层，记录 verb/noun/action。

预期：短 horizon 上 ICL 可能已经足够，TTT 不应显著领先；如果 TTT 在短 horizon 也明显变好，要检查是否视觉帧预算不公平或模型只学到了类别先验。

### 实验 B：OSCA 或 SVCBench 的状态维护

对同一个视频在多个时间点提问，比较每个 query 的状态预测和更新轨迹。重点看人物离开视野、物品被遮挡、物品状态改变、相机移动和再次出现等情况。

预期：TTT 的优势如果存在，应该随 query 距离和遮挡/场景变化增加而出现；如果只在最后一个问题上有效，不能称为稳定的状态记忆。

### 实验 C：自建搬家/重排受控集

用 VirtualHome 或 Habitat 3.0 生成同一对象在房间 A 与房间 B 之间移动的序列，给模型输入：

```text
day 1: person A puts the red mug on the kitchen table
scene change / relocation
day 2: person B asks where the red mug is likely to be
```

把“场景发生变化”与“仅增加无关 filler”分开。问题分为：

* 物品最后位置；
* 谁移动了物品；
* 物品与人物关系是否保持；
* 物品状态是否因动作改变；
* 下一步最可能发生什么。

这是最容易证明 TTT 真正有用、也最容易发现 fast weights 读出失败的可控实验。正式论文结果仍需在真实第一人称数据上复现。

## 10. 判断边界的标准

我们可以把结论按下面的形式写，而不是只写“TTT 有提升”或“TTT 没提升”：

* **适合**：小模型、长历史、输入不能全部放进上下文、需要保留对象—人物—状态绑定、未来标签是结构化且可验证的任务。
* **不适合**：短视频、当前帧已经包含答案、需要精确读取少数视觉细节、没有可靠自监督写入目标、或问题需要检索原始证据而不是压缩后的状态。
* **待验证**：场景变换后的身份绑定、多人交互、关系覆盖/撤销、连续状态计数和行动条件的未来状态预测。

最重要的消融不是继续增加抽帧密度，而是比较：

```text
same frames + ICL
same frames + TTT
same frames + explicit retrieval
TTT with write-all
TTT with learned gate
```

如果 TTT 只有在 2 FPS 的高密度输入下胜过 ICL，而在 0.5 FPS 下消失，结论应是“收益依赖密集观察”，不能写成“TTT 更擅长长期预测”。如果加入 gate 后，在保持准确率的同时减少更新次数，才说明 TTT 适合流式部署。
