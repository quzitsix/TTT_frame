# 让 Spatial-TTT 记住视频内容：诊断与训练方案

## 先区分三件事

当前 Q9 实验已经证明了“视频会改变 fast weights”，但还没有证明“模型能从这些
参数中稳定读出正确的视频事实”。这两个结论不能互相替代。

本项目现在有三类参数：

1. 官方完整 checkpoint 中的语言模型和 Spatial-TTT slow parameters。它们是在
   官方数据上训练得到的通用读写规则与模型先验，不是 Q9 这段家庭视频的情景记忆。
2. `fast_weights.safetensors`。它由 Q9 的视频 token 在线更新，是本次 episode 的
   参数记忆。
3. `memory.json`。它只保存配置、计数、位置偏移和校验信息，不保存语义事实。

`spatial_videoqa ask --without-memory` 仍会加载同一份官方训练模型，只是绕过第 2
项。因此“有 memory”和“无 memory”都回答 `kitchen cabinet`，并不是训练模型与
未训练模型相同；它只说明当前问题的最高概率答案没有被 Q9 fast weights 改变。
“厨房里有什么”也很容易被语言模型用常识回答，不适合判断 episodic memory。

当前 `SpatialVideoMemory.ingest_frames()` 使用 `@torch.no_grad()`，每个视频块之后
还会 detach fast state。这个过程只执行测试时写入，没有 `loss.backward()` 或
`optimizer.step()`，所以 Q9 视频没有训练 slow parameters 或语言模型。

## 怎样按顺序核对视频

18 个片段的唯一可靠顺序来自 `media_index.json` 的 `start_sec`，不是文件名。使用：

```bash
cd /home/quzitsix/TTT_frame
python scripts/watch_supermemory.py list
python scripts/watch_supermemory.py play --fullscreen
```

项目已经生成了一个可拖动进度条的 1080 秒合并文件：

```text
/home/quzitsix/TTT_frame/runs/viewer/supermemory-q9-ordered.mp4
```

可以直接播放：

```bash
ffplay -fs runs/viewer/supermemory-q9-ordered.mp4
```

也可以只看第 5 个片段，即绝对时间 240--300 秒：

```bash
python scripts/watch_supermemory.py play --start 5 --end 5
```

观看时记录以下字段，而不是只写一段全局摘要：

```json
{
  "start_sec": 248.0,
  "end_sec": 267.0,
  "room": "kitchen",
  "objects": ["red mesh bag", "white disposal container"],
  "action": "put/throw away",
  "final_state": "the bag is inside the white container near the island",
  "certainty": "high"
}
```

时间、对象、动作和最终位置必须绑定在同一条记录中。只记录物体清单无法监督
“谁把什么放到哪里”或“最后在哪里”。

## 先建立能区分 memory 的问题集

在改模型之前，从人工核对的时间线中建立至少四类问题：

| 类型 | 示例 | 检查能力 |
|---|---|---|
| 视频特有对象 | 红色网袋里原来装了什么？ | 是否见过具体对象 |
| 动作与位置绑定 | 红色网袋最后被放到哪里？ | 是否把对象、动作、地点绑在一起 |
| 时间更新 | 杯子最后一次出现在哪里？ | 是否保留最新状态而不是早期位置 |
| 顺序与否定 | 是先打开抽屉还是先拿起杯子？ | 是否记住事件顺序并抑制常识猜测 |

不要把笼统的 `What objects did I interact with in the kitchen?` 作为主指标。更好的
问题应包含只看这段视频才能知道的事实，并为每题保存证据时间段和可接受答案。
训练题与测试题应按事件分开，不能只是把同一句话换一种表述后放进测试集。

每个问题至少跑四个条件：

1. 原始 Qwen 基座；
2. 官方 Spatial-TTT checkpoint，`--without-memory`；
3. 官方 checkpoint 加正确的 Q9 fast memory；
4. 官方 checkpoint 加另一段视频的 memory，或把 Q9 片段打乱后产生的 memory。

正确的 episode memory 应在第 3 项提高有证据问题的准确率；时间问题在打乱顺序后
应明显变差；换成另一段视频的 memory 不应继续给出 Q9 的具体事实。仅仅看到两段
自由文本不同，或只看到 fast weights 数值变化，都不算语义记忆通过。

当前单题对比可直接使用：

```bash
python scripts/compare_videoqa.py \
  --device cuda:2 \
  --local-memory runs/q9_full_history_lora_local_60s \
  --codex-memory runs/q9_full_history_lora_codex_sol \
  --spatial-memory runs/q9_full_history_spatial_official_16f \
  --max-new-tokens 96 --concise \
  --question 'Where did I put the empty red mesh bag, and what was beside that place?'
```

脚本会同时显示 `Spatial-TTT + memory` 与同 checkpoint 的
`Spatial-TTT same checkpoint (--without-memory)`，并打印实际使用的问题，避免 shell
变量残留造成假对照。

## 当前输入为什么还不够强

`q9_full_history_spatial_official_16f` 对每个 60 秒片段只抽 16 帧，即平均约 3.75 秒
一帧。短暂的拿起、放下和容器开合可能完全落在两个采样点之间。提高采样密度能
降低漏看，但不能单独解决语义读出：当前视频 write 使用的固定提示只有
`Observe the visible environment.`，没有对对象、动作、位置和时间关系提供监督。

建议先在一两个有人工金标的 60 秒片段上使用 `4 秒 / 8 帧` 做诊断，再决定是否对
全部 1080 秒使用高密度设置。这样可先判断问题来自漏帧还是读写规则。完整视频直接
提高到该密度会从 288 帧增加到约 2160 帧，计算和显存代价也会显著增加。

## 真正需要训练的目标

要让 Spatial-TTT 适配本项目的“视频输入后撤销视频，只靠参数回答”，需要新增独立
的 offline trainer，不能复用带 `no_grad` 的在线 `ingest`。每个训练 episode 应按
以下顺序执行：

1. reset 当前 fast state；
2. 按时间顺序写入一个或多个视频块，不保存视频 KV；
3. 保留写入产生的 autograd graph；
4. 清除视频输入后，在 read 模式下输入带金标答案的文本问题；
5. 只在 assistant 答案 token 上计算交叉熵；
6. 一次性对该 episode 的问答 loss 反向传播，更新 TTT slow parameters 和选定的
   语言参数；
7. optimizer step 后丢弃该 episode 的 fast state；
8. 用更新后的 slow parameters 重新写入验证视频，再做无视频查询。

训练数据应同时包含密集场景描述和有证据的 QA。场景描述负责对象、数量、空间关系
与状态变化；QA 负责训练文字查询怎样读出对应事实。只用一条 1080 秒视频会严重
过拟合，至少要按独立视频或独立事件划分 train/validation/test。

实现时还有三个约束：

- `SpatialQwenMemory` 构造时默认冻结全部模型参数，训练器必须显式解冻 slow
  parameters 和计划训练的语言参数；
- 训练 write 不能调用 `finish_ingest()` 或 `detach()`，也不能开启 gradient
  checkpointing，否则会切断计算图或重复写同一观察；
- 现有 `spatial.safetensors` 只保存新增的 Spatial 参数。若训练语言模型参数，需要
  新的 full-model 或 language-delta checkpoint 格式。

第一版可只训练 Spatial slow parameters，并给语言 attention 加小型 LoRA，降低
2B 全量训练的成本。调大 `ttt_scale` 只能让分支影响更强，不能把错误的写入变成
正确语义，因此不应作为准确率修复。

## 阶段验收

建议把实验分成四个门槛：

1. **写入门槛**：视频前后的 fast weights 不同；当前已经通过。
2. **调用门槛**：read 模式确实调用 21 个 TTT 层，`--without-memory` 不调用；当前
   已经通过。
3. **语义门槛**：在未参与训练的视频特有问题上，正确 memory 稳定优于同 checkpoint
   的无 memory 与错误 memory；当前尚未通过。
4. **长期门槛**：追加后续片段后，早期事实、最新状态和顺序题仍保持准确；当前尚未
   建立足够的有证据题集来判断。

在第 3 个门槛通过之前，当前结果应表述为“Spatial fast-weight 机制已接通”，不应
表述为“模型已经获得准确的家庭长期视觉记忆”。
