# Experiment C-PNG：受控物体状态、跨场景记忆与下一步预测（历史机制诊断）

> 当前实验是由文字事件卡组成的合成机制测试（PNG-card proxy），不是家庭第一人称真实视频结果。它适合检查状态写入、跨场景保持和读取接口是否工作，不能证明模型已经学会现实世界中的视觉记忆或未来预测。
> 当前主要视觉 C 评测已切换到 [SVCBench 真实视频 matched-prefix 协议](SVCBENCH_REAL_VIDEO.md)。

## 1. 正式协议

C-PNG 的 fixture schema v2 包含 12 个 episode。每个 episode 有 3 个物体、3 个居民和 5 类问题：最终位置、移动者、所有者、最终状态和下一步动作。每个 episode 分别生成三种条件：

| condition | 观测变化 | 作用 |
|---|---|---|
| `relocated` | 物体从场景 A 移到场景 B | 测试场景变化后的状态保持与读取 |
| `same_scene_control` | 相同的 move/update 发生在场景 A | 控制事件数和写入数，隔离 scene cue 的影响 |
| `filler_only_control` | 删除目标 move，用等预算干扰写入替代 | 测试普通保持，并检查模型会不会凭问题前提虚构移动 |

三种条件每个 episode 都有 **16 张可观测卡**。symbolic runner 中每种条件都有 **27 次结构化写入**。`same_scene_control` 的显式检索表会覆盖同 key 的旧状态，因此活跃 entry 数少于另外两种条件；默认 `max_entries=128` 时不会触发容量淘汰，但小容量实验需要单独控制这一点。

### 未来信息边界

fixture schema v2 把未来事件单独保存为 `future_event`，它不属于 `events`，不会被渲染、编码或写入记忆。symbolic、视觉和 JEPA runner 还会过滤 legacy fixture schema v1 manifest 中遗留的 `forecast` 事件。因此模型能看到所有者、move 和最终位置等当前状态，但看不到下一步动作答案卡。

`next_action` 使用公开的确定性策略：

```text
如果物体发生过移动，它的所有者下一步检查物体的最终位置；
如果没有发生移动，答案为 unknown。
```

所以该题测量的是“保留当前状态后按公开规则组合预测”，不是从数据中发现开放世界行为规律。策略对所有方法公开，答案值本身仍被隐藏。

### fixture schema v2 的已知限制

当前 12 个 episode 并不对应 12 个相互独立的内容样本：按任一 condition 下的完整可观测事件流合并 exact duplicate 后，跨条件联合形成 **9 个 exact-content clusters**。监督 readout 和配对统计必须至少按这些 cluster 隔离，不能把 episode ID 直接当作独立划分单位；但 cluster 之间仍共享确定性的生成周期，也不能视为 9 个独立同分布随机样本。

生成器还保留了几项可被模型利用的结构捷径。`moved_object`、最终位置和最终状态随 episode index 按 `% 3` 同步循环，因此模型可能从模板规律恢复标签，而不需要形成持久记忆；`filler_only_control` 的 PNG 又直接显示 `control distractor`、`actor: control` 和 `control_records: 4` 等内部控制字段。最后，`same_scene_control` 除了去掉背景变化，还把两个 location 文本从 `living room shelf`、`side table` 缩短为 `shelf`、`table`，所以 relocated 与 same-scene 的差异不是纯场景 cue 干预。正式结果只能作为当前 schema v2 的机制诊断。当前 C-real 主评测改用了真实视频的前缀计数任务，而不是这套 fixture 的 schema v3；两者任务不同，分数不能直接比较。

## 2. Symbolic memory 结果

运行文件：`runs/experiment_c_v2_final.summary.json`。配置为 12 episodes、10 trials、64 维 semantic code、`scene_drift_strength=0.35`，共 7,200 行预测。每个 arm/condition 有 600 个样本。

| arm | relocated | same scene | filler only | coverage |
|---|---:|---:|---:|---:|
| Oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Explicit retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LaCT fast weights | **0.6383** | **0.7333** | **0.4967** | 1.0000 |
| None | — | — | — | 0.0000 |

Oracle 和 explicit retrieval 全对，说明 condition transformation、gold 和下一步动作组合逻辑是可解且一致的。`none` 没有预测覆盖率，不能把它当作 0% 准确率模型。

LaCT 在三个条件合并后的题型结果如下，每类有 360 个样本：

| query type | accuracy | correct / 360 | relocated − same scene |
|---|---:|---:|---:|
| location | 0.8111 | 292 | -0.0500 |
| mover | 0.8528 | 307 | -0.0167 |
| owner | 0.3028 | 109 | -0.2000 |
| state | 0.6833 | 246 | 0.0000 |
| next action | 0.4639 | 167 | -0.2083 |

LaCT 的 relocated 总准确率比 same-scene 低 0.0950。主要损失来自 owner 和 next action；next action 又依赖 owner 与 final location，因此这个结果符合组合读取链路的结构。它表明当前关联式 fast weights 对 scene cue 变化敏感，但不能直接外推到真实视频。

复现命令：

```bash
conda run --no-capture-output -n meowbench \
  python scripts/evaluate_experiment_c.py \
  --data data/experiment_c/manifest.json \
  --output runs/experiment_c_v2_final \
  --trials 10 \
  --scene-drift-strength 0.35
```

## 3. 视觉 PNG-card 严格 MCQ

正式视觉测试使用 schema 为 `experiment_c_relocation_v2` 的完整 12 个 episode。每个 arm 覆盖 12 episodes × 3 conditions × 5 query types，共 **180 题**。每题都有 4 个唯一选项，正确位置在 A/B/C/D 各出现 45 次。所有正式 arm 都使用 `separate` layout，即每次生成只回答一题。主指标是严格 `choice_accuracy`；解析失败和运行错误均按错误回答计入 180 题分母，开放回答 semantic scorer 仅用于诊断。

比较包含五个读取状态：

- `ICL`：原始 Qwen3-VL-2B-Instruct 在回答每道题时直接接收同一组 16 张卡。
- `blind`：同一个原始 Qwen3-VL 只接收题目和选项，不接收卡片。它是 ICL 的直接因果对照。
- `Spatial fast`：加载官方 Spatial-TTT 完整 checkpoint，每个 condition 将 16 张卡写入一次，保留 episode-dependent fast state，然后做 5 次纯文本读取。
- `Spatial slow-only`：使用相同 checkpoint、Spatial 读分支、ingest 和 position budget，但在问答前丢弃本 episode 产生的 fast state，只读官方 checkpoint 中学到的初始状态。它是测量 **test-time fast update 因果效应**的主对照。
- `Base-attention bypass`：完成相同 ingest 和 position budget，但回答时绕过整条 Spatial 读分支，直接使用原 attention。它是架构诊断，不是只移除 fast update 的对照。

| arm | correct / 180 | choice accuracy | parse rate | 解释 |
|---|---:|---:|---:|---|
| Qwen3-VL ICL | **89** | **0.4944** | 0.9833 | 直接视觉上下文 |
| Text-only blind | 48 | 0.2667 | 1.0000 | ICL 的无视觉对照 |
| Spatial fast | 48 | 0.2667 | 1.0000 | 官方 slow weights + episode fast state |
| Spatial slow-only | **51** | **0.2833** | 1.0000 | 相同 Spatial 读路径，丢弃 episode fast state |
| Base-attention bypass | 49 | 0.2722 | 1.0000 | 绕过整条 Spatial 读分支，仅作诊断 |

分条件结果：

| arm | relocated | same scene | filler only |
|---|---:|---:|---:|
| ICL | 0.5167 | 0.4833 | 0.4833 |
| Blind | 0.1833 | 0.2333 | 0.3833 |
| Spatial fast | 0.3000 | 0.3500 | 0.1500 |
| Spatial slow-only | 0.3333 | 0.3500 | 0.1667 |
| Base-attention bypass | 0.3000 | 0.3333 | 0.1833 |

分题型结果：

| arm | location | mover | owner | state | next action |
|---|---:|---:|---:|---:|---:|
| ICL | 0.5556 | 0.1389 | 0.6667 | 0.6389 | 0.4722 |
| Blind | 0.0833 | 0.2222 | 0.0556 | 0.5556 | 0.4167 |
| Spatial fast | 0.2500 | 0.2500 | 0.3611 | 0.2778 | 0.1944 |
| Spatial slow-only | 0.2500 | 0.2500 | 0.3611 | 0.3611 | 0.1944 |
| Base-attention bypass | 0.3056 | 0.2500 | 0.3333 | 0.3056 | 0.1667 |

ICL 比 blind 多答对 41/180 题，准确率差为 **+0.2278**。合并 exact duplicate 后，在 9 个 exact-content clusters 上枚举 sign flip 得到 `p=0.00781250`；由于 cluster 之间仍共享 `%3` 生成周期，这个值是 fixture 内诊断，不能解释成独立同分布总体推断。将 180 个 query 当作独立样本的 row-level exact McNemar 给出 `p=0.000003`，其独立性假设更弱，所以只作探索性诊断。这组配对结果说明 Qwen3-VL 能从 PNG 卡片中读取有用信息。其中 ICL 有 3 题没有产生可解析的单字母答案，已按错误回答计分，没有从分母中删除。

Spatial fast 比严格匹配的 slow-only **少答对 3/180 题**，差值为 **-0.0167**；两者的 left-only/right-only correct 为 1/4，content-group exact sign-flip `p=0.500000`，row-level McNemar `p=0.375000`。因此当前 C-PNG 没有显示 test-time fast update 的正向效果；这一结论来自 fast 对 slow-only，不来自 base-attention bypass。

Spatial fast 比 base-attention bypass 少 1/180 题，差值为 -0.0056，content-group sign-flip 和 row-level McNemar 都为 `p=1.000000`。这个比较同时移除了 Spatial slow/read 分支与 fast state，因此仅作架构诊断。

### 帧与 checkpoint 的可比边界

除不接收视觉输入的 blind 外，ICL 和三个 Spatial arm 使用的唯一观测集都是每个 condition 的同 16 张卡，但总视觉计算预算不相等。`separate` ICL 为 5 道题各重新编码 16 帧，每个 condition 合计 **5 次视觉编码、80 次帧呈现**；Spatial 各 arm 只做 **1 次 16 帧 ingest**，之后为 5 次文本读取。因此本表在需要视觉输入的 arm 之间匹配了唯一帧集合，且五个 arm 都使用相同的问题 layout，但没有匹配总视觉 token/FLOP 预算。

ICL/blind 使用原始 Qwen3-VL-2B-Instruct，Spatial 三个读取状态使用官方 Spatial-TTT 完整 checkpoint，其中包含已调整的 backbone/slow parameters。因此 ICL 对 blind 和 Spatial fast 对 slow-only 是各自的直接配对对照；ICL 与 Spatial 的绝对分数只能比较端到端系统，不能将差值单独归因于 memory 形式。

视觉 runner 行内记录基础模型绝对路径和 config fingerprint；正式汇总另以 `--hash-model-artifacts` 对 Qwen 权重、processor/tokenizer 文件和 Spatial checkpoint 建立逐文件 SHA-256 清单，并校验 Spatial 行内 checkpoint hash。跨机器复现应保留这个 artifact manifest，不能只比较模型路径。

- Qwen 模型目录 tree SHA-256：`b6e2cc74528a784288866831a2c80c21be0a65b2479e7212bcc7243c3081ab9b`
- Qwen `model.safetensors` SHA-256：`7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`
- Spatial-TTT `model.safetensors` SHA-256：`5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947`

正式产物：

ICL 的 180 行由当前 runner 生成的三个互不重叠 4-episode shard 合并而成；blind 也使用同一 runner 和完全相同的文字提示重新运行，二者唯一实验差异是是否提供 16 张观测卡。五个 arm 都直接记录完整 `run_protocol`。runner hash 按可解释的配对族校验：ICL/blind 都是 `e3fc64c…`，Spatial fast/slow-only/base 都是 `227ba5c…`；不同 checkpoint 家族之间不要求 runner hash 相等，也不用于单变量因果归因。正式汇总还会核对 manifest hash、题目网格、未来边界、帧预算和配对指纹，从 manifest 重建每行 question/gold/options，并从 `raw_response` 重新解析选择；统计按 9 个 exact-content 组聚类。

- `runs/experiment_c_visual_v2_icl_separate_ep12_mcq.jsonl`
- `runs/experiment_c_visual_v2_blind_separate_ep12_mcq.jsonl`
- `runs/experiment_c_visual_v2_spatial_separate_ep12_mcq.jsonl`
- `runs/experiment_c_visual_v2_spatial_slow_only_separate_ep12_mcq.jsonl`
- `runs/experiment_c_visual_v2_spatial_base_separate_ep12_mcq.jsonl`
- `runs/experiment_c_visual_v2_spatial_separate_ep12_mcq.summary.json`
- `runs/experiment_c_visual_v2_spatial_slow_only_separate_ep12_mcq.summary.json`
- `runs/experiment_c_visual_v2_spatial_base_separate_ep12_mcq.summary.json`
- `runs/experiment_c_visual_v2_ep12.summary.json`（五臂配对正式汇总）
- `runs/experiment_c_visual_v2_ep12.md`（自动生成的表格报告）

使用 5 张 GPU 并行复现五个 arm：

```bash
ARMS=(icl blind spatial spatial_slow_only spatial_base)
for INDEX in "${!ARMS[@]}"; do
  ARM="${ARMS[$INDEX]}"
  CUDA_VISIBLE_DEVICES="$INDEX" conda run --no-capture-output -n meowbench \
    python scripts/evaluate_experiment_c_visual.py \
    --data data/experiment_c/manifest.json \
    --episodes 12 --max-frames 0 --max-side 448 --max-new-tokens 8 \
    --answer-mode mcq --mcq-layout separate --arm "$ARM" --device cuda:0 \
    --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
    --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
    --output "runs/experiment_c_visual_v2_${ARM}_separate_ep12_mcq.jsonl" &
done
wait

conda run --no-capture-output -n meowbench \
  python scripts/summarize_experiment_c_visual.py \
  runs/experiment_c_visual_v2_icl_separate_ep12_mcq.jsonl \
  runs/experiment_c_visual_v2_blind_separate_ep12_mcq.jsonl \
  runs/experiment_c_visual_v2_spatial_separate_ep12_mcq.jsonl \
  runs/experiment_c_visual_v2_spatial_slow_only_separate_ep12_mcq.jsonl \
  runs/experiment_c_visual_v2_spatial_base_separate_ep12_mcq.jsonl \
  --data data/experiment_c/manifest.json \
  --hash-model-artifacts \
  --output runs/experiment_c_visual_v2_ep12.summary.json \
  --report runs/experiment_c_visual_v2_ep12.md
```

## 4. V-JEPA 2 表征对照

JEPA 实验使用 Meta 官方冻结 V-JEPA 2 ViT-L/16 encoder，在完整 12 个 episode 上提取 16 帧时序特征，然后对每个题型做 leave-one-content-group-out kernel-ridge readout。每个测试 episode 的 readout 会排除所有完整可观测事件流与它相同的 episode，避免 exact duplicate 同时进入训练集和测试集。匹配的 pixel arm 对 `16×16 RGB` 像素特征使用完全相同的分组和 readout。三个 condition 分别有 10、10 和 9 个 content group，但仍评测全部 12 个 episode。

这是一项监督表征 probe。冻结 V-JEPA 2 本身没有语言 decoder、在线 fast weights 或持久记忆，因此它不是 TTT arm，不能与 Spatial-TTT 的无监督在线写入分数直接比较。

| condition | arm | exact | accuracy | seen-label exact | seen-label accuracy | prior expected |
|---|---|---:|---:|---:|---:|---:|
| relocated | V-JEPA 2 | 38/60 | 0.6333 | 38/54 | 0.7037 | 0.2240 |
| relocated | pixels | 40/60 | 0.6667 | 40/54 | 0.7407 | 0.2240 |
| same scene | V-JEPA 2 | 35/60 | 0.5833 | 35/54 | 0.6481 | 0.2240 |
| same scene | pixels | 40/60 | 0.6667 | 40/54 | 0.7407 | 0.2240 |
| filler only | V-JEPA 2 | 53/60 | 0.8833 | 53/60 | 0.8833 | 0.6946 |
| filler only | pixels | 52/60 | 0.8667 | 52/60 | 0.8667 | 0.6946 |
| **overall** | **V-JEPA 2** | **126/180** | **0.7000** | — | — | — |
| **overall** | **pixels** | **132/180** | **0.7333** | — | — | — |

| condition | arm | location | mover | owner | state | next action |
|---|---|---:|---:|---:|---:|---:|
| relocated | V-JEPA 2 | 1.0000 | 0.2500 | 0.5000 | 1.0000 | 0.4167 |
| relocated | pixels | 1.0000 | 0.3333 | 0.5000 | 1.0000 | 0.5000 |
| same scene | V-JEPA 2 | 1.0000 | 0.1667 | 0.3333 | 1.0000 | 0.4167 |
| same scene | pixels | 1.0000 | 0.3333 | 0.5000 | 1.0000 | 0.5000 |
| filler only | V-JEPA 2 | 1.0000 | 1.0000 | 0.4167 | 1.0000 | 1.0000 |
| filler only | pixels | 1.0000 | 1.0000 | 0.3333 | 1.0000 | 1.0000 |

Pixel control 在 relocated 和 same-scene 中都高于 V-JEPA 2；filler-only 中 V-JEPA 2 多答对 1 题，三种条件合计仍由 pixels 高 6/180。这说明当前 PNG 卡的文字、颜色和固定模板已经提供很强的低层统计线索；高 JEPA 准确率不能解释为更好的长期视觉记忆。filler-only 中 mover/state/next-action 分别固定为 `unknown/intact/unknown`，该条件的 group-aware sampled-label-prior 期望已经达到 0.6946，也会明显抬高总分。

此前 ordinary leave-one-episode-out 版本得到 V-JEPA 2 `142/180 = 0.7889`、pixels `148/180 = 0.8222`。由于它只排除测试 episode ID，训练 fold 可能仍包含与测试样本完全相同的观测事件流，存在 exact duplicate train/test leakage。这两个数值只保留为历史诊断，不能再作为 held-out 表征结果。

官方 checkpoint、资源占用、逐 fold 结果和复现命令见 [JEPA_BASELINE.md](JEPA_BASELINE.md)。正式结果文件为：

- `runs/experiment_c_jepa_v2_relocated_12ep_16f.json`
- `runs/experiment_c_jepa_v2_same_scene_12ep_16f.json`
- `runs/experiment_c_jepa_v2_filler_only_12ep_16f.json`

## 5. C-PNG 内部可以得出的结论

| 实验层次 | 当前证据 | 能回答的问题 |
|---|---|---|
| Symbolic fixture schema v2 | LaCT relocated 0.6383，same-scene 0.7333 | fast-weight 关联状态是否能跨 cue 变化保持 |
| Visual C-PNG strict MCQ | ICL 0.4944 vs blind 0.2667；Spatial fast 0.2667 vs slow-only 0.2833 | 直接视觉上下文是否有用，以及 test-time fast update 是否在同路径对照上增益 |
| Frozen V-JEPA 2 probe | group-excluded JEPA 0.7000，pixels 0.7333 | 冻结表征中标签是否可由监督 readout 恢复 |

当前最稳定的发现是：LaCT symbolic memory 确实保留了一部分状态，但 scene cue 变化会损害 owner 和组合式 next-action 读取；Qwen3-VL ICL 在 C-PNG 上明显高于配对 blind，说明卡片中有可读信息；官方 Spatial-TTT 的 episode fast update 比 slow-only 低 3/180，当前没有正向 memory 证据；V-JEPA 2 能提供可分特征，但没有超过相同 readout 下的低分辨率像素。这三类分数来自不同的监督和读取条件，不应直接排名。

本页的后续工作只用于进一步定位机制：

1. **Visual explicit**：在同一 16-card 协议上增加可审计的显式状态表/检索基线，将视觉读取错误与记忆写入错误分开。
2. **StreamTTT**：以相同 future-withheld、同帧序列和 fast/slow-only 消融接入预测式 TTT 对照，测量在线更新本身的增益。
3. **真实视频主评测**：扩展 [SVCBench matched-prefix 协议](SVCBENCH_REAL_VIDEO.md)的 source-video 覆盖与采样率消融；不将它与本页 C-PNG 分数做直接排名。
