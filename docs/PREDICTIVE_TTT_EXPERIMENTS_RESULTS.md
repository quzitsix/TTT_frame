# Predictive-TTT 实验总报告：A、B、C

本文汇总当前仓库中预测型 TTT 的三个实验方向。A 是 FHO-STA 短期预测边界；B 保留 HomeSentinel 与早期 SVCBench 流式状态实验；当前主要视觉 C 评测是真实视频 matched-prefix 状态/计数，旧 C-PNG 受控搬移与 V-JEPA 结果保留为机制/表征诊断。已完成的结果都保留了逐条 JSONL，或保存为带有逐条 `rows` 数组的 JSON；报告中的小数可以由对应脚本重新生成。

## 结论先行

实验 A 目前没有可报告的 Ego4D 数值：服务器没有可用的 FHO-STA 视频和标注，不能用不存在的数据填结果。

实验 B 的 HomeSentinel 快速结果显示，显式视觉 ICL 的 lexical token-F1 高于已有 Spatial-TTT 子集，但两者视觉帧预算不同，不能据此判断 ICL 优于 TTT。完整视频采样的 ICL 为 0.2001，已有 Spatial-TTT 为 0.1660，无记忆读出为 0.1730；这更像是“当前 TTT readout 没有体现稳定增益”的信号。

SVCBench 早期 smoke 只下载了一个约 14.8 MB 的 `simulated/0030.mp4` 和标注，没有下载约 80 GB 的完整数据集。它保留了 47 个原始 query points 中每四个取一个的 13 点轨迹，但 ICL 只看最近 8 秒而 Spatial-TTT 读取完整前缀，因此是 **legacy unmatched smoke**。其 ICL 与 Spatial-TTT 的 question-macro exact accuracy 都为 0.1250，只作历史诊断，不是当前主要 SVCBench 结果。

当前主要视觉 C 评测已切换到 [SVCBench 真实视频 matched-prefix 协议](SVCBENCH_REAL_VIDEO.md)。Natural9 的 9 段视频、42 条 trajectories、125 个 query points 已完成，strict summarizer 验证 **625/625** 行通过且无解析/运行失败。question-macro exact 为：blind 0.3060、ICL 0.2889、Spatial fast 0.3131、slow-only 0.2980、base bypass 0.3060。Spatial fast 相对 slow-only 只增加 1/125 个净正确点（query delta +0.0080，question-macro +0.0151；left-only/right-only 6/5），来源视频上 3 个为正、2 个为负、4 个为零，不能视为稳定优势。真实视频输入排除了“Spatial 只是不适合文字卡”这一单一解释，但当前 writer/readout 仍没有形成可靠的精确计数记忆。

旧 C-PNG fixture schema v2 已完成无未来泄漏的机制诊断。Symbolic probe 中 LaCT 的 relocated 为 0.6383，same-scene 为 0.7333；严格四选一中 ICL 为 0.4944、blind 为 0.2667、Spatial fast 为 0.2667、slow-only 为 0.2833。冻结 V-JEPA 2 与匹配 pixel readout 分别为 0.7000 与 0.7333。这些分数分别测试合成写入/读取链路和监督表征可读性，不是真实视频性能，也不与 C-real 分数直接比较。

## A：FHO-STA 短期预测边界

当前状态是 **blocked by data availability**，不是模型失败。`docs/FHO_STA_EXPERIMENT_A_RESULTS.md` 保留了完整配置与复现命令，但本机没有 Ego4D FHO-STA 的可读 annotation/video，因此没有生成 noun/verb accuracy，也没有填入估计数字。等 Ego4D 数据可用后，应固定相同观测边界和帧预算，再比较 blind、ICL-4f/8f 与 Spatial-TTT-4s8f/8s4f，并按 horizon 分层。

## B：长历史与结构化 counting

### B1 HomeSentinel

评测使用 `/data/HomeSentinel/asuka` 的现有视频和 benchmark queries，选取 `video_cutoff_idx <= 10` 的 15 个 event/home 查询、10 个视频。ICL runner 只取每个观察视频 2 帧，历史上限为 32 帧；quick 版只在每个视频的前 60 秒取样，full-video 版在整个已观察视频上均匀取样。两者都使用 Qwen3-VL-2B native video processor、max-side 448、cuda:2。

同一批 query 在已有 Spatial-TTT run6 中筛出对应的 `memory` 与 `without_memory` 行：run6 使用 60 s/4 frames 的视频块，在完整 asuka stream 上写入 Spatial fast weights。它的可用性和上下文范围更大，因此与 ICL 不是严格等帧比较。

| condition | n | mean token-F1 | exact | gold substring |
|---|---:|---:|---:|---:|
| ICL，first 60 s quick | 15 | 0.2148 | 0 | 0 |
| ICL，full-video sampling | 15 | 0.2001 | 0 | 0 |
| Spatial-TTT memory，run6 子集 | 15 | 0.1660 | 0 | 0 |
| Frozen/no-memory readout | 15 | 0.1730 | 0 | 0 |

这些分数是开放式回答的 token overlap 诊断，不是人工语义准确率。ICL full-video 相比 Spatial-TTT 的 paired mean token-F1 差为 +0.0340；Spatial-TTT 相比 frozen readout 为 -0.0070。结论应是：当前实验没有显示稳定的 TTT 增益，下一步应先做等帧数、等观测边界的 Spatial-TTT 对照，再调 writer/readout。

详细表格和逐题结果见 [`docs/EXPERIMENT_B_RESULTS.md`](./EXPERIMENT_B_RESULTS.md)。runner 是 [`scripts/evaluate_homesentinel_icl.py`](../scripts/evaluate_homesentinel_icl.py)。

### B2 legacy unmatched SVCBench structured-counting smoke

通过官方 [SVCBench repository](https://github.com/buaa-colalab/SVCBench) 和 [dataset card](https://huggingface.co/datasets/buaaplay/SVCBench) 的公开文件，只选择 `simulated/0030.mp4`。下载内容为：

- `videos/simulated/0030.mp4`：约 14.8 MB；
- `vcbench_eval.jsonl`：约 1.2 MB；
- 两个问题，O2-Unique 33 点、E1-Action 14 点，共 47 个原始点。

报告的 quick smoke 每 4 个点取 1 个，实际评估 13 点（O2-Unique 9 点、E1-Action 4 点），max-side=224、chunk=4 s/8 frames。所有 query 都检查 `timestamp <= query_time`，没有把未来帧送入模型。ICL 只看最近 8 s/8 frames；Spatial-TTT 按时间增量写入 prefix 后用 text-only answer 读取 fast weights。

| condition | questions | valid/queries | coverage | question-macro exact accuracy | GPA | MoC | UDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICL recent window | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.5625 | 0.3542 |
| Spatial-TTT full prefix | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.8750 | 0.0625 |

这是小样本、低分辨率且视觉前缀不匹配的历史 smoke，不代表完整 SVCBench。曾启动 448 px/47 点版本，但单点成本过高且没有产生可用结果，已停止；不能把它写成“完整轨迹已完成”。当前主要 SVCBench 评测见 [C-real matched-prefix 协议](SVCBENCH_REAL_VIDEO.md)。

## C：当前主要真实视频 matched-prefix 评测

当前 C 使用 Natural9 多来源真实图像视频和 SVCBench 的结构化计数标注，另保留 3 段 Ego4D 作为严格第一人称子集。每个 query point 作为独立 episode，blind 和 ICL 共享文字时间先验，ICL 与三个 Spatial 读取状态共享相同的截止时间、帧集合与顺序。数据 provenance、五臂协议、审计规则、运行命令和结果统一见 [`SVCBENCH_REAL_VIDEO.md`](SVCBENCH_REAL_VIDEO.md)。

| arm | query exact | question macro | video macro | family macro | UDA |
|---|---:|---:|---:|---:|---:|
| blind | 0.2400 | 0.3060 | 0.2849 | 0.2995 | 0.1927 |
| ICL | **0.2640** | 0.2889 | 0.2794 | 0.2776 | **0.4427** |
| Spatial fast | 0.2400 | **0.3131** | **0.2887** | **0.3062** | 0.1849 |
| Spatial slow-only | 0.2320 | 0.2980 | 0.2664 | 0.2900 | 0.2083 |
| Spatial base bypass | 0.2400 | 0.3060 | 0.2849 | 0.2995 | 0.1927 |

ICL 的 query micro 和 UDA 较高，但 question/video/family macro 都略低于 blind；其 `model.generate` 多模态路径与 blind 的手写 text-only greedy 路径也不完全相同，因此只作视觉基线描述。主要 TTT 比较使用同一 checkpoint、写入流和读取路径：Spatial fast − slow-only 的 query/question/video/family delta 分别为 +0.0080、+0.0151、+0.0223、+0.0162。这个小正差由 6 个 fast-only 正确点和 5 个 slow-only 正确点构成，同时 UDA 从 0.2083 降到 0.1849，证据是混合而不稳定的。

计数标定诊断也显示明确的输出坍缩：gold 范围为 0–47，Spatial fast 的 125 个预测中 59 个为 1、63 个为 2、3 个为 3，MAE 3.368；ICL 的范围为 1–10、MAE 2.784。也就是说，当前 fast state 偶尔改变 exact 对错，却没有学会随长前缀可靠累积数值。

SVC counting 和下面 C-PNG 的物体搬移/下一步问答是不同任务，只比较各自协议内的配对 arm，不比较两页的绝对分数。

## C-PNG：受控搬移与重排的历史机制诊断（fixture schema v2）

fixture schema v2 共有 12 个 episode。每个 episode 在三种条件下都提供 16 张观测卡和 27 次结构化 key/value 写入，并包含 location、mover、owner、state、next action 五类问题。`future_event` 与观测 `events` 分开存储，评测时不渲染、不编码，也不写入记忆；mover 与 owner 强制不同。`relocated`、`same_scene_control` 和 `filler_only_control` 使用匹配的观测与写入预算，避免 legacy fixture schema v1 中控制条件不等长的问题。

这套 fixture 仍是机制 smoke，而不是独立同分布的真实视频基准。12 个 episode 在任一条件出现相同观测流就归为同组后，跨三种条件的并集形成 **9 个 exact-content clusters**；这些 cluster 仍共享生成器规律，不能当作 9 个独立随机样本。此外，`moved_object`、最终 location 和最终 state 按 episode 索引的三周期共同变化；filler 卡还直接显示 `control distractor`、`actor: control` 和 `control_records: 4` 等内部控制字段。`relocated` 与 `same_scene_control` 除背景外还把 `living room shelf` 改成 `shelf`、把 `side table` 改成 `table`，因此并非只改变场景线索的纯干预。这些规律都可能形成不依赖长期记忆的标签捷径。当前 C-real 主评测使用另一个真实视频计数任务，它不是这套 fixture 的修订版。

### C1 symbolic mechanism probe：12 episodes × 10 trials

`scripts/evaluate_experiment_c.py` 使用 64 维稳定 semantic code，在 `scene_drift_strength=0.35` 下比较结构化 oracle、固定大小显式检索表、LaCT fast-weight 关联记忆和无写入 control。每个 arm/condition 有 600 个判断，四个 arm、三种条件合计 7,200 行。

| arm | relocated | same-scene | filler-only | coverage |
|---|---:|---:|---:|---:|
| oracle | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| explicit retrieval | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| LaCT | 0.6383 | 0.7333 | 0.4967 | 1.0000 |
| none | — | — | — | 0.0000 |

LaCT 汇总到三种条件后的各问题准确率为：location 0.8111、mover 0.8528、owner 0.3028、state 0.6833、next action 0.4639。relocated − same-scene 的配对差值分别为 -0.0500、-0.0167、-0.2000、0、-0.2083；总体差值也是 0.6383 − 0.7333 = **-0.0950**。因此，修正未来泄漏和预算后，没有复现 legacy fixture schema v1 中“换场景反而显著提高”的结果；当前 scene cue drift 尤其损害 owner 和 next-action 绑定。该实验隔离的是关联记忆机制，不包含视觉编码器。

正式产物为 `runs/experiment_c_v2_final.jsonl` 和 `runs/experiment_c_v2_final.summary.json`。

### C2 C-PNG 严格四选一视觉评测：12 episodes × 3 conditions

每个 arm 评测 12 个 episode、三种条件和五类问题，共 180 题。每题有四个唯一候选项，gold 的 A/B/C/D 位置各出现 45 次；主指标是包含错误和解析失败在分母内的 choice accuracy，随机机会水平为 0.2500。五个 arm 都采用逐题 `separate` MCQ；除无视觉输入的 blind 外，各视觉 arm 使用相同的 16 张唯一可观测卡，future card 始终隐藏。

- `ICL` 在每道题中直接提供 16 张卡；`blind` 只提供题目和选项。
- `Spatial-TTT` 先把 16 张卡写入 episode fast state，再用纯文本问题读取。
- `Spatial slow-only` 完成相同 ingest 后清空 episode fast state，但仍走官方 Spatial checkpoint 的读取路径。这是隔离视频 fast update 的主要对照。
- `base-attention bypass` 同样完成 ingest，但回答时绕过整条 Spatial 读取分支；它不是纯 fast-memory 消融，只作为次要诊断。

| arm | relocated | same-scene | filler-only | overall | parse rate |
|---|---:|---:|---:|---:|---:|
| Qwen3-VL ICL | 0.5167 | 0.4833 | 0.4833 | **89/180 = 0.4944** | 0.9833 |
| text-only blind | 0.1833 | 0.2333 | 0.3833 | **48/180 = 0.2667** | 1.0000 |
| Spatial-TTT，episode fast state | 0.3000 | 0.3500 | 0.1500 | **48/180 = 0.2667** | 1.0000 |
| Spatial slow-only | 0.3333 | 0.3500 | 0.1667 | **51/180 = 0.2833** | 1.0000 |
| base-attention bypass | 0.3000 | 0.3333 | 0.1833 | **49/180 = 0.2722** | 1.0000 |

ICL 相对 blind 增加 41/180，paired accuracy delta 为 +0.2278；按 9 个 exact-content clusters 枚举的 sign-flip `p=0.00781250`。由于 cluster 之间仍共享确定性生成周期，这个 p 值只作 fixture 内诊断；查询级 McNemar `p=0.000003` 的独立性假设更弱，也只作探索性统计。ICL 的三条不可严格解析回答已按错误计入分母。

完整 Spatial 相对 slow-only 少对 3/180，paired delta 为 -0.0167（left-only/right-only 1/4，content-group `p=0.500000`）；说明当前视频 fast-state 写入没有带来可检测的增益。完整 Spatial 相对 base-attention bypass 为 -1/180（content-group `p=1.000000`），但这组比较同时改变了整个 Spatial 读取分支，不能单独归因于 fast weights。Spatial 输出还明显偏向 A：gold A 只有 45/180，而完整 Spatial 预测了 130 次 A。

除 blind 外，各视觉 arm 的唯一观测集都是同一 16 张卡，但计算预算仍不相等：逐题 ICL 每个 condition 重复编码五次，共 80 次帧呈现；三个 Spatial arm 只 ingest 一次，共 16 次帧呈现。ICL 使用原始 Qwen3-VL，Spatial 还加载了官方 Spatial checkpoint，因此跨系统的绝对分数不能只解释为“是否使用记忆”。主要因果结论来自同一官方 checkpoint 下的 Spatial 与 slow-only 配对。

视觉 runner 行内记录 Qwen 模型路径和 config fingerprint；正式汇总使用 `--hash-model-artifacts` 额外保存 Qwen 权重、processor/tokenizer 文件和 Spatial checkpoint 的逐文件 SHA-256，并核对 Spatial 行内 checkpoint hash。JEPA cache 也包含 checkpoint、官方源码树、runner 与字体指纹；relocated 额外哈希输入 PNG，两个内存渲染 control 则由 manifest、runner 和字体共同固定。因此正式产物具备字节级模型清单，跨机器复现时仍应以清单而不是路径为准。

本次 Qwen 模型目录 tree SHA-256 为 `b6e2cc74528a784288866831a2c80c21be0a65b2479e7212bcc7243c3081ab9b`，其中 `model.safetensors` 为 `7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0`；Spatial-TTT checkpoint 为 `5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947`。

正式校验汇总为 `runs/experiment_c_visual_v2_ep12.summary.json` 和 `runs/experiment_c_visual_v2_ep12.md`。ICL 的 180 行来自当前 runner 的三个互不重叠 4-episode shard；blind 使用同一 runner 和同一文字提示重新运行，二者只相差视觉卡输入。五臂都直接记录协议元数据；runner hash 在配对族内一致（ICL/blind 为 `e3fc64c…`，Spatial 三臂为 `227ba5c…`）。汇总器对 manifest hash、题目网格、未来边界、帧预算与配对指纹逐行校验，还会从 manifest 重建 question/gold/options 并重解析 `raw_response`，最后按 9 个 exact-content 组聚类。旧 batched Spatial 因只生成一个字母而无法解析，只保留为输出格式诊断，不进入性能表。

### C3 V-JEPA 2 冻结表征对照：全 12 episodes

V-JEPA 对照加载 Meta 官方 V-JEPA 2 ViT-L/16 checkpoint，把每个 16-card 序列编码成冻结特征，再为五种问题分别训练 leave-one-content-group-out centered kernel-ridge readout。每个测试 episode 的训练 fold 会排除所有具有相同完整观测事件流的 episode，避免 exact duplicate 同时进入训练和测试。相同分组和 readout 也作用于 `16 frames × 16 × 16 RGB` 的低分辨率 pixel feature。每个 condition 有 12 episodes × 5 questions = 60 个 held-out 判断；relocated、same-scene、filler-only 分别包含 10、10、9 个 condition 内 content group。

| condition | V-JEPA 2 + readout | pixel + same readout | V-JEPA seen-label | pixel seen-label | prior expected |
|---|---:|---:|---:|---:|---:|
| relocated | 38/60 = 0.6333 | 40/60 = 0.6667 | 38/54 = 0.7037 | 40/54 = 0.7407 | 0.2240 |
| same-scene | 35/60 = 0.5833 | 40/60 = 0.6667 | 35/54 = 0.6481 | 40/54 = 0.7407 | 0.2240 |
| filler-only | 53/60 = 0.8833 | 52/60 = 0.8667 | 53/60 = 0.8833 | 52/60 = 0.8667 | 0.6946 |
| **all** | **126/180 = 0.7000** | **132/180 = 0.7333** | — | — | — |

Pixel control 在 relocated 与 same-scene 都高于 V-JEPA，V-JEPA 只在 filler-only 多对一题。V-JEPA 的 relocated − same-scene 为 +0.0500，即 60 题中的三题；pixel 差值为 0。filler-only 的高分也不是长期记忆证据：该条件中 mover、state 和 next action 是固定 `unknown/intact/unknown` 控制标签，使 group-aware sampled-label prior expectation 已达到 0.6946。

这是一项**监督表征 probe**：每个 fold 的 readout 只使用其他 content group 训练。它不是 TTT，也没有持续在线状态；其监督条件与 C2 的零样本 VLM 四选一不同，因而不能把两组绝对分数直接排名。低分辨率 pixel control 的总分更高，也说明文字卡模板和上述生成规律足以暴露大量标签信息。部署、checkpoint 校验、逐问题结果和显存记录见 [`docs/JEPA_BASELINE.md`](./JEPA_BASELINE.md)。

### C4 legacy / leakage diagnostics

legacy fixture schema v1 的 full-history PNG-card 输入含有显式 forecast target card，控制条件和写入预算也未完全匹配。旧 symbolic 结果 LaCT relocated 0.8833、same-scene 0.5611，以及旧 open-answer 视觉结果 ICL token-F1 0.3455、Spatial-TTT 0.0182，现仅保留为数据流与泄漏诊断，**不得与正式 C-PNG 表混用，也不得作为预测能力证据**。fixture schema v2 早期使用普通 leave-one-episode-out 的 V-JEPA 142/180 和 pixel 148/180，也因 exact duplicate 观测流跨越训练/测试 fold 而降级为历史泄漏诊断。在 C-PNG 的监督表征诊断内，只采用 C3 的 leave-one-content-group-out 三条件结果。

## 复现命令

### HomeSentinel B

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 --sample-seconds 60 \
  --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10.jsonl --overwrite

conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 --sample-seconds 0 \
  --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10_fullvideo.jsonl --overwrite
```

### Legacy unmatched SVCBench B smoke

```bash
SVC_ROOT=/data/quzitsix/svcbench_subset bash scripts/download_svcbench_subset.sh
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n meowbench python scripts/evaluate_svcbench.py \
  --annotations /data/quzitsix/svcbench_subset/subset/svcbench_subset.json \
  --video-root /data/quzitsix/svcbench_subset \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 --max-side 224 --query-stride 4 \
  --output runs/svcbench_0030_stride4.jsonl
```

### C-real SVCBench matched-prefix

真实视频的下载、分片运行与 strict summarizer 命令见
[`SVCBENCH_REAL_VIDEO.md`](SVCBENCH_REAL_VIDEO.md)。只有通过该页所列协议审计的产物才进入当前主结果。

### C-PNG 机制诊断（fixture schema v2）symbolic、strict MCQ 与 V-JEPA

```bash
conda run --no-capture-output -n meowbench python scripts/prepare_experiment_c.py \
  --output-dir data/experiment_c --episodes 12 --seed 20261015
conda run --no-capture-output -n meowbench python scripts/evaluate_experiment_c.py \
  --data data/experiment_c/manifest.json --output runs/experiment_c_v2_final \
  --trials 10 --report runs/experiment_c_v2_final.md

# C-PNG strict visual MCQ；五臂都使用全 12 episodes 和逐题读取。
# 可把 ARM 分配到不同空闲卡并行执行；每个进程写独立文件。
GPU=7
for ARM in icl blind spatial spatial_slow_only spatial_base; do
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n meowbench \
    python scripts/evaluate_experiment_c_visual.py \
    --data data/experiment_c/manifest.json --episodes 12 --max-frames 0 \
    --max-side 448 --max-new-tokens 8 \
    --answer-mode mcq --mcq-layout separate --arm "$ARM" --device cuda:0 \
    --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
    --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
    --output "runs/experiment_c_visual_v2_${ARM}_separate_ep12_mcq.jsonl"
done

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

# Official frozen V-JEPA 2 + matched pixel readout, all 12 episodes.
# Runner hashes the complete observed event stream and automatically uses
# leave-one-content-group-out validation, excluding exact duplicates.
for CONDITION in relocated same_scene_control filler_only_control; do
  case "$CONDITION" in
    relocated) NAME=relocated ;;
    same_scene_control) NAME=same_scene ;;
    filler_only_control) NAME=filler_only ;;
  esac
  CUDA_VISIBLE_DEVICES="$GPU" conda run --no-capture-output -n meowbench \
    python scripts/evaluate_experiment_c_jepa.py \
    --data data/experiment_c/manifest.json --condition "$CONDITION" \
    --clip-scope pre_forecast --episodes 12 --clip-frames 16 \
    --device cuda:0 --dtype bfloat16 \
    --output "runs/experiment_c_jepa_v2_${NAME}_12ep_16f.json" \
    --feature-cache "runs/experiment_c_jepa_v2_${NAME}_12ep_16f_features.pt"
done
```

## 当前决策与下一步

1. 当前主要 C 评测是真实视频 matched-prefix SVCBench；在 strict summarizer 通过前不填入或解读未完成的数字。
2. 后续优先扩大 source-video 数，并在五臂共享观测边界的前提下做采样率/分辨率消融；不把同一视频中的相关 query 当作独立证据。
3. 旧 HomeSentinel 和 `simulated/0030` 结果因预算或视觉前缀不匹配而只作历史诊断；如重做，必须使用相同 query cutoff 和真实帧集合。
4. C-PNG 的 symbolic、五臂 MCQ 和 V-JEPA/pixel readout 只用于机制或表征诊断；继续扩大同类文字卡的价值有限。
5. 若将 V-JEPA、visual explicit state 或 StreamTTT 接入真实视频对照，必须使用与 Spatial 相同的观测边界，并单独报告其监督条件。
