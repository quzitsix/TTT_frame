# Idea 03：原生 Spatial 关联寻址的真实视频 forecasting 评测

日期：2026-10-25。正式运行目录：
`runs/forecast_bench_v0/idea03_native_spatial_v2/`。
全部 200 题已完成，先冻结预测，再独立评分。**本轮查询搜索未改善 forecasting：
test57 为 6/57，直接读取为 9/57，full ICL 为 21/57；错误视频搜索为 9/57。**
结果是探索性负结果，不能据此否定完整 Idea3，也不能推出 RL 必然更好。

## 研究问题与实现范围

检验对固定 fast weights 优化临时查询地址，能否改善六选一未来动作预测，并区分
搜索目标上升与真实动作正确率上升。采用本地 **官方 Spatial-TTT-nano checkpoint**
的原生 21 层 SwiGLU 点积关联记忆，未把 Idea1 的 MSE 预测 MLP 套进关联目标。

本轮实现是 **原生 Spatial 关联寻址诊断**，不是完整 Idea 卡片的端到端训练系统。
本仓库跨视频 chunk 丢弃 KV、只传递 fast state 的参数记忆路径也不是官方完整混合
流式协议。没有新增 benchmark 离线训练；候选的 text-V 是未经视觉语义对齐训练的
代理目标，其有效性正是需要检验的限制。不能把这轮结果称为完整 Idea3 的上限。

新增文件：

- `ttt_frame/idea3_spatial_reader.py`：原生 Q/K/V 捕获、只读状态快照、候选编码、
  批量独立地址搜索和视觉关联诊断。
- `scripts/run_idea3_spatial.py`：严格前缀重放和各对照的逐题分数。
- `scripts/launch_idea3_spatial.py`：独立分片运行与完整性检查后合并。
- `scripts/evaluate_idea3_spatial.py`：validation 选择、冻结预测、随后独立评分。
- `scripts/summarize_idea3_spatial.py`：检查和归档。
- `scripts/audit_idea3_frames.py`：与原始 ICL 的实际帧哈希和时间戳逐题核对。

旧 `spatial_*`、`forecast_*`、Idea1/scaling 模块及旧脚本未修改，保留旧源码 SHA 协议。
基线重新核验见 [IDEA03_BASELINE_AUDIT.md](IDEA03_BASELINE_AUDIT.md)。

## 数据和因果协议

沿用 `/data/share/raw/bench_v0` 的 200 题，六选一、约 1830 秒可见前缀，预测其后
约五秒动作。固定分组为所有佩戴者共同的 recording day：train111（第 1–4 天）、
validation32（第 5 天）、test57（第 6–7 天）。不随机重分，不跨题累计状态。

推理只读取公开 `inputs_mcq.jsonl`、本题 `context.mp4`/`input.mp4` 和固定划分。
不读取答案、私有动作注释或 future 视频。所有候选使用同一算法，未编码字母/位置。
每题先保存所有分数；选择程序只使用 validation 标签；保存 200 题所有 arm 的预测
及 SHA 后，另一个 `--stage score` 调用才使用 test 答案。

wrong-video 使用按公开 ID 排序后第一个不同录制日的 train 前缀，映射提前写入协议。
validation/test 的 donor 均来自更早的训练日；train 第一天的 wrong 控制可以使用
第二天的离线训练素材，仅作错误记忆反事实诊断。donor 不随答案或运行顺序改变。
接收题保留自己的查询位置偏移。donor 不是当前视频证据，也不是额外主模型输入。

test57 已被反复用于探索，标签为银标签且窗口重叠。两天测试不足以做可信的录制日
聚类显著性判断；报告正确数和配对得失，不把 57 道题当 IID 来给出显著性结论。

## 写入与读取的精确定义

写入完全沿用旧 Spatial baseline：历史 30 帧、近期 60 帧，分别取均匀左端点，
max side 224；分角色打包、每次约 8 帧，共 12 次写入。保持官方点积关联 ascent、
Muon、momentum、空间卷积和训练好的 slow 参数。原写入包含视频输入序列中的
包装文本 token，本轮未更改这一定义。每题独立 reset，写入后冻结状态。

`W0` 是该官方 checkpoint 的 learned initialization，不是全零矩阵。
只读快照只复制 weights/norms，不保留 optimizer momentum/pending 队列。

文本地址提取使用 reset 状态，保留接收题的位置偏移。纯文本前缀为：

```text
Question: <公开问题>
Candidate action: <结尾保留一个空格>
```

question-only 前缀最后一个 token 的原生 Q 提供每层、每 head 的初始地址 `q0`。
每个候选单独接在此前缀后，取最后一个真实候选 token 的原生 V，按 head 归一化为
`v_a`。没有聊天后缀、选项字母或候选编号。一次 question-only、六次独立候选前向，
同时计算 reset 模型的候选 token 平均条件 log likelihood，作为所有 arm 共用 prior。

每层的目标为：

```text
delta(q,a) = sum_heads <f_W(q), v_a> - <f_W0(q), v_a>
J(q,a) = delta(q,a) - 0.01 * sum_heads ||q-q0||²
q <- project_L2_ball(q0, radius)(q + 1.0 * grad_q J)
```

搜索三步，每个 head/token 单独球约束，半径候选为 0.1、0.5、1.0。
每个候选独立保留包含初始地址在内的最佳迭代，W/W0 全程只读；结束即丢弃查询。
六候选沿 token 轴批量执行，保留各自独立梯度和最佳迭代；测试对照了原逐候选实现。
最后除以 head 数并对 21 层取平均。`scores.search_*` 保存含正则的 `J`；完整诊断
另外保存未正则化 association，避免把两者混称。

直接读的分数是 `J(q0,a)=delta(q0,a)`。混合读出为：

```text
score_a = zscore_within_question(prior_a)
        + beta * zscore_within_question(memory_objective_a)
```

只在本题六候选内标准化；没有跨题/test 统计。标准差不超过 `1e-12` 时返回零。
beta 网格为 0、0.01、0.1、1、10、100。validation 正确数最高者胜出；平局选较小
beta，再选较小半径。beta=0 允许 validation 拒绝记忆贡献。精确分数平局按候选
文本排序，仅相同文本才用字母。选项重排测试覆盖编码、搜索、标准化和最终选择。

## 对照与解释方式

| Arm | 定义 |
|---|---|
| prior / reset / frozen_init | 同一个冻结初始化候选似然；reset 的 association delta 实测严格为零 |
| direct | 原始 q0 读取；独立 validation 选择 beta |
| direct_matched | 原始 q0 读取；使用 search 选定的相同 beta，定位搜索本身的差异 |
| search | 三步查询搜索；validation 选择半径和 beta |
| wrong_direct / wrong_direct_matched | 换为固定错误 train 视频的状态，系数沿用对应本题记忆对照 |
| wrong_search | 同样的 search 设置与 prior，但换成错误视频状态 |
| direct_raw / search_raw / wrong_raw | 只用记忆分数的诊断；search 使用混合 arm 选出的半径，不独立调参 |
| full ICL | 原始 Qwen 的旧 90 帧六选一预测，逐题匹配；不重新调用付费 API |

`direct` 与 `search` 比较各自 validation 最佳完整方法；`direct_matched` 与 `search`
比较相同 beta 下的读取变化。wrong 控制必须和对应相同系数的当前视频结果配对。
若搜索 association 大幅上升而 wrong-video 也同样受益，不能解释为读取到本视频事实。

额外的 visual-key 诊断对最后一个已观察视觉 V，比较原视觉 K、公开问题的 Q 和
搜索后的 Q。它只检查前缀内的局部关联可访问性：未来动作问题与最后视觉 token
**不是同一事实的文本/视觉描述**，因此差距不能证明正确语义事实已经写入，也不能
证明长延迟记忆保留。诊断分数不是 cutoff 后 forecasting 成绩。

## 结果和成本

validation 选定直接读取 `beta=1`；搜索 `beta=1, radius=0.5`、三步。因此本轮
`direct_matched` 与 `direct` 完全一致。test 标签仅在预测冻结后的评分调用中使用，
未根据 test 更换配置。以下各行均使用相同题集；validation 数字是选择集成绩。

| 方法 | Validation32 | 探索性 test57 | Test 准确率 |
|---|---:|---:|---:|
| 原始 Qwen full ICL，90 帧 | 6/32 | 21/57 | 36.84% |
| 直接读取 / direct_matched | 8/32 | 9/57 | 15.79% |
| 查询搜索，三步，r=0.5 | 6/32 | 6/57 | 10.53% |
| 重置 / 冻结初始化 / prior | 4/32 | 3/57 | 5.26% |
| 错误视频直接读取 | 7/32 | 10/57 | 17.54% |
| 错误视频查询搜索 | 2/32 | 9/57 | 15.79% |
| 纯记忆直接读取（诊断） | 3/32 | 10/57 | 17.54% |
| 纯记忆搜索（诊断） | 2/32 | 10/57 | 17.54% |
| 错误视频纯记忆搜索（诊断） | 1/32 | 15/57 | 26.32% |

纯记忆结果保留用于诊断，没有据其 test 成绩改选主方法。当前 prior 的 3/57 使用
候选平均条件似然，与旧官方 checkpoint 的 A–F next-token blind 读出 11/57 不同；
两者不能直接用来解释在线记忆收益。重置的 association delta 在全部 200 题精确为零。

配对“改对”指右侧对照错误、左侧方法正确；“改错”含义相反，详细题目 ID 见归档。

| Test57 配对 | 改对 | 改错 | 净增正确数 |
|---|---:|---:|---:|
| 搜索 vs full ICL | 3 | 18 | -15 |
| 直接读取 vs full ICL | 5 | 17 | -12 |
| 搜索 vs 直接读取 / direct_matched | 3 | 6 | -3 |
| 错误视频直接读取 vs 当前视频直接读取 | 4 | 3 | +1 |
| 错误视频搜索 vs 当前视频搜索 | 5 | 2 | +3 |

搜索对所有候选的正则化目标均未下降，test342 个候选的平均目标增量为 0.328232，
但正确数降低。当前视频的两种读出均未超过对应错误视频，未建立视频特异性收益。
这说明当前代理目标与最终动作任务之间仍有缺口，尚不能定位为单独的写入或读取问题。

额外前缀诊断中，200 题平均 association：最后视觉 K 为 0.867523，问题 Q 为
0.060041，优化后 Q 为 0.554010。优化能提高该局部目标；由于问题与最后视觉 token
不描述同一事实，这些值不能证明未来动作信息已写入、长时保留或语义读取成功。

**输入核验通过 200/200**：与旧 Qwen full ICL 的帧计划、全部/历史/近期像素 SHA、
请求和实际时间戳以及采样配置逐项一致。两者都覆盖约 1800+30 秒可见视频，
30+60 帧、无音频；本次每题 2205 个视觉 token、每层 12 次写入。核验不读取答案。

同帧比较消除了此前 Idea1 3660 帧对 ICL 90 帧的输入数量差异，但仍有以下差异：
原始 Qwen checkpoint 与官方 Spatial checkpoint 不同，读出不同，搜索与候选似然
使用更多查询计算。共享输入不等于同计算，也不足以宣布架构优劣。

原生 fast weights：66,060,288 个 FP32，264,241,152 bytes（252 MiB）；
包括 momentum 的实际在线状态为 528,998,736 bytes。模型、临时 reset/current/wrong
快照、query autograd 和文本前向显存另计。没有使用 bench train 标签拟合新读出；
validation 只选择上述混合系数和半径，官方 checkpoint 的原始训练仍是额外先验。

正式运行在单张 RTX 4090 D（cuda:4）上并发四个 worker，墙钟 **938.914 秒**。
以下为 200 题均值，均为共享设备上的测量区间，不是隔离部署延迟或 GPU 时间：

| 成本项 | 秒/题 |
|---|---:|
| 完整实验：解码、当前写入、文本、全部控制及诊断 | 16.808 |
| 视频解码 | 11.637 |
| 当前视频写入 | 2.741 |
| 七次文本前向提取（问题一次、候选六次） | 0.415 |
| 选定 r=0.5 的当前状态搜索（含直接读，21 层、六候选、三步） | 0.233 |
| 当前/错误状态各三个半径的全部六次搜索 | 1.401 |
| 额外视觉诊断 / reset 诊断 | 0.241 / 0.233 |

单个选定方法只需要当前写入、七次文本前向和一次半径搜索；它的各项可见成本不包含
全部控制，但没有单独计时端到端部署路径。四个 worker 各写两个 donor，共八次解码/
写入区间累计 119.142 秒，位于逐题时间之外；模型加载和启动同样不在逐题时间内。
逐题完整时间之和 3361.688 秒包含并发等待，不能当墙钟或 GPU-hours。
单 worker PyTorch allocated 高水位最大 6,147,055,104 bytes（5.725 GiB），是进程
累计高水位，不是每题重置的显存测量，也不含其他进程和全部 CUDA 开销。

旧 Qwen full ICL 记录的前向均值 0.371 秒、解码均值 12.073 秒，单次含全部选项的
字母读出，输入 2922–2961 tokens。旧运行硬件占用和并发不同，因此不报告速度比。

## 逐题归档与核验

完整归档：[data/bench_v0_idea03_spatial_v1](../data/bench_v0_idea03_spatial_v1)。

- `scores.jsonl`：200 题所有候选原始分数、三半径、错误记忆、reset、写入统计、
  像素/时间哈希和成本；未保存完整 fast-state 张量，复现时重放视频前缀。
- `predictions.jsonl`：评分前冻结的 200 题预测；`validation_selection.json` 保存完整网格。
- `per_case.csv` / `per_case.jsonl`：200 题各方法预测和正确性，包括 full ICL。
- `summary.json`：全部划分指标、配对计数及得失题目 ID；`results.md` 为自动汇总。
- `inference_protocol.json` / `protocol.json` / `split_manifest.json`：模型、源码、数据、
  种子与配置；`manifest.json` 校验核心文件，`supplement_manifest.json` 校验附加审计日志。
- `provenance/`：帧审计、启动/分片完成凭据、donor 审计、日志及最终测试记录。

归档保持原始字节（包括 CSV 的标准 CRLF 换行）；局部 `.gitattributes` 禁止自动
换行转换，以免 checkout 改变 SHA。附加 wrong/current 配对见
`provenance/wrong_memory_pairs.json`，由已评分逐题正确性重新汇总。

推理协议 fingerprint：`5f70467da3daff0e402754059c80b9604d9293838674844c5c3a8e98e8292c97`。
完整分数 SHA256：`07c76a9786a0d178081a97d1faa2495100022f0c459138bd525bc7b55d670ba6`。
独立复算不调用 evaluator 数值函数，从原始分数重新计算标准化/选择，核验了 200 题
预测、39 组指标及 144 组配对计数/ID，并重新检查全部像素/时间证据，均一致。

## 复现

本次基础 Git SHA：`2cd12bfcc72d68694e027f87fb78d6337cc387c3`；新增源码 SHA
记录于 inference protocol，最终 Git 提交包含全部新增实现。种子 0。
本地原始模型 `/data/quzitsix/models/Qwen3-VL-2B-Instruct`，官方权重
`/data/quzitsix/models/Spatial-TTT-nano/model.safetensors`，后者 SHA256：
`5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947`。
模型其他文件及 tokenizer 的 SHA 也保存在嵌套 baseline protocol。

环境为 conda `meowbench`，Python `3.11.16`、PyTorch `2.9.1+cu128`、
Transformers `4.57.6`、PyAV `18.1.0`，bfloat16 主干、FP32 fast weights。
无需安装依赖或付费模型服务。旧 V1 一题 smoke 记录保留在 git 忽略的 runs 中；正式
V2 使用批量候选搜索，不能混用 V1/V2 源码协议。

下面使用新的输出目录重跑，先检查 GPU4 当前可用显存（启动器要求每 worker 8 GiB）：

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python scripts/run_idea3_spatial.py \
  --prepare --protocol runs/forecast_bench_v0/idea03_reproduce/protocol.json
conda run --no-capture-output -n meowbench python scripts/launch_idea3_spatial.py \
  --protocol runs/forecast_bench_v0/idea03_reproduce/protocol.json \
  --output-dir runs/forecast_bench_v0/idea03_reproduce --device cuda:4 --workers 4 --threads 2
conda run --no-capture-output -n meowbench python scripts/evaluate_idea3_spatial.py \
  --stage predict --scores runs/forecast_bench_v0/idea03_reproduce/scores.jsonl \
  --inference-protocol runs/forecast_bench_v0/idea03_reproduce/protocol.json \
  --split-manifest runs/forecast_bench_v0/research_day_split.json \
  --output-dir runs/forecast_bench_v0/idea03_reproduce/evaluation
conda run --no-capture-output -n meowbench python scripts/evaluate_idea3_spatial.py \
  --stage score --scores runs/forecast_bench_v0/idea03_reproduce/scores.jsonl \
  --inference-protocol runs/forecast_bench_v0/idea03_reproduce/protocol.json \
  --split-manifest runs/forecast_bench_v0/research_day_split.json \
  --output-dir runs/forecast_bench_v0/idea03_reproduce/evaluation
conda run --no-capture-output -n meowbench python scripts/audit_idea3_frames.py \
  --scores runs/forecast_bench_v0/idea03_reproduce/scores.jsonl \
  --donor-audits runs/forecast_bench_v0/idea03_reproduce/scores_{0,1,2,3}.donors.json \
  --output runs/forecast_bench_v0/idea03_reproduce/frame_match_audit.json
conda run --no-capture-output -n meowbench python scripts/summarize_idea3_spatial.py \
  --run-dir runs/forecast_bench_v0/idea03_reproduce/evaluation \
  --output-dir data/bench_v0_idea03_reproduce
```

验证测试：

本轮最终完整测试为 **668 passed in 65.08s**。覆盖旧实验回归及新增的候选搜索
等价性、状态不变、候选重排、reset 零分、临时捕获恢复、错视频规则、因果标签隔离、
模型/源码/数据协议、分片完整性和像素审计。

```bash
conda run --no-capture-output -n meowbench python -m pytest -q \
  tests/test_spatial_probe.py tests/test_idea3_spatial_reader.py \
  tests/test_run_idea3_spatial.py tests/test_launch_idea3_spatial.py \
  tests/test_evaluate_idea3_spatial.py tests/test_summarize_idea3_spatial.py \
  tests/test_audit_idea3_frames.py
conda run --no-capture-output -n meowbench python -m pytest -q
```

## 对后续 RL 研究的约束

本轮地址优化是推理搜索，不是测试时 RL。它没有从真实后续观测接收语义奖励，也没有
训练策略或多步信用分配。即使优化目标下降/association 上升，也不自动形成事实奖励。
后续 RL 应检验有合法延迟反馈的写入/保留/遗忘/获取决策，并与相同反馈的监督 gate、
CE/MSE、固定/随机策略及显式检索比较。

更新后的 [ideax_input.md](../ideax_input.md) 定义这些要求；旧文逐字备份为
[ideax_input.pre_rl_20261025.md](archive/ideax_input.pre_rl_20261025.md)。
原始 `ideax_output/` 保持不变。
