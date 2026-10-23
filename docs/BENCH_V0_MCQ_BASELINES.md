# bench_v0：已有架构的完整选择题基线

本实验固定 `/data/share/raw/bench_v0` 的全部 **200 道六选一动作预测题**，建立后续
TTT 实验可复用的参照。它与 [Idea 01 的 latent 预检](IDEA01_BENCH_V0.md) 不同：
这里直接回答原始问题并计算选择题准确率，不将视觉预测 MSE 当作答题分数。

## 2026-10-23 实测结果

**九组均完成全部 200 题，共 1800 条有效预测，missing/invalid/error 全为零。**
结果支持视频对原始 Qwen 有帮助，但本次 Spatial 参数更新没有带来净准确率收益。

| 方法 | 全部 200 题 | 固定测试 57 题 |
|---|---:|---:|
| 原始 Qwen，无视频 | 26/200（13.00%） | 9/57（15.79%） |
| 原始 Qwen，近期 ICL | 57/200（28.50%） | 23/57（40.35%） |
| 原始 Qwen，完整采样 ICL | 58/200（29.00%） | 21/57（36.84%） |
| 官方 checkpoint，无视频 / bypass | 37/200（18.50%） | 11/57（19.30%） |
| 官方 checkpoint，近期 ICL / bypass | 34/200（17.00%） | 11/57（19.30%） |
| 官方 checkpoint，完整采样 ICL / bypass | 32/200（16.00%） | 10/57（17.54%） |
| Spatial，保留本题视频 fast 更新 | 35/200（17.50%） | 12/57（21.05%） |
| Spatial，去掉本题更新 / slow-only | 35/200（17.50%） | 13/57（22.81%） |
| Spatial，写入后 bypass / 相同位置 | 34/200（17.00%） | 11/57（19.30%） |

这里“完整采样”指历史 30 帧加近期 60 帧，并非完整逐帧视频。随机六选一期望
16.67%；不对这批有重叠的样本作独立同分布显著性结论。

### 结果怎样理解

1. **这批题不能靠原始 Qwen 的文字盲答取得很高分数。** 无视频 13.00%，
   完整 ICL 29.00%，提高 16 个百分点。全部题的完整历史相对仅近期只净多答对
   1 题；57 题测试子集上仅近期反而多答对 2 题，不能据此宣称长历史更有效。
2. **Spatial 记忆参与了计算，但这次没有转化成净预测收益。** 200 题的 fast
   与 slow-only 六选项 logits 均不同，平均最大绝对差 0.8836；其中 73 题
   改变最终选项。13 题由错变对、13 题由对变错，总分持平；测试子集改对 4
   题、改错 5 题。与 bypass 相比有 57 题改变选项，净多答对 1 题。
3. **官方 checkpoint 在此任务上有明显的 A–D 偏好。** 六个相关组各自的
   200 次回答中均没有 E/F。Spatial 的 A/B/C/D/E/F 分布为
   `95/32/39/34/0/0`，而 gold 为 `34/34/33/33/33/33`。原始 Qwen 的完整
   ICL 为 `44/40/41/23/25/27`。这是观测到的迁移/读出偏差；其训练来源原因
   尚未验证，不能把分数差全部归因于 TTT 更新规则。
4. **低分不是输出解析失败。** 全部 1800 条预测中，不受限最高 token 与受限
   A–F 选择完全相同，六字母总概率最低约 0.9996。模型确实在直接输出这些选项，
   没有把解释文本误判为答案。本轮没有根据结果重排选项、校准 logits 或调参。

所有 200 题都通过跨架构解码像素 SHA 一致检查；每题 Spatial 写入 90 个真实帧、
2205 个视觉 token，各层均记录正次数更新。日志差异只证明记忆影响，配对正确率
才衡量影响是否有用。后续 Idea 01 应至少同时比较原始 Qwen 的近期/完整 ICL，
并保留自己模型的 no-update、wrong-history 等消融；仅超过当前 17.5% Spatial
不能证明已经优于 ICL。

额外重放首题的参数审计确认：21 个 TTT 层共 63 个 `w0/w1/w2` 张量均实际改变，
各层合并相对 L2 变化为 0.849–1.184；不是只有更新计数增加。`reset()` 后位置
从 968 回到 0，运行时状态清空，重新构造的有效初始权重与写入前逐位相同。
该单题验证见 [`memory_state_audit.json`](../data/bench_v0_baseline_v1/memory_state_audit.json)，
不能将它扩展成对每题权重变化幅度的测量。

### 运行与归档

正式运行于北京时间 2026-10-23 21:50:28–22:04:51 完成，约 **14 分 22 秒**。
使用 GPU 2/3/4/5，八个 worker；其他卡上已有任务。本机环境为 Python 3.11.16、
PyTorch 2.9.1+cu128、Transformers 4.57.6。测试 `python -m pytest -q`：
**377 passed**。

并发运行下，原始 Qwen 完整 ICL 平均 forward/读出约 0.371 秒；Spatial 写入
平均 3.266 秒、只读回答 0.064 秒。两类 worker 每题共同 90 帧解码平均约 12 秒。
这些是本次并发观测，不能把只读 0.064 秒当作 Spatial 端到端成本，也不能把每题
所有对照共用的一次解码重复当成九次独立系统实测。

协议指纹：`8b089206266dd191c2f329cadaa3581d7ce77db96596fb48b63e7595fdbf1da5`。
逐日、佩戴者、标签质量、选项位置、研究划分及配对结果见归档中的
[`aggregate_metrics.json`](../data/bench_v0_baseline_v1/aggregate_metrics.json)。
逐题完整日志和评分报告在 `runs/forecast_bench_v0/mcq_baseline_v1/eval/`。

## 数据与输入边界

每题有截止前 1800 秒 context、30 秒 recent，预测之后 5 秒中的一个具体动作。
输入清单 SHA-256：
`7a3b6eb61eef3111f409bd2781ec1634ce9f2bad3a66a2dd6919ef049bf29df8`。

- 推理只打开公共 `inputs_mcq.jsonl` 以及本题 `context.mp4`、`input.mp4`。
  答案、私有元数据、caption、audit 和 `future.mp4` 不进入模型。
- context 在 0、60、…、1740 秒取 30 帧；recent 在 0、0.5、…、29.5 秒取 60 帧。
  两者的全局时间范围为 `[0,1830)`，最终可见时刻为 1829.5 秒。
- 这是覆盖完整历史时间范围的 **90 帧稀疏采样**，并非逐帧观看 30 分钟。
  recent-only 使用同一批近期 60 帧；无视频组只获得问题、选项和共同时间说明。
- RGB 最长边 224，音频关闭，保留真实时间戳。解码选择不晚于请求时刻的帧，
  保存请求时间、实际 PTS、源媒体 size/mtime，以及解码像素和时间的 SHA-256。
- 每题独立重置 fast weights、KV 和位置状态，禁止跨题累计。不同题历史可能重叠，
  本题 future 可能出现在另一题 past 中，因此这一隔离是必要条件。

## 九组架构

| 结果键 | 权重来源 | 回答时的信息 |
|---|---|---|
| `qwen_blind` | 原始 Qwen3-VL-2B-Instruct | 无视频 |
| `qwen_recent_icl` | 原始 Qwen | 近期 60 帧，原生 video ICL |
| `qwen_full_icl` | 原始 Qwen | 历史 30 + 近期 60 帧，原生 video ICL |
| `blind` | 官方 Spatial-TTT checkpoint | 分支 bypass，位置 0，无视频 |
| `recent_icl` | 官方 Spatial checkpoint | 分支 bypass，近期 60 帧 ICL |
| `full_icl` | 官方 Spatial checkpoint | 分支 bypass，共同 90 帧 ICL |
| `spatial` | 官方 Spatial checkpoint | 90 帧写入本题 fast weights，随后仅文本读取 |
| `spatial_slow_only` | 同上 | 丢弃本题更新，读取官方学到的初始 fast weights |
| `spatial_base` | 同上 | 写入后旁路整个 Spatial 分支，保留相同文本位置 |

**官方 Spatial checkpoint 会替换完整模型权重，包括视觉塔和语言模型。** 因此
原始 Qwen 与官方 checkpoint 的差值不能全部归因于 TTT；同 checkpoint 下
`spatial − spatial_slow_only` 是累计视频更新的主要对照，`spatial_base` 进一步
检查整个 Spatial 分支的作用。

这里的 Spatial 使用本项目已有的 **parameter-only** 接口：分块之间清空视频 KV，
回答时没有历史帧或视频 KV。使用官方 checkpoint 不等于复现官方保留历史 KV 的
完整混合推理流程；以下分数评价的是本项目这条参数记忆路径。

Spatial 每次写入 8 帧；context/recent 分别分块，小尾块合并，保持偶数帧并避免
额外重复帧。默认 context 分为 8/8/8/6，recent 分为 7 个 8 帧块和 1 个 4 帧块，
共 12 次写入。问题和选项只在读出时提供，不参与写入。ICL 采用一次完整视频输入；
它与分块 Spatial 使用相同 RGB 和时间戳，但上下文注意力范围及计算成本不同。

## 读出与评分

所有方法使用相同中文任务说明及原始问题/选项。一次 forward 取得首个回答 token 的
logits，在六个独立 token `A…F` 中取最大值。该规则是 **受限选择题读出**，不是
自由生成准确率。每题还保存六个 logits、六选项内归一化概率、六字母占全词表的总
概率，以及不加限制时的最高 token，便于检查格式服从和选项偏差。并列时按 A–F 顺序
取第一个；不利用正确答案选择提示词或解码参数。

独立评分脚本在预测完成后读取 `answers.jsonl`、`metadata.jsonl`。每组分母固定
200，missing、invalid、error 都计错。标签是自动审核的 silver labels：193 条
supported、7 条 unclear，没有人工核验；随机六选一期望为 16.67%。报告包括：

- 全 200 题准确率、按录制日 macro、佩戴者、GT 一致性及干扰项构成分组；
- A–F 预测分布、按正确选项位置的准确率和受限读出诊断；
- 相同题目上的配对正确/错误变化与预测不一致率；
- 既有非官方按天研究划分：训练 111、验证 32、测试 57，全部固定。

本轮各组没有用这 200 题训练；后续若使用 111 题进行训练，应将 **同一批 57 题的
成绩**作为主要比较，不把包含训练题的 200 题总体分数称为测试性能。题目历史相互
重叠且跨佩戴者共享场景，不能把 200 个题目当成独立同分布的统计试验。

数据自带的 `code/evaluate_mcq.py` 要求 Slurm 环境，本机没有该运行条件。项目评分器
采用同样的单字母/选项文本匹配规则，并导出每组标准 `{"id":…, "prediction":…}`
文件，供之后在原环境交叉评分。

## 复现与续跑

先冻结协议，再启动 worker。协议包含公共输入 SHA、模型文件 SHA、Spatial checkpoint
SHA、执行代码 SHA、采样参数和读出方式；不同分片必须共享同一个协议。任何代码或
输入变化都需要新建协议，不能混入旧结果。

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python scripts/evaluate_forecast_mcq.py \
  --prepare --protocol runs/forecast_bench_v0/mcq_baseline_v1/protocol.json \
  --vision-linear-patch

conda run --no-capture-output -n meowbench python scripts/run_forecast_mcq_multigpu.py \
  --protocol runs/forecast_bench_v0/mcq_baseline_v1/protocol.json \
  --output-dir runs/forecast_bench_v0/mcq_baseline_v1/eval \
  --qwen-devices cuda:2 cuda:3 cuda:4 cuda:5 \
  --spatial-devices cuda:4 cuda:5 cuda:4 cuda:5 \
  --split-manifest runs/forecast_bench_v0/research_day_split.json
```

上述设备安排对应本轮可用显存，实际重跑前应查看 `nvidia-smi`。launcher 不终止其他
任务，重复设备表示同卡多个 worker。每个分片有独立日志和逐题预测；`--resume`
只跳过已完整、无错误并且协议一致的 episode。截断文件、半题、重复行和错误结果会
明确拒绝，不被当作完成。所有分片成功后才生成正式汇总。

完整运行结束后检查相同帧及记忆影响；该诊断只读预测，不读正确答案：

```bash
conda run --no-capture-output -n meowbench python scripts/analyze_forecast_mcq_predictions.py \
  --predictions runs/forecast_bench_v0/mcq_baseline_v1/eval/*_shard*.jsonl \
  --output runs/forecast_bench_v0/mcq_baseline_v1/prediction_diagnostics.json
```

诊断核对跨模型解码像素 SHA、90 帧/2205 视觉 token、每层更新记录，并比较
fast 与 slow-only/base 的六选项 logits。更新次数大于零本身不能证明权重有效；
logits 改变也只证明对读出产生影响，准确率和配对得失才衡量这种影响是否有用。

单进程示例：

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_forecast_mcq.py \
  --protocol runs/forecast_bench_v0/mcq_baseline_v1/protocol.json \
  --family spatial --device cuda:4 \
  --output runs/forecast_bench_v0/mcq_baseline_v1/spatial_all.jsonl
```

## 执行优化

本机 Qwen 视觉塔的 patch embedding 将每个已切好的 `2×16×16` patch 单独送入
Conv3d，每个 patch 仅产生一个空间/时间位置。该运算等价于将卷积核展平后作线性
映射。本轮显式启用实例级、可撤销的 `LinearVisionPatchEmbed`；权重、输入帧、
token 排列和模型结构语义不变，FP32/BF16 的浮点舍入差异单独验证。

实测 PyTorch 2.9.1+cu128 在本机对连续 BF16 输入选择 `SlowDilated3d`，
4 帧的整个视觉塔约需 7.02 秒，线性路径约 0.018 秒。真实帧验证中，FP32 视觉
输出相对 L2 误差约 `8e-6`；BF16 patch 输出相对误差约 `5.8e-5`，经过后续
视觉层放大到约 **5.4%**。因此这不是逐位一致的替换，也不能承诺所有题答案不变。
首题 `egolife_001` 的九组 MCQ 选择在两种路径下全部一致；这只是执行核验，
不是全数据数值等价证明。原生卷积的先前 smoke 还启用了等长 SDPA，而正式运行关闭
该选项；独立 SDPA 核验在所测输入上逐位一致。

所有正式比较组固定开启线性路径，避免不同组混用执行方式。后续实验应沿用同一
配置；若切回原生卷积，应新建基线。数值明细和首题核验保存于运行目录的
`vision_patch_embed_validation.json`、`smoke_execution_comparison.json`。

另有可选等长分组 SDPA 实现，但它不能解决此次卷积瓶颈，正式配置不启用。
原始路径保留，旧实验和已保存模型的默认执行方式不被修改。

## 暂不列分数的架构

- 旧 q9 LoRA checkpoint 记忆属于另一段视频。新题需要每例重新生成 past-only teacher
  并训练适配器，同时保留同 teacher 的显式文本 ICL 对照，才能公平评价。
- V-JEPA 2 旧实现是 C-PNG 的监督 probe，缺少新动作文本的选择题 readout；旧记录中的
  模型和源码路径当前也不存在。不能直接将旧 probe 准确率移到此表。
- Idea 01 当前只提供延迟 latent 预测，仍需独立训练动作候选 readout；该阶段沿用同一
  数据隔离、固定测试子集及上述基线。

这些项目不会被填成零分或伪装成已完成评测。

可长期复用的协议、固定划分、标准预测和聚合指标归档在
[`data/bench_v0_baseline_v1`](../data/bench_v0_baseline_v1/README.md)，详细日志留在
`runs/forecast_bench_v0/mcq_baseline_v1/`。后续改变采样、提示、执行精度或选择题
读出时应重新测对应基线，而不是直接与不同协议的数值作优劣判断。
