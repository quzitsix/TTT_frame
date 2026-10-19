# C-real：SVCBench 真实视频 matched-prefix 评测

> **主实验是 Natural9。**它包含 9 段非模拟真实图像视频、7 种来源、42 条
> question trajectories、35 个语义族和 125 个 query points。Natural9 已完成 625 行
> 五臂评测；本文数值来自通过协议审计的 strict summarizer 产物，不是手工估计。
>
> 3 段 Ego4D 已完成的实验保留为严格第一人称 pilot/subset，用来验证协议和给出先行信号，
> 不再代表 C-real 的主结论。

## 1. 为什么改用真实视频

原来的 C-PNG 用渲染后的文字事件卡检查写入、保持和读取链路。它适合做机制测试，但卡片
中的文字、颜色、固定版式和生成周期都是强线索，模型可以读卡或利用模板。Spatial-TTT 在
这种输入上的弱结果不能代表它处理连续真实画面的能力。

C-real 改用 SVCBench 的真实图像视频。模型在每个查询时刻只能观察此前的视频前缀，回答
事件或物体“到目前为止出现了多少次”。它检验连续视觉识别、状态累积以及 episode 内
test-time fast update。这个任务属于 **observed-prefix visual state/counting**，不是未来动作
预测，也不是完整的家庭长期记忆基准。

## 2. 主数据：Natural9

Natural9 从 [SVCBench 官方数据](https://huggingface.co/datasets/buaaplay/SVCBench)中选择
9 段非模拟、非 TOMATO 合成循环的视频。选择清单是
[`data/svcbench_real_natural9_v1.json`](../data/svcbench_real_natural9_v1.json)。它扩大了
视觉形式和计数目标，但必须准确理解其范围：这是 **non-simulated real-image、multi-source**
集合，不是纯第一人称数据，也不是纯家庭场景数据。

| 来源 | 视频数 | capture 类型 | 文件 | trajectories / query points |
|---|---:|---|---|---:|
| Ego4D | 3 | 可穿戴相机第一人称连续录像；室内穿门、骑行与道路活动 | 3 个 Ego4D UUID 视频 | 7 / 24 |
| ARKitScenes | 1 | 手持设备室内扫描序列 | `42898849.mp4` | 7 / 15 |
| ScanNet | 1 | 室内 RGB-D 扫描序列 | `scene0645_00.mp4` | 9 / 24 |
| ScanNet++ | 1 | 室内高分辨率扫描序列 | `5eb31827b7.mp4` | 9 / 24 |
| RoomTour3D | 1 | 房屋/房间巡游视频，不等同于可穿戴第一人称 | `B8Y35YnKONk.mp4` | 3 / 10 |
| CODa | 1 | 车载道路采集 | `6.mp4` | 2 / 7 |
| YouTube | 1 | 有剪辑的网络视频，本选择中为企鹅片段 | `youtube_0019.mp4` | 5 / 21 |
| **合计** | **9** | **7 种来源** |  | **42 / 125** |

视频时长为 137.834–1090.334 秒，总计约 54.7 分钟；但本选择最晚的 query time 是
325.57 秒，因此每个独立 episode 最多重放约 5.43 分钟的可见前缀。任务包含累计事件、
累计唯一物体、当前可见物体、近期新增物体和片段出现次数，35 个语义族只合并明确同义或
相同计数目标。

可复核 provenance：

- Hugging Face revision：`4c9bd87ef3b0f269ca8b4503c081f5f38bc1fc9a`
- 完整 `vcbench_eval.jsonl` SHA-256：
  `5537dc652e2d9329baeb3f303a08242261fe0cc6194ff5d43bea0f2fe48a9e82`
- Natural9 grouped subset SHA-256：
  `41f4121b38d2dbed9a8d3ff21f911cdb4e9c4a85dc492c4f6a7a0e45b47ac1d8`
- Natural9 选择清单 SHA-256：
  `43dbff2be74bd72a67484d0e89cdf0f90ded736f449b629b8f627dd137cea2e8`
- 9 个视频共 647,532,714 bytes；每个视频的独立 SHA-256 和问题映射写在选择清单中。

下载脚本固定 revision，并在下载后逐个校验视频大小和 SHA-256：

```bash
SVC_ROOT=/data/quzitsix/svcbench_real_natural9_v1 \
  bash scripts/download_svcbench_natural9.sh
```

正式 annotation 为
`/data/quzitsix/svcbench_real_natural9_v1/subset/svcbench_subset.json`，视频位于
`/data/quzitsix/svcbench_real_natural9_v1/videos/`。

## 3. matched-prefix 五臂协议

每个 query point 都是独立 episode。模型状态先 reset，再从视频开头写到该 query 的
cutoff；不同 query 不继承 fast state，也不会看到其他 query 的答案。这个设计形成干净的
配对比较，但不等同于一个永久在线状态连续服务整段视频，见第 8 节。

### 3.1 共同观测边界

- 默认 `chunk_seconds=4`、`frames_per_chunk=4`，即约 1 fps。
- 只保留时间戳 `<= query_time - cutoff_margin` 的帧；正式配置
  `cutoff_margin=0`。边界 chunk 解码后仍逐帧过滤，不能读取未来帧。
- 从完整可见前缀均匀选择，最多 512 张真实帧，最长边缩放到 224 px。Natural9 最晚查询
  只有 325.57 秒，所以正式配置中 512 帧上限不会截断任何前缀。
- ICL 与三个 Spatial arm 使用相同的 `frame_timestamps`、帧顺序和 frame-bundle SHA-256。
- 所有 arm 收到相同的 query time 与 cutoff time 文字提示，避免只让某一 arm 获得时间先验。
- Qwen 的 temporal padding、实际编码帧数、encode-call 数和 Spatial write 次数都逐行记录，
  strict summarizer 会审计这些字段。

### 3.2 五个 arm 与主比较

五个 arm 共享同一 Qwen 模型、processor、官方 Spatial-TTT checkpoint 和生成参数。

| arm | 输入/状态 | 解释 |
|---|---|---|
| `blind` | position 0 的纯文本回答，不提供视频帧 | 题目、语言先验和显式时间提示控制 |
| `icl` | 把 matched frame bundle 作为 native video context 编码一次 | 直接视觉上下文基线 |
| `spatial` | 按时间写入相同帧，从该 episode 的 fast state 读取 | 完整 Spatial-TTT 条件 |
| `spatial_slow_only` | 完成相同 ingest 后丢弃 episode fast state，从 learned initial/slow state 读取 | **test-time update 的主对照** |
| `spatial_base` | 完成相同 ingest 后绕过 Spatial 读分支，以 base attention 回答 | 架构诊断对照 |

主要比较是 **`spatial - spatial_slow_only`**。两者保持 checkpoint、写入流和读取分支，只在
读取时移除当前 episode 形成的 fast state。`spatial - spatial_base` 同时改变读取架构，只能
作诊断。

`icl - blind` 可以回答“提供直接视频上下文是否有帮助”，但它不是严格的纯像素因果消融：
ICL 经过 Qwen native multimodal video token/attention 路径，blind 只经过文本生成路径，二者
解码图和计算量并不完全相同。因此该差值包含视频信息和多模态解码路径的共同影响。

## 4. Natural9 运行与严格汇总

模型和 checkpoint：

- Qwen：`/data/quzitsix/models/Qwen3-VL-2B-Instruct`
- Qwen 模型工件 SHA-256：
  `979b5459898351c7b614e5fb91e97377247fd027db2bac7ea2558dd7503f0230`
- Spatial checkpoint：`/data/quzitsix/models/Spatial-TTT-nano/model.safetensors`
- checkpoint SHA-256：
  `5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947`
- matched runner SHA-256：
  `1aba8e69c5309e86ce1fd62fb1bbc08e09a2c6e4917e092daf3d032dc7aa6e33`

Natural9 固定 9 个 shard，每个 shard 独占一个 source video。先做不加载模型的 preflight；
把示例 GPU 列表换成当时确认空闲的物理卡：

```bash
conda run --no-capture-output -n meowbench \
  python scripts/run_svcbench_natural9_multigpu.py \
  --gpus 0,1,4,5,7 \
  --plan-only
```

确认后启动。launcher 把一张物理卡限制为 worker 内的 `cuda:0`，完成一个 shard 后自动领取
下一个；它默认拒绝有计算进程的 GPU，不会终止已有进程。已有 shard 只有在 JSONL、summary、
协议、哈希、行数、五臂覆盖和错误状态全部通过验证后才会跳过，否则 fail closed。

```bash
conda run --no-capture-output -n meowbench \
  python scripts/run_svcbench_natural9_multigpu.py \
  --gpus 0,1,4,5,7
```

launcher 的默认正式参数为 `bfloat16`、1 fps、224 px、512 帧、Spatial 每 8 帧一次 write、
`cutoff_margin=0`、`max_new_tokens=8`。输出目录是
`runs/svcbench_real_natural9_matched_v1/`。

九个 shard 全部完成后运行严格汇总：

```bash
conda run --no-capture-output -n meowbench \
  python scripts/summarize_svcbench_matched.py \
  runs/svcbench_real_natural9_matched_v1/shard{0..8}.jsonl \
  --annotations /data/quzitsix/svcbench_real_natural9_v1/subset/svcbench_subset.json \
  --selection data/svcbench_real_natural9_v1.json \
  --output runs/svcbench_real_natural9_matched_v1/summary.json \
  --report runs/svcbench_real_natural9_matched_v1/summary.md
```

strict summarizer 必须确认 125-point 网格的五臂完全一致、没有重复或缺失、题目/gold/subtype/
query time/视频哈希/运行协议相同、所有帧未越过 cutoff、frame bundle 与预算字段匹配，并从
raw answer 重新解析整数。parse/runtime failure 仍留在分母中并算错。预期总行数为
**125 × 5 = 625**。

## 5. 指标、相关性与 query 0039 的顺序

主指标是整数 exact-count accuracy，同时报告：

- query micro：125 个查询点直接合并；
- question macro：先在 42 条 trajectory 内平均；
- source-video macro：先在每段视频内做 question macro，再对 9 段视频平均；
- semantic-family macro：先在每个语义族内做 question macro，再对 35 个语义族平均。

GPA、MoC、UDA 按 SVCBench 官方 subtype 适用范围计算。trajectory 中任意答案解析失败时，
对应官方 trajectory metric 标为未评测；失败位置不会被删除或压缩。Natural9 中有 10 条
trajectory 只有一个 query point，无法提供随时间变化的充分证据，因此必须同时查看各指标的
applicable/evaluated coverage。

125 个 query points 不是 125 个独立样本。同一视频的时间点共享画面，语义近似问题也相关。
本实验报告 question/video/family macro 和配对差值，不做把逐行样本当 IID 的显著性检验。

SVCBench 的 question `0039` 还暴露了一个顺序问题：released `query_index` 顺序对应时间
`73.0, 227.0, 185.7` 秒，而真实时间顺序是 `73.0, 185.7, 227.0` 秒。主指标保留官方
`query_index, query_time` 顺序；strict summarizer 另外按 `query_time, query_index` 计算
chronological UDA sensitivity。重排不改变逐点 exact、GPA，也不影响该 O1-Snapshot 问题上
不适用的 MoC；它可能改变 UDA，所以最终报告必须同时给出两种 UDA，不能静默重排标签。

## 6. Natural9 主结果

`runs/svcbench_real_natural9_matched_v1/summary.json` 已显示
`protocol_validation: passed`：625/625 行完整，五个 arm 的 parse coverage 都为 1.0000，
没有 runtime error。

| arm | query exact | question macro | video macro | family macro | parse coverage | GPA | MoC | UDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blind | 0.2400 | 0.3060 | 0.2849 | 0.2995 | 1.0000 | 0.3060 | 0.9828 | 0.1927 |
| icl | **0.2640** | 0.2889 | 0.2794 | 0.2776 | 1.0000 | 0.2889 | 0.9425 | **0.4427** |
| spatial | 0.2400 | **0.3131** | **0.2887** | **0.3062** | 1.0000 | **0.3131** | 0.9253 | 0.1849 |
| spatial_slow_only | 0.2320 | 0.2980 | 0.2664 | 0.2900 | 1.0000 | 0.2980 | 0.9828 | 0.2083 |
| spatial_base | 0.2400 | 0.3060 | 0.2849 | 0.2995 | 1.0000 | 0.3060 | **1.0000** | 0.1927 |

| paired comparison | query delta | question macro delta | video macro delta | family macro delta | left-only / right-only |
|---|---:|---:|---:|---:|---:|
| icl − blind | +0.0240 | -0.0171 | -0.0055 | -0.0219 | 16 / 13 |
| spatial − spatial_slow_only | +0.0080 | +0.0151 | +0.0223 | +0.0162 | 6 / 5 |
| spatial − spatial_base | 0.0000 | +0.0071 | +0.0038 | +0.0067 | 6 / 6 |

主 TTT 比较呈现一个很小且不稳定的正差：Spatial fast 比 slow-only 净多答对 **1/125**，
question macro 高 1.51 个百分点。9 个 source videos 中，该差值在 Ego4D-e5d、RoomTour 和
ScanNet++ 为正，在 ScanNet 和 YouTube 为负，另外 4 个为零；42 条 trajectory 中只有 5 条
出现非零 exact delta。与此同时，Spatial 的 UDA 从 slow-only 的 0.2083 降到 0.1849，说明
fast state 没有更可靠地跟随真实计数方向。当前结果不支持“Spatial-TTT 已形成稳定真实视频
计数记忆”的结论。

ICL 在 query micro 上比 blind 高 2.4 个百分点，UDA 也明显更高，但 question、video 和
family macro 都略低。它在一些长计数轨迹上能输出更大的数字，却没有稳定命中精确总数；
再加上第 3.2 节所述解码路径差异，不能把这组差值单独归因为视频像素。

描述性 count calibration 更清楚地显示失败形态。Natural9 的 gold 范围为 0–47；下表的 MAE
只因本次所有 125 个答案都成功解析才有定义，若存在解析失败，strict summarizer 会把主 MAE
设为 null，而不会删除失败点后报告一个更好看的值。

| arm | MAE | prediction range | prediction histogram |
|---|---:|---:|---|
| blind | 3.296 | 1–2 | `1:39, 2:86` |
| icl | **2.784** | 1–10 | `1:19, 2:61, 3:17, 4:11, 5:3, 6:3, 7:6, 10:5` |
| spatial | 3.368 | 1–3 | `1:59, 2:63, 3:3` |
| spatial_slow_only | 3.296 | 1–3 | `1:39, 2:84, 3:2` |
| spatial_base | 3.288 | 1–3 | `1:39, 2:85, 3:1` |

Spatial fast 的 122/125 个答案都坍缩在 1 或 2，且从不输出 gold 中存在的 0 或大于 3 的
计数；它的 MAE 还略高于 slow-only。ICL 能把计数扩展到 10，因而 MAE 和更新方向更好，
但仍无法覆盖 24–47 的长事件计数。当前核心瓶颈既包含事件识别，也包含累计计数的数值标定。

125 个独立 episode 合计选择了每个视觉 arm 10,003 张真实帧。ICL 实际编码 10,066 张
（63 张 temporal padding、125 次 encode）；每个 Spatial arm ingest 10,003 张、实际编码
10,066 张，并完成 1,274/1,274 次 write。这里是跨独立 query 重放后的累计预算，不是 9 段
视频的唯一帧数。

question `0039` 是唯一 source order 与时间顺序不同的 trajectory；本次五个 arm 在该问题上
两种顺序的 UDA 都相同，因此 chronological sensitivity 对全局 UDA 的改变量均为 0。Natural9
中重复出现的 3 个 Ego4D 视频也与先前 pilot 的对应结果逐行一致：答案、预测、frame count、
timestamps 和 frame-bundle SHA-256 没有差异。

最终证据文件为：

- `runs/svcbench_real_natural9_matched_v1/shard{0..8}.jsonl`
- `runs/svcbench_real_natural9_matched_v1/shard{0..8}_summary.json`
- `runs/svcbench_real_natural9_matched_v1/summary.json`
- `runs/svcbench_real_natural9_matched_v1/summary.md`

解释时首先看 `spatial - spatial_slow_only`，再查看它是否跨 source video 和 semantic family
一致。本次结果的符号并不跨视频一致，所以不能只凭 query micro 的 +0.0080 宣称获得可泛化
优势。真实视频输入排除了“Spatial 只是不适合文字 PNG 卡”作为唯一解释；下一步应优先检查
计数目标的自监督写入信号、fast-state readout 标定和持续流更新，而不是继续扩大同类文字卡。

## 7. 已完成的严格第一人称 Ego4D-3 pilot

pilot 使用同一 matched-prefix 五臂协议和相同模型工件，仅包含 3 段 Ego4D 第一人称视频、
7 条 trajectories、5 个语义族和 24 个 query points。选择清单为
[`data/svcbench_real_ego4d_pilot_v1.json`](../data/svcbench_real_ego4d_pilot_v1.json)，严格结果
位于 `runs/svcbench_real_matched_v1/summary.json` 和 `summary.md`。验证已通过：120/120 行，
五臂 parse coverage 都为 1.0000，无 runtime error。

| arm | query exact | question macro | video macro | family macro | parse coverage | GPA | MoC | UDA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| blind | 0.2500 | 0.2738 | 0.2639 | 0.2917 | 1.0000 | 0.2738 | 0.9286 | 0.1190 |
| icl | 0.2083 | 0.2190 | 0.2000 | 0.2533 | 1.0000 | 0.2191 | 1.0000 | 0.4524 |
| spatial | 0.2500 | 0.2738 | 0.2454 | 0.2583 | 1.0000 | 0.2738 | 1.0000 | 0.0714 |
| spatial_slow_only | 0.2083 | 0.2262 | 0.2083 | 0.2250 | 1.0000 | 0.2262 | 0.9286 | 0.1905 |
| spatial_base | 0.2500 | 0.2738 | 0.2639 | 0.2917 | 1.0000 | 0.2738 | 1.0000 | 0.1190 |

| paired comparison | query delta | question macro delta | video macro delta | family macro delta | left-only / right-only |
|---|---:|---:|---:|---:|---:|
| icl − blind | -0.0417 | -0.0548 | -0.0639 | -0.0383 | 3 / 4 |
| spatial − spatial_slow_only | +0.0417 | +0.0476 | +0.0370 | +0.0333 | 2 / 1 |
| spatial − spatial_base | 0.0000 | 0.0000 | -0.0185 | -0.0333 | 2 / 2 |

该 pilot 中每个 arm 的 matched selection 共 3064 张真实帧。ICL 提供 3064 张、实际编码
3072 张（8 张 temporal padding，24 次 encode）；每个 Spatial arm ingest 3064 张、实际编码
3072 张（8 张 padding，387/387 次 write）。

结果只能作为先行信号：Spatial fast 相比 slow-only 的 query exact 高 1/24（+4.17pp），
question macro 高 4.76pp，但净改善集中在一个 Ego4D 骑行视频的 `vehicle_turn_any` 族；ICL
也没有超过带时间提示的 blind。3 个相关 video clusters 无法支持泛化结论，Natural9 才是主
实验。

## 8. 已知限制

1. Natural9 增加了真实视频和来源多样性，但仍只有 9 个 source-video clusters；其中只有
   3 段是严格第一人称 Ego4D，家庭相关主要来自室内扫描和 room tour，不能写成“第一人称
   家庭数据集”。
2. 每个 query 独立 reset 并重放前缀，适合因果配对，却没有测持续在线 fast state 依次服务
   多个查询的行为、遗忘或吞吐。persistent-stream protocol 应作为单独实验，不能与本表混合。
3. “matched”保证五臂的可见边界和唯一真实帧集合相同，不保证 FLOP、token 数或 encode-call
   数相同。ICL 一次编码整束 native video，Spatial 每 8 帧写入；`icl - blind` 的解码路径也不
   完全相同。
4. 1 fps、224 px 会漏掉短暂事件或小物体。需要做 2 fps/448 px 消融时，必须同时提高所有
   视觉 arm 的输入，不能只给某一方法更多画面。
5. 所有 arm 都接收显式 query/cutoff time，可能利用事件频率先验；blind 共享该提示用于量化
   这条捷径。
6. 五臂共享官方 Spatial checkpoint 的 learned slow/initial state。端到端 arm 可以比较，
   但 test-time fast update 的归因只看 `spatial - spatial_slow_only`。
7. 该任务回答已观察前缀中的计数，不直接测未来动作预测。旧的
   [C-PNG 结果](EXPERIMENT_C_RESULTS.md)保留为写入/读取、future boundary 和消融线路的机制
   诊断；[V-JEPA 2 probe](JEPA_BASELINE.md)是冻结表征加监督 readout，也只作表征诊断，不能
   与本页的 ICL/Spatial 数字直接排名。

## 9. 官方资料

- [SVCBench 官方 GitHub](https://github.com/buaa-colalab/SVCBench)
- [SVCBench 项目页](https://buaa-colalab.github.io/SVCBench/)
- [SVCBench Hugging Face 数据集](https://huggingface.co/datasets/buaaplay/SVCBench)
- [SVCBench 论文（arXiv:2603.12703）](https://arxiv.org/abs/2603.12703)
- [Ego4D 官方站点](https://ego4d-data.org/)
