# TTT_frame 实验总览与阶段性结论

整理日期：2026-10-09。本文汇总当前 `TTT_frame` 仓库和已保存运行产物中的实验结果，覆盖 LaCT、生成式 LoRA-TTT、Codex/teacher 对照、Spatial-TTT，以及 HomeSentinel/asuka。不同实验使用了不同数据、问题集、采样和评分标准，表格中的数字只在同一实验内部可直接比较。

## 一句话结论

项目已经完成了从“参数快速权重能否写入”到“真实长视频能否按顺序摄入并保存”的工程闭环，但还没有证明模型获得了可靠的第一人称家庭长期视觉记忆。

目前最可靠的结论是：

1. **机制路径已经打通。** 视频输入能够改变 LoRA 参数或 Spatial fast weights；删除视频帧和历史 KV 后，模型仍可以从参数分支生成答案。
2. **LoRA 真实视频基线有小规模收益，但不稳定。** SuperMemory 五道选择题中，盲答 1/5，LoRA 两个配置都是 3/5；EPIC 自由回答出现循环、幻觉和错误 teacher 目标。
3. **Spatial-TTT 能改变读出，但答案不稳定。** Q9 的开放式回答中 memory 比 paired control 更接近证据，但正式选择题三路都答错；19 道 Codex 派生选择题中 dense Spatial 为 14/19，较强的 Qwen3.5 纯文本模型为 16/19，但 Qwen3.5 没看视频。
4. **HomeSentinel 全量结果没有显示总体收益。** 52 段视频、138 道开放题、60 秒/4 帧配置下，Spatial memory 的平均 token F1 为 0.1554，without-memory 对照为 0.1624；event 子集略好，home/owner 子集略差。
5. **早期 LaCT 的“搬家效应”已经撤回。** 交叉实验显示 filler 来源、查询场景和中心化参考集会改变结论，原来的 SAME/MOVED 差异不能解释成真实家庭搬家记忆能力。
6. **Caption 文本参照明显强于当前 Spatial 视觉链路。** 同一个 Qwen3-VL-2B 只读 caption 的整体 token F1 为 0.2262，比 Spatial memory 高 0.0708；这更像是视觉覆盖、写入和读出问题，而不是单纯的模型规模问题。

因此，当前项目状态应表述为：**TTT 参数记忆的输入—更新—保存—读出链路可运行；长期事实的可靠写入、寻址、抗干扰和抗遗忘仍未验证。**

## 1. 项目路线和参数状态

仓库中有三条容易混淆的路线：

| 路线 | 输入和记忆 | 查询时使用 | 主要作用 |
|---|---|---|---|
| LaCT | 冻结 CLIP 向量写入外部 SwiGLU fast weights | 向量相似度/强制选择 | 早期机制和合成绑定实验 |
| VideoQA LoRA-TTT | 冻结 Qwen3-VL 生成 observation/QA，更新语言 decoder 的 q/v LoRA | 基座 + LoRA 生成答案 | 真实视频参数记忆基线 |
| Spatial-TTT | Qwen3-VL 视觉 token 直接更新三块 fast-weight 矩阵 | 官方 Spatial read 分支生成答案 | teacher-free 的视觉参数记忆路线 |

LoRA 路线本次 Qwen3-VL-2B 的关键参数是 28 层语言层的 q/v 投影，共 56 个目标层，rank=16、alpha=32，FP32 LoRA 参数约 12.25 MiB。写入时 teacher 生成临时 observation 和 QA，学生只对答案 token 做文本交叉熵更新；正式查询时不再输入历史视频、历史文字、训练样本或历史 KV。

Spatial-TTT 路线使用官方 `Spatial-TTT-nano` 完整 checkpoint。Qwen3-VL-2B 中 21 个 TTT 层保存三块 fast weights，更新后的状态约 252 MiB，含动量等运行态约 505 MiB 以上；视频帧、视觉 embedding 和历史 attention KV 在块之间不会作为长期记忆保留。Spatial 的官方 slow weights 是“如何写入和读取”的参数，视频只更新当前 episode 的 fast weights。

## 2. 早期 LaCT/CLIP 机制实验

这些实验不是第一人称家庭视频评测，而是用可控向量和合成场景检查快速权重本身。它们的价值是暴露了表征各向异性、Muon 更新和读出塌缩问题，不能直接当作当前 LoRA 或 Spatial 的视频准确率。

### 2.1 基础机制检查

- 定向写入 object→person 的关联后，纯向量读回实验在不同学习率和 head 配置下达到 **1.000**；置换 object→person 后，读出跟随置换而不是恒等映射；不写入时接近随机。这证明 `write(tokens, values=...)` 的参数读写路径有效。
- 修复了上游 LaCT minimal 实现的 Muon 赋值错误：上游代码把正交化后的梯度写进了权重。修复后的 `lr=0` 是 no-op，偏差约 0.0003；上游默认 Muon 配置在 `lr=0` 时也会改写状态，偏差约 20.416。
- CLIP 表征约 **94% 的能量集中在公共均值方向**，原始 object key 与中性房间 filler key 的余弦约 **0.711**。中心化和归一化把 key 混叠明显推远，保持窗口从原始的约 1 次写入提升到约 8 次写入。
- 随机单位向量 filler 能造成类似的权重旋转，却在 16 次写入后仍保持 **1.000**；真实房间 filler 两次写入就会崩到随机水平。这说明损伤主要来自 key 混叠，不是“权重旋转量”本身。
- 绑定信息并不一定被完全擦除：原始读出在 2、4、8、16 次 filler 后约为 0.271、0.250、0.250、0.250；对读出做均值修正后分别为 1.000、0.729、0.458、0.354；匈牙利匹配在 2、4、8、16 次为 1.000、1.000、0.750、0.604。这提示部分失败来自读出塌缩，而不只是写入信息消失。
- Muon 会丢弃大部分学习率幅度信息：`base_lr` 从 1e-4 到 100 时，更新步幅约恒定在 3.63；关闭 Muon 时步幅约从 0.03 到 15.5。也就是说，打开 Muon 后只调学习率不能按普通 SGD 直觉控制写入强度。
- head_dim 16→512、inter_multi 1→4 没有改变保持窗口；在这个合成设置中，没有找到能够替代表征修正的容量超参数。

### 2.2 “搬家”结论被撤回

实验原本在 A 家写入四个 object→person 绑定，再用不同家庭 filler 和查询背景模拟搬家。完整 2×2 结果如下：

| filler 来源 / query 场景 | query A | query B |
|---|---:|---:|
| filler A | 0.662 ± 0.033 | 0.362 ± 0.025 |
| filler B | 0.845 ± 0.027 | 0.812 ± 0.031 |

filler 主效应约 **+0.316**，query 主效应约 **−0.166**，二者方向相反；只看 SAME/MOVED 对角线会把两个因素混成一个数字。中心化参考集也会改变 gap 符号：narrow、broad、home-A-only 参考集得到的 gap@8 分别约为 +0.037、−0.019、−0.062。

结论是：这个 fixture 可以说明背景 filler 和查询背景的作用不同，但**不能回答真实家庭搬家是否损伤人物—物品绑定**。这一结论已在 `HANDOVER.md` 和 `ARCHITECTURE_AND_EVALUATION.md` 中明确标为撤回。

## 3. 真实视频 LoRA-TTT：SuperMemory 五题

数据是 `supermemory-pilot-v2` 的 18 个连续片段，覆盖 0–1080 秒；五道原始四选一题分别涉及网袋位置、浸泡要求、洋葱处理顺序、盐是否加碘、电池和盒子的顺序。每组参数记忆使用 70 个视频块、276 帧、280 条自动 QA 和 70 条摘要。历史前缀有大量重叠，不能宣传成 70 个独立视频样本。

### 3.1 有标准答案的结果

| 条件 | 正确数 | 准确率 | 摄入时间 | 查询 P50/P95 |
|---|---:|---:|---:|---:|
| 冻结模型盲答 | 1/5 | 20% | 0.17 秒，无视频 | 0.065/0.490 秒 |
| LoRA，12 步/块，lr=2e-4 | 3/5 | 60% | 27.52 分钟 | 0.053/0.062 秒 |
| LoRA，3 步/块，lr=2e-4 | 3/5 | 60% | 23.39 分钟 | 0.055/0.065 秒 |
| 显式文字 notes | 2/5 | 40% | 22.03 分钟 | 0.299/0.406 秒 |
| 冻结模型直接看帧 | 1/5 | 20% | 11.28 秒 | 179.019/234.474 秒 |

两档 LoRA 都比盲答多答对两题，但只有五题、一个 seed，且选项有明显先验，不能据此宣布长期记忆成立。两档答对的题集合不同：12 步版本答对 Q14，3 步版本答对 Q9；Q204 连盲答都答对。

两档 LoRA 的正式答案：

| 题目 | gold | blind | LoRA 12 步 | LoRA 3 步 | notes | oracle |
|---|---|---|---|---|---|---|
| Q9 网袋丢弃位置 | A | C | C | A | C | C |
| Q14 手机浸泡要求 | D | C | D | C | C | C |
| Q15 洋葱顺序 | C | B | C | C | C | C |
| Q19 盐是否加碘 | D | C | A | C | B/弃答 | A |
| Q204 电池与盒子顺序 | C | C | C | C | C | B |

### 3.2 EPIC 单视频诊断

EPIC `P07_106.MP4` 约 16.2 秒，8 秒一块，实际写入 3 块、9 帧、12 条 QA 和 3 条摘要。原设置 36 次更新、摄入 51.95 秒、峰值 CUDA allocated 约 5.36 GiB。

| 参数版本 | 总更新数 | 自由问题重复 | 模板问题重复 | 指令控制 |
|---|---:|---:|---:|---:|
| 12 步/块，lr=2e-4 | 36 | 2/8 | 2/3 | 2/2 |
| 3 步/块，lr=2e-4 | 9 | 2/8 | 0/3 | 2/2 |
| 12 步/块，lr=5e-5 | 36 | 1/8 | 1/3 | 2/2 |

原始参数回答反复生成“black coffee pot on a black coffee pot”；少步版本反复生成“white plastic bag”；低学习率版本虽然更流畅，却生成了采样帧不支持的彩球描述。冻结模型通常拒答，直接输入同样 9 帧能看到部分厨房信息，但仍有重复和细节错误。

教师目标存在实际噪声：例如问题问“用什么搅拌液体”，答案却是“深色液体”；黄色手套被观察文字写成正在手上佩戴，但帧中它在水槽内。训练 loss 下降不能证明事实被正确写入或能被未见问题读回。

## 4. teacher 与单片段对照

对 `supermemory-pilot-v2` Q9 60 秒片段（gold=A）做过 teacher 对照：

| 条件 | 输出 |
|---|---|
| frozen base | D |
| local LoRA v2 | C |
| Codex CLI LoRA v2 | D |
| Spatial smoke，未加载训练 checkpoint、scale=0.1 | D |

这次 Codex CLI teacher 没有超过本地 teacher，不能证明 GPT 订阅分析会提高质量。它也不是同一完整评测集上的独立比较，只有一个问题，且视觉帧无法直接确认“扔入白色垃圾桶”的动作已经完成。

## 5. Spatial-TTT：Q9 1080 秒视频

### 5.1 官方 checkpoint 与采样密度

官方 `Spatial-TTT-nano` 完整权重约 5.02 GB，包含训练后的语言模型和 Spatial slow weights。三种 60 秒分块配置如下：

| 配置 | 总帧数 | 视觉 token | 摄入时间 |
|---|---:|---:|---:|
| 60 秒 / 4 帧 | 72 | 7,056 | 114.1 秒 |
| 60 秒 / 8 帧 | 144 | 14,112 | 227.0 秒 |
| 60 秒 / 16 帧 | 288 | 28,224 | 454.9 秒 |

另有密集版 4 秒/8 帧：18 个片段、270 个 chunk、2,160 帧、约 211,680 个视觉 token，摄入约 **3,843 秒（64 分钟）**。它只改变输入采样密度，没有反向训练官方 slow weights。

### 5.2 Q9 开放式和正式选择题

开放式问题 `Where did I throw the empty red mesh bag earlier?` 的运行记录：

- 16 帧 memory：回答在 trash can，方向上接近金标白色垃圾桶；
- 同一 checkpoint 的 without-memory：回答物品位于房间中央/活动区域，没有稳定定位；
- 密集 4 秒/8 帧 memory：回答 trash can；
- 密集版本 without-memory：回答 on floor。

这说明 fast-weight read 分支会改变读出，但开放式回答不完整，不能替代正式评分。使用发布包正式 prompt（完整四选项、只输出字母）重新运行时，4 秒/8 帧 memory、60 秒/16 帧 memory 和官方 checkpoint without-memory **全部输出 C，gold 为 A，均为 0/1**。因此此前“trash can”只能作为语义方向性现象，不能写成 Q9 正确率。

### 5.3 当前模型是否知道厨房物体

对 `What objects did I interact with in the kitchen?`，16 帧 memory 和 without-memory 都生成 microwave、coffee maker、toaster、kettle、cutting board 等常见厨房物品列表。该问题容易被模型的厨房先验回答，不能证明模型检索了 1080 秒视频。

## 6. Codex 派生 19 道 MCQ

`data/q9_codex_mcq.jsonl` 从保存的 Codex 场景分析构造 19 道选择题，覆盖 location、relation、placement、order、long-range。选项字母平衡为 A=5、B=5、C=5、D=4。题目与 Codex 分析存在来源关系，因此这是定向 sanity check，不是独立 benchmark。

### 6.1 三种记忆方法

| 方法 | 总体 | location | relation | placement | order | long-range |
|---|---:|---:|---:|---:|---:|---:|
| LoRA + local teacher | 11/19 (57.9%) | 1/4 | 3/5 | 2/2 | 3/4 | 2/4 |
| LoRA + Codex teacher | 10/19 (52.6%) | 0/4 | 2/5 | 2/2 | 3/4 | 3/4 |
| Spatial-TTT，4 秒/8 帧 | 14/19 (73.7%) | 3/4 | 5/5 | 2/2 | 2/4 | 2/4 |

Spatial 在这组题上最好，但题目来自 Codex 分析，不能独立证明 Spatial 已学会长期视觉记忆。

### 6.2 Spatial paired controls

| 方法 | 总体 | location | relation | placement | order | long-range |
|---|---:|---:|---:|---:|---:|---:|
| Spatial-TTT，4 秒/8 帧 | 14/19 (73.7%) | 3/4 | 5/5 | 2/2 | 2/4 | 2/4 |
| Spatial-TTT，60 秒/16 帧 | 12/19 (63.2%) | 3/4 | 4/5 | 2/2 | 2/4 | 1/4 |
| Spatial checkpoint without-memory | 12/19 (63.2%) | 2/4 | 5/5 | 2/2 | 2/4 | 1/4 |

四道 long-range 题的结果是 local LoRA 2/4、Codex LoRA 3/4、dense Spatial 2/4、sparse Spatial 1/4、no-memory 1/4。dense memory 相对 paired control 修正了部分关系和顺序题，但也把一个原本正确的顺序题改错，说明 fast weights 改变读出不等于稳定正确。

### 6.3 Qwen3.5 纯文本基线

服务器上的 `/data/hf_models/Qwen/Qwen3.5-35B-A3B` 是 35B 总参数、约 3B active 的 hybrid MoE，不能直接加载现有 Qwen3-VL-2B 的 Spatial/LoRA memory。它只接收问题和选项，没有视频、Codex evidence、gold 或 TTT memory，却得到 **16/19 (84.2%)**；将选项位置打乱后仍为 **17/19**。

这说明 19 道题包含较强的语言先验：不自然干扰项、常识布局和较长的正确选项足以给纯文本模型较高分。Qwen3.5 结果不能被写成视频记忆结果，也不能直接和 Spatial 的视觉记忆能力比较。

## 7. HomeSentinel/asuka 全量 Spatial-TTT

这是目前最重要、也最接近真实家庭连续观察的评测。数据位于 `/data/HomeSentinel/asuka`，包含 52 段完整 `indoor_video.mp4`，总室内时长约 33,300 秒，题库 138 道开放题：owner 52、home 42、event 44。

评测按 `/data/HomeSentinel/video_order.json` 的时间顺序处理。有限 cutoff 的 event/home 题在对应视频后回答，`cutoff=-1` 的 owner/home 长期题等 52 段全部处理完再回答；因此没有把未来视频泄漏给 episodic 问题。每题有两个臂：同一 Spatial fast-weight 状态的 `memory`，以及相同基座/位置但绕过 fast weights 的 `without-memory`。

配置为 Qwen3-VL-2B、官方 Spatial-TTT nano checkpoint、BF16、60 秒窗口、每窗口最多 4 帧、max side 448、最多生成 128 token。run6 于 2026-09-29 完成，耗时约 **5,339 秒（89 分钟）**，两臂各 138 行，276 行全部 `ok`。

### 7.1 结果

| category | Spatial memory mean token F1 | without-memory mean token F1 |
|---|---:|---:|
| home | 0.1951 | 0.2172 |
| event | 0.1224 | 0.1186 |
| owner | 0.1514 | 0.1552 |
| **all 138** | **0.1554** | **0.1624** |

按 cutoff：episodic 题为 0.1364 vs 0.1402，full-history 题为 0.1724 vs 0.1822。两臂答案有 96/138 道不同；token F1 上 memory 胜 33 道、without-memory 胜 41 道、持平 64 道。exact normalized 两臂都是 0/138，gold substring 都是 2/138。

结论：在当前通用 checkpoint 和 60 秒/4 帧稀疏采样下，Spatial memory 没有带来总体词法提升；event 略有提升，但 home/owner 下降。开放题 token F1 不是人工语义评分，同义词和解释长度都会影响它，结果应作为基线诊断，而不是最终能力分数。

### 7.2 工程问题和修复

全量过程中发现某些视频的最后短窗口只有 1 帧，而 Qwen3-VL 的 temporal processor 要求时间维大于 2。现在只在 processor 输入内部复制末帧到至少 3 帧，原始帧数统计保持不变，并加入回归测试。

由于服务器上其他 Qwen3.5 批处理会回收默认 tmux/systemd 作业，评测脚本后来加入了每段视频后的 checkpoint 和 resume：最新 fast weights、Spatial slow weights 和 `progress.json` 保存在 `/data/TTT_frame_homesentinel_asuka_spatial_latest`。最终 checkpoint 的进度为第 52 段，预测与 summary 位于：

- [`predictions.jsonl`](../runs/homesentinel_asuka_spatial_60s4f_run6/predictions.jsonl)
- [`summary.json`](../runs/homesentinel_asuka_spatial_60s4f_run6/summary.json)
- [`memory.json`](/data/TTT_frame_homesentinel_asuka_spatial_latest/memory.json)

### 7.3 Caption-only 参照

HomeSentinel 还有一条不同模态的参照：不读视频，只把截止到题目允许时间的
`merged_captions.json` 有序文本交给模型回答。它不是 TTT memory；caption 已经把视频
中的场景、地点、活动和部分事件转成了文字，因此这项结果主要衡量“已有文本观察被
语言模型检索”的文本参照。数据包括 52 个视频、1,414 个场景、10,724 个事件和
3,303 条 speech；脚本没有使用 `evidence.segment_ids` 或 gold 检索
caption，但 caption 本身是自动生成文本，仍可能包含错误或比原始稀疏视频更完整的
信息。summary 和 important 的最长输入约为 75.8k 和 137.4k tokens，因此它和 Spatial
的参数读出并不是同一种信息接口。

| 条件 | 模型 | 输入 | overall mean token F1 | gold substring | exact |
|---|---|---|---:|---:|---:|
| caption summary | Qwen3-VL-2B | summary/activity/location/time 文本 | **0.2262** | 11/138 | 0/138 |
| caption summary | Qwen3.5-35B-A3B | 同类 summary 文本 | **0.2427** | 19/138 | 0/138 |
| caption important | Qwen3.5-35B-A3B | summary 加每个场景最后一个结构化事件 | **0.2481** | 16/138 | 0/138 |
| Spatial memory | Qwen3-VL-2B | 原始视频，60 秒/4 帧，fast weights | **0.1554** | 2/138 | 0/138 |
| Spatial without-memory | Qwen3-VL-2B | 同一视频/基座，绕过 fast weights | **0.1624** | 2/138 | 0/138 |

按类别展开如下，便于区分长期 owner/home 题和有 cutoff 的 event 题：

| 条件 | owner | home | event |
|---|---:|---:|---:|
| Qwen3-VL-2B caption summary | 0.1632 | 0.3149 | 0.2161 |
| Qwen3.5 caption summary | 0.2039 | 0.2825 | 0.2505 |
| Qwen3.5 caption important | 0.2096 | 0.2778 | 0.2653 |
| Spatial memory | 0.1514 | 0.1951 | 0.1224 |
| Spatial without-memory | 0.1552 | 0.2172 | 0.1186 |

同模型 Qwen3-VL-2B 的 caption summary 已经高于当前 Spatial memory，说明瓶颈主要在
视觉输入覆盖、Spatial 写入和参数读出，而不是只换一个更大的语言模型。Qwen3.5 的
结果还混合了模型规模和架构差异，不能直接当作 TTT 能力比较。`important` 只增加了
每个场景的一个末事件，F1 略升但 substring 从 19 降到 16，说明开放题的词法指标会
受到文本组织和答案表达影响，不能把它解读成单调的事实覆盖率提升。

原始记录和输出位置：

- [`HOMESENTINEL_CAPTION_RESULTS.md`](/home/quzitsix/meowbench/docs/HOMESENTINEL_CAPTION_RESULTS.md)
- [`Qwen3-VL-2B caption summary`](</home/quzitsix/meowbench/runs/homesentinel-caption-qwen3-vl-2b-summary/summary.json>)
- [`Qwen3.5 summary`](</home/quzitsix/meowbench/runs/homesentinel-caption-qwen35-summary-batched/summary.json>)
- [`Qwen3.5 important`](</home/quzitsix/meowbench/runs/homesentinel-caption-qwen35-important/summary.json>)

## 8. 统一看这些结果

### 8.1 哪些结果可以支持“参数被改变”

- LoRA：启用/禁用 adapter 会改变回答；训练 loss 会下降；保存和新进程加载能工作。
- Spatial：同一官方 checkpoint 的 memory 与 without-memory 输出不同；Q9 开放式回答和 19 道 MCQ 中，fast-weight read 会改变部分题的结果。
- HomeSentinel：96/138 题两臂答案不同，说明视频写入状态参与了读出，但总体分数没有提升。

### 8.2 哪些结果不能支持“可靠长期记忆”

- 五道 SuperMemory 题只有 3/5，且盲答和选项先验影响明显。
- Codex 19 题从 Codex 分析构造，存在 teacher/题目来源泄漏风险。
- Qwen3.5 的 16/19 没有视频输入，主要测语言和选项先验。
- HomeSentinel memory 的总体 F1 低于 without-memory，且是自动 gold 的词法诊断，不是人工语义评分。
- EPIC 中 teacher 目标本身有语义配对错误，低 loss 和流畅回答不代表写入了真实事实。
- 当前没有完成“人工事实写入上界、错误历史控制、同义改写一致性、抗干扰曲线、跨场景人物—物品绑定”的完整实验。

### 8.3 当前最可能的瓶颈

1. **感知和 teacher 目标质量。** 模型没有看到的细节无法写入；错误 observation/QA 会被直接蒸馏。
2. **Spatial slow checkpoint 与任务分布不匹配。** 官方 checkpoint 不是 HomeSentinel 家庭记忆训练得到的；项目的直接视觉写入 prompt 也比官方训练监督简单。
3. **采样密度。** 60 秒/4 帧会遗漏短动作、细小物品和屏幕文字；HomeSentinel 的总体结果不能代表 4 秒/8 帧或训练后 Spatial。
4. **参数寻址和读出。** fast weights 改变输出，但可能把多个事实混成常识性答案，或在长序列中覆盖旧状态。
5. **评测题质量。** 选择题容易被选项风格和语言先验解决；开放题的 token F1 又会惩罚合理同义表达。

## 9. 当前代码和复现实验入口

主要文档和产物：

| 内容 | 文件 |
|---|---|
| LoRA 架构、参数、真实视频细节 | [`ARCHITECTURE_AND_EVALUATION.md`](ARCHITECTURE_AND_EVALUATION.md) |
| LoRA 五题和 EPIC 报告 | [`REPORT_20260911_REAL_VIDEO.md`](REPORT_20260911_REAL_VIDEO.md) |
| teacher/Codex 接口和单片段结果 | [`TEACHER_PILOT_RESULTS.md`](TEACHER_PILOT_RESULTS.md)、[`TEACHER_BRIDGE.md`](TEACHER_BRIDGE.md) |
| Spatial 原理和模型修改 | [`SPATIAL_TTT.md`](SPATIAL_TTT.md) |
| Q9 官方 checkpoint 运行 | [`SPATIAL_OFFICIAL_Q9_RUN.md`](SPATIAL_OFFICIAL_Q9_RUN.md) |
| Q9 密集 4 秒/8 帧 | [`SPATIAL_DENSE_4S8F_RUN.md`](SPATIAL_DENSE_4S8F_RUN.md) |
| Codex 19 题和 Qwen3.5 对照 | [`CODEX_MCQ_EVAL.md`](CODEX_MCQ_EVAL.md) |
| HomeSentinel 全量评测 | [`HOMESENTINEL_SPATIAL_EVAL.md`](HOMESENTINEL_SPATIAL_EVAL.md) |
| 评测源码 | [`scripts/evaluate_homesentinel_spatial.py`](../scripts/evaluate_homesentinel_spatial.py) |

HomeSentinel 全量结果可重新查看：

```bash
cd /home/quzitsix/TTT_frame
python - <<'PY'
import json
from pathlib import Path
summary = json.loads(Path("runs/homesentinel_asuka_spatial_60s4f_run6/summary.json").read_text())
print(json.dumps(summary["summary"], ensure_ascii=False, indent=2))
PY
```

如果中断后要继续 HomeSentinel，应使用脚本的 `--checkpoint-dir` 和 `--resume-from`，不要把半截 JSONL 当成完整结果。当前仓库修复和文档提交已经推送到远程，最近相关提交为：

- `a1d66a8`：加入 HomeSentinel Spatial-TTT 评测脚本和说明；
- `b2682b9`：处理 Qwen3-VL 短尾窗口并加入测试；
- `a60e79d`：加入 checkpoint/resume；
- `2a262ed`：记录 HomeSentinel 全量结果。
- `5f2730e`：加入本总览及 caption 对照、8 卡密集训练方案。

## 10. 下一步建议

下一阶段不应先扩大视频长度或增加优化步数，而应建立一个小而严格的可读回闭环：

1. 人工核对 10–20 个视频事实，记录准确时间区间、物品、位置、动作和顺序。
2. 同时保存帧级事实、teacher observation、写入后参数状态和查询输出，区分“没看到”“没写入”“写入后被覆盖”“寻址失败”“生成失稳”。
3. 对每个事实使用原问题、同义改写、无选项开放题，并加入错误历史和打乱视频控制。
4. 先测原始基座、官方 Spatial without-memory、Spatial memory、LoRA memory 和显式 notes，再比较同一采样帧和同一文本目标。
5. 只有单事实可稳定读回后，才增加无关 filler、相似物体、位置更新和跨视频人物—物品关系，画出随干扰量变化的保持曲线。
6. 如果要让 Spatial 适应家庭记忆，应训练 slow weights/读出模块或设计明确的 teacher-forced offline trainer；仅把 `ttt_scale_init` 调大不能替代训练。

### 10.1 使用 8 张卡做密集采样训练

这件事技术上可行，但需要区分“密集采样评测”和“训练 Spatial slow parameters”。当前
[`evaluate_homesentinel_spatial.py`](../scripts/evaluate_homesentinel_spatial.py) 已经可以
用 `--chunk-seconds 4 --frames-per-chunk 8` 做密集的在线 fast-weight 写入；仓库中的
[`SpatialOfflineTrainer`](../ttt_frame/spatial_trainer.py) 只提供可微的单 episode API，
还没有把 HomeSentinel caption 转成 QA batch、按场景切分、DDP、checkpoint/resume
封装成完整训练 CLI。

HomeSentinel 的 52 段视频总时长约 **33,297 秒**。按当前配置估算：

| 配置 | 约 chunks | 约帧数 | 相对当前 60 秒/4 帧 |
|---|---:|---:|---:|
| 60 秒/4 帧 | 580 | 2,320 | 1× |
| 4 秒/8 帧 | 8,341 | 66,728 | 约 28.8× |

Q9 已测得 4 秒/8 帧约 14.2 秒/chunk；按线性比例，HomeSentinel 全量密集摄入约需
**33 小时单卡**，实际应先测 1–2 段视频。8 卡可以把独立 episode 或视频 shard 分到
不同进程，理想情况下约 4–6 小时，但不能把 8 卡分别得到的 fast weights 直接拼成一个
连续的 52 视频 memory：Spatial 更新是非线性的，当前代码也没有合法的 fast-state
merge 操作。

推荐的训练形态是：

1. 按视频划分 train/validation/test，例如 40/6/6，避免同一视频的 caption 和问题
   同时出现在训练与测试中。
2. 每个 rank 加载一份 Qwen3-VL-2B 和 Spatial slow 参数副本，用 `torchrun`/DDP
   处理不同的短 episode；每个 episode 先按 4 秒/8 帧写视觉，再在无视频的 read 模式
   用 `activity_summary` 和结构化 event 做 teacher-forced 文本交叉熵。
3. 每个 episode 只保留 8–32 个 dense chunk（约 32–128 秒），更新后 reset fast state，
   对 slow 参数做梯度同步。不能对 8,341 个 chunk 的完整历史保留反向图，否则显存和
   反向时间都会失控；这需要 TBPTT/短 episode。
4. 首轮只训练 Spatial 新增参数，先尝试 readout-only，再开放写入矩阵；语言模型保持
   冻结。每卡的 slow 参数、Adam 状态和 fast runtime state 都远小于 24 GB，真正的
   显存瓶颈是视觉激活和反向图。
5. 训练完成后，把保存的 slow checkpoint 重新加载，在单个进程中按 `video_order.json`
   的真实顺序重放 52 段视频，再用相同的 HomeSentinel 138 题和 without-memory 对照
   评测。训练时不能使用 benchmark gold、`evidence.segment_ids` 或查询答案。

先跑一个 1–2 段视频/约 10 分钟的 throughput pilot，比较 4 秒/8 帧和 8 秒/8 帧，
确认每个 episode 的显存、loss 是否下降以及 caption 监督是否包含事实，再启动 8 卡
全量训练。当前能直接执行的密集**评测**命令是：

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python -u \
  scripts/evaluate_homesentinel_spatial.py \
  --data-root /data/HomeSentinel/asuka \
  --order-path /data/HomeSentinel/video_order.json \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 \
  --chunk-seconds 4 --frames-per-chunk 8 --max-side 448 \
  --max-new-tokens 128 \
  --checkpoint-dir /data/TTT_frame_homesentinel_asuka_spatial_dense_latest \
  --output runs/homesentinel_asuka_spatial_4s8f/predictions.jsonl
```

这条命令会密集写入和评测，但**不会训练 slow parameters**。真正的 8 卡训练需要补充
caption→QA 数据转换、短 episode 反向、DDP 梯度同步和 slow/optimizer checkpoint；
在这些组件加入前，不应把 `4s/8f` 的在线摄入称为重新训练。

当前最重要的验收标准是：**memory 相比 paired without-memory，在独立人工事实集上稳定提高准确率，并且同义改写、错误历史和追加无关视频不会使早期事实随意改变。**
