# 研究任务：RL 何时能帮助第一人称视频的 TTT 记忆与 forecasting？

> 更新：2026-10-25；项目：`/home/quzitsix/TTT_frame`；Python：conda `meowbench`。
> 本文是下一轮 IdeaX 的研究输入，不是 RL 已有效的结论，也不授权 IdeaX 覆盖旧提案。
> 旧输入完整备份：[ideax_input.pre_rl_20261025.md](docs/archive/ideax_input.pre_rl_20261025.md)。
> 原文 SHA256：`6ff75a6c34f27a7713bcf6152c3c306aa48e56725cad79c78c9a0541b4aa3859`。
> 证据标记：**实际结果 / 已实现 / 机制假设 / 计划 / 待核验**。不能把计划写成结果。

## 1. 研究问题与当前执行优先级

用户研究第一人称视频中的参数化记忆和 forecasting，希望知道 TTT 如何改善预测、
在什么条件下能超过 ICL。下一轮提案重点是：

**在相同可见视频、合法反馈和计算预算下，RL 能否学会控制记忆的写入、保留、遗忘、
更新强度、检索地址和视觉细节获取，从而改善后续预测？其收益是否超过直接监督、
可微优化、固定策略、显式检索和 ICL？**

RL 只是待检验的学习工具。现有视觉 MSE 下降但动作准确率没有提高，既不能证明
forecasting 饱和，也不能证明 RL 必然有效。必须说明 RL 在解决哪个决策和信用分配
问题，不能把负 MSE 改名为 reward 就宣称新的 RL 机制。

旧 **Idea 3：查询地址搜索** 的原生 Spatial 诊断已完成全部 200 题，结果为负。
先阅读实际结果文档 `docs/IDEA03_BENCH_V0_RESULTS.md`，不得假定 Idea 3 已带来提升，
也不得把代理目标上升当作 RL 的语义奖励依据。旧提案位于
`ideax_output/`，保持原文。RL 提案写入新的 `ideax_rl_output_20261025/`。

## 2. 当前能据以推理的项目证据

### 2.1 数据与评价边界

真实视频数据 `/data/share/raw/bench_v0` 包含 200 道六选一未来动作题，六位佩戴者、
七个录制日；每题约 1800 秒历史和 30 秒近期视频，目标是 cutoff 后约 5 秒动作。
窗口有重叠，禁止随机重分题目。固定划分文件为
`runs/forecast_bench_v0/research_day_split.json`：

| 用途 | 录制日（全部佩戴者共同分组） | 题数 | 可用监督 |
|---|---|---:|---|
| train | 第 1–4 天 | 111 | 规定离线训练可用动作标签；过去视频可构造延迟视觉配对 |
| validation | 第 5 天 | 32 | 选超参数与停止规则，不当额外训练标签 |
| test | 第 6–7 天 | 57 | 最终答案仅用于已冻结预测的评分 |

每题独立初始化在线状态；禁止跨题积累参数、奖励或检索内容。题目公开的问题、选项和
cutoff 前视频可以用于预测。测试答案、`future.mp4`、未来 caption 和私有答案元数据
不能用于当前题更新、查询搜索或模型选择。候选动作到视觉目标的映射只能在 train
拟合；候选编码排除 A–F 字母和位置线索。

这 57 题已经被反复探索，后续比较必须标为**探索性比较**。标签是自动筛选的
**银标签**，没有独立人工核验测试结论。两天测试及同日重叠不支持将 57 题视为 IID
来宣称显著性。新的确认性结论需要未查看的新录制日/参与者和独立人工审计。

当前没有密集历史动作真值。后续帧到达可以提供视觉特征、重构/预测误差等自监督反馈；
不能默认它会提供正确 verb/noun、事件边界或语义成功奖励。teacher 伪标签仍是伪标签。

### 2.2 已完成的 Idea 1 与 scaling

**实际结果**见 [完整 Idea 1 评测](docs/IDEA01_BENCH_V0_MCQ_RESULTS.md)、
[容量与训练量比较](docs/IDEA01_SCALING_RESULTS.md)，逐题归档在
`data/bench_v0_idea01_v1/`、`data/bench_v0_idea01_scaling_v1/`。

Idea 1 是冻结 Qwen3-VL-2B 视觉塔后的独立预测 fast MLP，**不是完整官方 Spatial-TTT**。
缓存每题约 `[3660,8,256]`，2 FPS、完整 1830 秒前缀。近期预测器使用 train 拟合后冻结；
四个 horizon 的 fast MLP 在历史结果实际到达后拟合视觉残差，动作读出在 train111
监督拟合、validation32 选择正则。候选独立编码，读出不接收方法名。

| 已有比较 | 探索性 test57 结果 | 可解释范围 |
|---|---:|---|
| recent predictor + 共享语义读出 | 24/57，42.11% | 无在线参数更新对照 |
| residual TTT：H128/512/2048，random/pre200/pre800，u1 | 均 24/57 | 对应冻结初始化也为 24/57；未建立在线 TTT 动作收益 |
| random u4 residual，三个容量 | 均 23/57 | 更多在线更新未稳定改善动作预测 |
| 原始 Qwen full ICL | 21/57，36.84% | 旧 90 帧 VLM 基线，输入与监督不匹配 |
| 原始 Qwen recent ICL | 23/57，40.35% | 旧近期视觉基线 |

Scaling 共 12 配置 × 200 题，fast 参数量为 262,144 / 1,048,576 / 4,194,304。
200/800 step 初始化只用 train 内合法视觉配对进行普通回归训练，**不是端到端元训练**。
既有数值必须从固定逐题产物核查；ICL 核验入口为
`data/bench_v0_baseline_v1/` 和 `docs/BENCH_V0_MCQ_BASELINES.md`。

已观察前缀内成熟预测的 MSE：recent 为 0.513887；H128 random u1 residual 为
0.496274；H2048 pre800 residual 为 0.487428，其对应冻结初始化为 0.503286。
这些预测先生成，随后等待目标真正到达再评分。它们不是 cutoff 后最终动作准确率，
不能将更低 MSE 当作动作成功，也不能将训练配对数当成独立动作标签数。

**机制假设**：问题可能在写入、保留、语义对齐、读取或小样本读出，而非只在 fast
参数量；目前尚未唯一定位。RL 方案必须区分这些解释，不能从相关现象直接推导因果。

### 2.3 与官方 Spatial 路线及其他旧 idea 的关系

- `ttt_frame/spatial_memory.py`、`spatial_model.py`、`spatial_videoqa.py` 实现本地
  Spatial 路径；21 层、每层 4 heads，fast weights 共 66,060,288 个 FP32 值，
  约 252 MiB；momentum 另占约 252 MiB。它的写入为点积关联 ascent，不是 Idea1 MSE。
- `ttt_frame/spatial_probe.py` 已有 `association_delta` 与
  `projected_inverse_address_search`。分数为
  `<f_W(q),v> - <f_W0(q),v>`，仅更新临时查询 q，W/W0 冻结。每个 head/token 受
  L2 球约束，保留包括初始查询的最佳迭代；reset 分差应精确为零。
- Idea 3 研究“信息已写入，但普通读取难以获得”。若采用 Idea1 适配版，必须另命名、
  明确目标变化，不能冒充官方 Spatial 的完整实现。官方路径与小原型的结果分开报告。
- Idea 2 的 `key_transport.py` 已有 Procrustes 与拒绝条件；Idea 4 函数响应记忆
  缺完整教师状态轨迹；Idea 5 细节获取缺 low/high 配对验证。它们是可复用入口，
  不是已经验证有效的 RL 基础。

## 3. 必须先定义何谓 RL

### 最新实际证据：Idea 3 的同帧比较

本轮使用官方 Spatial-TTT-nano 的原生关联 fast weights，跨 chunk 只传参数状态，
冻结记忆后优化临时查询；不是 Idea1 MSE 适配，也不是完整 outer-trained Idea3 或
官方完整混合流式协议。text-V 候选目标未经视觉语义对齐训练。

| 方法 | Validation32 | 探索性 test57 |
|---|---:|---:|
| 直接读取，beta=1 | 8/32 | 9/57，15.79% |
| 三步查询搜索，beta=1、r=0.5 | 6/32 | 6/57，10.53% |
| reset / 冻结初始化 | 4/32 | 3/57，5.26% |
| 错误视频直接读取 / 搜索 | 7/32 / 2/32 | 10/57 / 9/57 |
| 原始 Qwen full ICL | 6/32 | 21/57，36.84% |

超参数仅用 validation，所有预测冻结后才使用 test 答案评分。200 题均与 full ICL
使用相同 90 帧，实际像素与时间戳核验通过；checkpoint、读出及计算预算仍不同。
搜索相对直接读改对 3 题、改错 6 题；相对 full ICL 改对 3 题、改错 18 题。
代理关联目标提高，但没有动作收益或当前视频特异性收益证据。
reset 使用新的候选似然读出，不能和旧字母盲答直接解释为记忆收益。

这支持优先检验奖励/代理目标与任务的对齐，不能推出完整 Idea3 无效、记忆没有信息
或 RL 必然更好。归档 `data/bench_v0_idea03_spatial_v1/` 含逐题分数、配对与审计。

### RL 的决策与反馈边界

在被动视频中，模型选择预测类别通常不会影响真实世界后续视频；若每次只拿当前题
正确/错误奖励，且没有后续受影响的内部状态，更接近 contextual bandit。
记忆写入、遗忘和预算分配会改变后续模型状态，才可能形成有意义的序列决策。
这仍需证明收益依赖多步信用分配，而非普通分类/监督 gate 即可解释。

建议用以下记号明确系统边界，按各方案具体化，不能只复述符号：

```text
环境隐状态 x_t；当前合法观测 o_t；可见历史 h_t = o_<=t
fast memory M_t；短缓冲 B_t；剩余计算/采样预算 c_t
策略输入 s_t = g(o_<=t, M_t 的可读取摘要, B_t, c_t)
a_t ~ pi_theta(. | s_t)                 # 记忆/获取/读取操作
M_(t+1), B_(t+1), c_(t+1) = F(M_t, B_t, c_t, o_t, a_t)
yhat_(t,h) = read(M_t, recent_t, query)  # 先保存预测及时间戳
r_(t+h) = R(yhat_(t,h), o_(t+h), costs)  # 仅当该观测真实到达后
```

`M_t`、当前观测等只是策略可见的部分信息，未必满足 Markov 性；若用 POMDP 近似，
说明摘要保留了什么、遗漏了什么。读取优化也消耗预算；一次查询中多步搜索若只是
优化固定目标，应称为 inference/search，不因有迭代就叫在线 RL。

必须分开四类实验：

| 类别 | 学习发生在哪 | 测试时什么变化 | 必须说明的限制 |
|---|---|---|---|
| 离线训练 RL 初始化/策略 | 训练日模拟因果轨迹 | 冻结策略控制 TTT，fast state 可在线更新 | 固定日志 offline RL 与训练视频上重新 rollout 不同；注明行为策略与覆盖 |
| 测试时在线 RL | 当前 episode 的合法延迟反馈到达后 | 明确 policy/value/utility 哪些参数更新 | 策略本身不得跨测试题携带状态；部署必须存在相同反馈 |
| 推理搜索 | 当前 query 的可计算目标 | 临时 q 或决策序列；结束丢弃 | 无真实 reward credit assignment，不包装成 RL |
| 伪标签自训练 | teacher 对已观察视频产出目标 | student/fast state 学伪标签 | teacher 错误和成本要计入，不叫环境真值奖励 |

直接预测视觉特征、CE 动作分类、可微 learned optimizer 和监督 gate 是必要基线，
不是因不用 RL 就降低优先级。奖励为负 MSE 可以是合理工具，但如果状态转移、
决策空间和优化目标与普通回归相同，必须解释 RL 估计器的独立价值及额外方差/样本成本。

## 4. 五类候选方向与每个方案必须回答的内容

最终生成五个有实质差异的候选。以下是研究范围，不是预定胜出架构：

1. **写入与更新预算**：何时 skip/write、选择哪些 heads/层、分配多少梯度步与学习率；
   检验在固定更新 FLOPs 下是否保留对较晚预测有用的信息。
2. **保留、遗忘与状态保护**：reset、decay、分区或保护旧记忆；检验早期操作对较晚
   再访/预测的影响，明确其与监督 gate、固定衰减的差异。
3. **检索与查询地址控制**：选择读取位置、query 调整方向或搜索停止；与 Idea 3
   固定三步搜索、相同 FLOPs forward reader 比较。单次即时得分不能冒充序列奖励。
4. **按需获取视觉细节**：在当前及已合法保存帧中选择 crop、分辨率、复查或写入；
   获取动作只能改变可见信息与成本，不能访问未来帧；先验证 low/high oracle 差距。
5. **延迟反馈与信用分配**：利用已成熟观测评价先前记忆操作；研究多 horizon、
   eligibility、局部或轨迹级 credit assignment，并允许“RL 不如可微/监督”作为结论。

每个候选必须填写下面九项，缺任一项即不完整：

| 必填项 | 要回答的具体问题 |
|---|---|
| 状态 | 哪些量策略能观察？是否含持久 fast weights、摘要、预算或历史损失？大小多少？ |
| 动作 | 离散/连续操作、范围、频率、no-op，以及哪些动作在部署真实可执行？ |
| 状态转移 | 操作如何改变后续记忆/观测/预算？视频轨迹本身是否仍然外生固定？ |
| 奖励来源 | 是人工动作标签、后到视觉特征、teacher 伪标签，还是内部代理？不能混称真值 |
| 可得时刻 | 何时能计算奖励？cutoff 时尚未成熟的项如何丢弃/延迟？哪些只用于离线训练？ |
| 更新参数 | policy、critic、fast W、slow W0、encoder、readout 分别何时训练/冻结/重置？ |
| 信用跨度 | 一个动作影响几秒、多少帧/更新/查询后的结果？为何不是 one-step bandit？ |
| 部署反馈 | 部署是否提供同一种反馈？没有时测试策略应冻结，哪些声称不再成立？ |
| 替代解释 | 同反馈的 CE/MSE、监督 gate 或固定策略能否得到相同效果？如何否定？ |

不要把 1800 秒轨迹的终点动作标签复制给每个历史 chunk 当作密集语义奖励。
可在 train 使用终点标签作为稀疏离线回报，但必须说明信用分配长度、样本数只有
111 道独立题、重叠轨迹以及优化方差；不能默认有百万条独立 RL episode。

## 5. 奖励、延迟反馈与防泄漏的具体要求

- 所有在线预测先落盘（时间戳、cutoff、状态/策略版本、horizon、候选分数），再等待
  相应真实观测到达后计算误差或奖励；到达较晚的帧不能悄悄替代原目标时刻。
- 原始最后一题的测试答案只能用于评分，不得向该题的 query search、动作选择、
  policy update 或 reward model 提供信息；评分之后也不能倒写同题预测。
- 在 train 学语义 reward model 可以作为方案，但需 train-only 拟合、独立验证校准，
  明确测试用的是代理奖励，并测试 reward hacking。不可宣称代理等同于人工动作成功。
- 视觉一致性、特征 MSE、surprise、重构损失等可以合法在线计算，但必须同时报告
  最终动作/记忆任务与代理目标，检验两者脱钩；更低代理误差不是成功的充分条件。
- 稀疏最终动作 reward、密集 latent reward、计算惩罚分别消融。权重、horizon、停止
  次数、奖励归一化与策略选择仅在 train/validation 确定。
- 当前 bench_v0 是每题独立 episode。跨 episode 自我进化若是研究目标，需新增协议，
  明确流顺序和不可回看的测试段；不能悄悄改现有评测。

## 6. 最低实验矩阵与结论门槛

每种 RL 方案必须与**相同反馈、相同可见信息**的替代学习方法比较：

| 比较轴 | 最低对照 |
|---|---|
| 优化方式 | 直接 CE/MSE、监督 gate、可微 unroll/meta-learning（若可行）、RL |
| 决策策略 | 固定、随机、等更新次数、简单 surprise/不确定性启发式、learned policy |
| 记忆是否来自本视频 | direct read、查询搜索、reset/空记忆、wrong-video、冻结初始化 |
| 长历史是否必要 | 全前缀、仅近期、打乱历史但保留近期、显式检索/固定 bytes bank |
| 架构与预算 | 相同帧 ICL/检索、匹配状态容量 recurrent/SSM、同 FLOPs reader/search |
| 学习边界 | offline policy frozen、online RL、无在线更新、同反馈自训练 |

wrong-video 的选择不能读取标签或未来，须固定配对且符合时间可得性；报告目标 episode
与 donor 信息，不能从另一个更晚 cutoff 的视频导入当前题未来。reset 的关联 delta 应
为零，search 不得修改 memory 参数；候选排列应仅排列输出，不能改变各候选分数。

ICL 与 TTT 的公平表必须列帧数、帧时间覆盖、分辨率、训练标签数量、编码成本、
每次写入/读取计算量、查询搜索次数、wall time、峰值显存、fast/optimizer/KV/bank
与临时 workspace bytes。旧 ICL 约 90 帧，Idea1 每题 3660 帧，且后者用了训练标签
读出；不能据 24/57 与 21/57 宣称等预算架构胜负。

必须保存逐题预测与正确数/准确率、相对 ICL 和 direct 的配对得失、无效预测、每一天
分组、不同种子、代理奖励与任务指标变化，以及真实运行成本。至少三 seed 的扩大
实验是计划要求，不能把目前 seed 0 数据写成多 seed 结论。置信区间需说明独立单元；
仅两个测试日不足以给可靠的 day-cluster 泛化推断。

在看 test 前固定机制假设、验证集选择规则、匹配预算方法、主指标、失败/停止条件。
若 wrong-video 与真视频同样受益，或更高 reward 只增加计算却不改善预测，应否定相应
机制解释。若同反馈 CE/MSE 或固定策略相当，允许停止 RL，保留更简单的方法。

## 7. 原始文献核验与查重任务

旧广义 TTT/视频记忆文献清单保存在备份中。它是线索，引用时需核验原始论文/官方实现，
不能照抄旧阅读声明冒充本轮精读。2026-10-25 直接打开下列 arXiv 原始摘要页，核对标题、
作者和版本元数据；**本轮仅摘要，不是全文复现，也不是对其效果的独立验证**：

| 来源 | 本轮核验的范围与对本项目的关系 |
|---|---|
| [Memory-R1](https://arxiv.org/abs/2508.19828)，Sikuan Yan 等，2025-08-27；页面 v5 修订 2026-01-14 | 摘要描述以 PPO/GRPO 学习外部记忆操作与答案利用。说明“RL 控制 add/update/delete/no-op”已有覆盖；与视频 fast-weight 状态如何不同需查正文 |
| [MEM1](https://arxiv.org/abs/2506.15841)，Zijian Zhou 等，2025-06-18；页面 v2 修订 2025-07-17 | 摘要描述 RL 学习固定大小的多轮记忆与推理状态。它不是本项目已实现的 fast-weight 写入规则；需核查状态表示、反馈与 credit horizon |
| [MemRL](https://arxiv.org/abs/2601.03192)，Shengtao Zhang 等，2026-01-06；页面 v2 修订 2026-02-12 | 摘要为非参数 episodic memory 的运行时学习与两阶段检索；“runtime RL”不等于神经参数更新。官方代码入口为 [MemTensor/MemRL](https://github.com/MemTensor/MemRL)，本轮未阅读实现 |

检索工具连续返回 HTTP 502 后，以上摘要页通过直接 HTTPS 读取成功；没有把检索失败
写成“没有相关工作”。这三个入口并非完整近期综述。继续检索 RL memory management、
active perception、budgeted sensing、learned optimizer、delayed-feedback contextual
bandit、POMDP memory 和 offline RL；优先找与候选同状态/同反馈的原始论文。

逐篇记录：标题、作者、日期/版本、原始 URL、官方代码、阅读范围、奖励与可得时刻、
状态/动作、训练反馈与部署反馈是否一致、是否更新模型权重、何种任务实际验证。
每个候选至少三个最接近方法和两个非 RL/非 TTT 替代来源，形成
“相同部分—新增部分—删去新增后退化为何物”的查重表。未搜到不代表不存在。

## 8. 资源、代码与可复现性约束

实际环境为 Linux，使用 `conda run --no-capture-output -n meowbench`，不要用旧 Windows
`claude` 路径，也不要在 base 安装包。2026-10-25 会话检查是 8 张 RTX 4090 D、每卡
49,140 MiB，所有卡都有其他进程；`/home` 所在盘约 221 GiB 空闲、`/data` 约 344 GiB
空闲。这是瞬时快照，不是独占资源。训练前重新检查任务/GPU/磁盘，优先本地模型和缓存。

可复用入口：

- `runs/forecast_bench_v0/idea01_mcq_v1/features/`、`text_features.pt`；
- `forecasts/recent_predictor.pt`、`mcq/shared_readout.pt`（上述 run 目录内）；
- `ttt_frame/predictive_memory.py`、`idea1_mcq_memory.py`、`idea1_scaling_memory.py`；
- `ttt_frame/spatial_probe.py`、`key_transport.py`；
- `scripts/run_idea1_scaling.py`、`evaluate_idea1_scaling.py` 与对应 tests；
- 官方模型入口及可用性以当前项目配置/磁盘核验为准，不自动假定其他服务器 H20 可用。

旧实验校验源码 SHA；优先新增版本化模块和 runner，禁止静默改写旧 forecast/spatial/
Idea1 模块及评测脚本。大型模型、视频和缓存不提交 Git；逐题结果、协议、源码版本、
模型来源/哈希、种子、复现命令和结果文档必须保存。预估成本与实测成本分开。

## 9. 请下一轮 IdeaX 交付的内容

只生成研究提案与调研资料，不把本输入当成新一轮训练指令。先调研至少八个内部候选，
淘汰机制重复、没有合法反馈、仅奖励改名或无法在当前资源验证者，最终输出五条有区别
的方案；允许包含一条明确检验“RL 没有必要”的反向方案。

每个 `idea_card.md` 用六节：问题与假设；最近工作与增量；架构与状态/动作/转移；
训练与推理/奖励时序；实验/对照/资源；失败与扩大条件。必须包含第 4 节的九项表格，
一个严格因果的训练/部署伪代码，以及同反馈 CE/MSE 等反事实。
每个 `literature.md` 写检索日期、原始来源、阅读范围、未完成核验与查重局限。
`overview.md` 给出五方案差异、最低验证成本、可用反馈、数据缺口、先做哪个实验以及
什么结果会让项目停止 RL。

输出目录固定为 `ideax_rl_output_20261025/idea_01` 至 `idea_05` 及 `overview.md`。
不要覆盖 `ideax_output/`。最终判断应帮助用户决定：RL 是否在当前可得数据上解决了
一个真实序列决策问题，以及它何时值得比直接监督、搜索、ICL 或检索多付训练成本。
