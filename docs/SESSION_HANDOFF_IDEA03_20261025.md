# Idea 03 会话交接与磁盘核验（2026-10-25）

本文保存用户在新会话提供的交接要求，并区分历史记录与本次实际核验。

## 本次启动核验

- 已实际执行 `pwd`、`git status --short`、GPU/进程/磁盘检查。
- 工作目录：`/home/quzitsix/TTT_frame`；Python 使用 `conda run -n meowbench`。
- `/AGENTS.md`、`/home/AGENTS.md`、`/home/quzitsix/AGENTS.md`、仓库根
  `AGENTS.md` 均不存在；仓库未发现嵌套 AGENTS。读取了
  `/home/quzitsix/.codex/AGENTS.md`，内容为历史 Windows 说明；按本次用户明确要求，
  使用实际 Linux 环境与 meowbench，不向 base 安装包。
- HEAD：`2cd12bfcc72d68694e027f87fb78d6337cc387c3`，本地 `main` 与
  `origin/main` 一致；远端 `git@github.com:quzitsix/TTT_frame.git`。
  此处是启动时本地 remote-tracking 状态，最终推送前另行核验远端。
- 启动时仅有以下未跟踪内容：
  `docs/SVCBENCH_TESTSET_README.md`、`ideax_input.md`、`ideax_output/`。
  前者和原始 idea 输出不得清理或覆盖；input 修改前保存原文备份。
- 实际 8 张 NVIDIA GeForce RTX 4090 D，每卡报告 49,140 MiB。
  GPU 0/3/4 分别占约 10,579/12,791/11,147 MiB；1/2/5/6/7 各占约 45–46 GiB。
  均有其他任务；本用户无在跑的训练/评测。不终止其他人的进程。
- 项目盘 `/dev/sda2`：879G，总可用约 221G；数据盘 `/dev/nvme0n1`：7.0T，
  可用约 344G，95% 使用率。优先复用缓存并限制新增大产物。
- 现存缓存大小：Idea1 约 5.7G，scaling 约 3.3G，baseline 约 9.5M。
- 官方权重已发现：`/data/quzitsix/models/Spatial-TTT-nano/model.safetensors`。
  具体可接入范围、协议和实验结果以本轮后续文档为准。

## 一、当前目标和授权

研究第一人称视频的参数化记忆与 forecasting：TTT 如何改善预测、何时超过 ICL。
优先完成 Idea3 在真实 `bench_v0` 上的 forecasting 评测，与 full ICL 比较，加入
直接读取、查询地址搜索、重置/空记忆、错误视频记忆和冻结初始化对照，保存逐题结果、
复现命令和完整实验文档。备份并更新 `ideax_input.md`，转向 RL 如何帮助 TTT。

用户已授权实现、训练、评测、必要文件修改及相关 Git 提交和推送。提交前审查差异，
只提交本任务文件；不将启动实验当完成，不仅给计划。

## 二、实际环境

- Linux 服务器，项目 `/home/quzitsix/TTT_frame`。
- Python/训练：conda `meowbench`，禁止向 base 安装包。
- 数据 `/data/share/raw/bench_v0`；优先本地模型/缓存/已有协议，注意付费 API 用量。
- 历史 Windows 描述不适用本次服务器操作。
- 旧会话工具接口失败不能延续为本会话结论；本会话已实际验证终端可用。

## 三、历史磁盘状态

历史未跟踪文件为 `docs/SVCBENCH_TESTSET_README.md`、`ideax_input.md` 和
`ideax_output/`，启动核验相符。原始 idea 必须保留，input 已获修改授权但先备份。
历史称 scaling 已提交推送至 `2cd12bfcc72d68694e027f87fb78d6337cc387c3`；
本次已核验本地 HEAD 和 remote-tracking ref，最终远端仍需核验。

## 四、数据及因果边界

历史实验：200 道六选一未来动作题，每题约 1800 秒长历史和 30 秒近期视频，预测
cutoff 后约 5 秒动作。6 位佩戴者、7 天，题目窗口重叠。沿用
`runs/forecast_bench_v0/research_day_split.json`：所有佩戴者共同按整天划分，
train111 为第 1–4 天，validation32 为第 5 天，test57 为第 6–7 天。

不得随机重新划分；每题独立初始化在线状态；预测仅使用 cutoff 前视频、公开问题
和选项。测试答案与 cutoff 后视频不能参与更新、查询搜索或模型选择。
train 标签可用于规定的离线训练，超参数仅用 validation；答案到视觉目标的映射如需
拟合只能使用 train。延迟更新必须先保存预测，待对应观测真正到达再评分与更新。
test57 已反复探索，结果必须标为探索性比较；当前为银标签，不能宣称独立人工核验。

## 五、Idea1 与 scaling 历史记录（需逐题核验）

Idea1 为冻结视觉编码器后的独立预测 fast MLP，是受参数记忆启发的原型，不等于
完整官方 Spatial-TTT。缓存 `runs/forecast_bench_v0/idea01_mcq_v1/features/`：
每题约 `[3660,8,256]`，2 fps，完整约 1830 秒前缀；冻结 Qwen3-VL-2B 视觉特征经
空间池化与投影。已有 `forecasts/recent_predictor.pt`、`text_features.pt`、
`mcq/shared_readout.pt`。

Scaling 为 12 配置×200 题：hidden128/512/2048，fast 参数
262,144/1,048,576/4,194,304，random u1/u4、pretrain200/800。
预训练为 train111 内合法视觉预测对的普通回归，不是端到端元训练。
历史成绩：recent+shared readout 24/57 (42.11%)；所有 u1 residual 及相应冻结
初始化 24/57；random u4 residual 23/57。成熟预测 MSE：recent 0.513887，
H128 random u1 residual 0.496274，H2048 pre800 residual 0.487428，对应冻结
初始化 0.503286。MSE 改善未稳定转化为动作收益；不能推出预测饱和或 RL 必然更好。
MSE 来自 cutoff 前已成熟预测，不能当最终未来动作成绩。

入口：`docs/IDEA01_SCALING_RESULTS.md`、`runs/forecast_bench_v0/idea01_scaling_v1/`、
`data/bench_v0_idea01_scaling_v1/`、`scripts/{run,launch,evaluate,summarize}_idea1_scaling.py`、
`ttt_frame/idea1_scaling_memory.py` 和 `idea1_scaling_pretrain.py`。
历史全测 `532 passed` 仅为历史记录，本轮按实际修改重测。

## 六、Idea3 实现要点

先读 `ideax_output/idea_03/{idea_card,literature}.md`、
`docs/IDEAX_IMPLEMENTATION_ROADMAP.md`、`ttt_frame/spatial_probe.py` 及相关测试。
目标是通过优化临时查询地址读固定 fast weights，检验已写入的信息是否难以直接读取。

已有 `association_delta = <f_W(q),v>-<f_W0(q),v>` 和
`projected_inverse_address_search`：仅更新临时 q，W/W0 冻结；每 head/token 的
L2 球约束；保留包括初始地址的最佳迭代；各候选独立且排列等变。
这些局部工具不是完成的真实视频 forecasting 实验。

Spatial 使用点积关联目标，Idea1 使用 MSE 预测 MLP，两者不能直接套同一目标。
先评估真正 Spatial 接入；若先做 Idea1 适配，须明确命名和目标变化，不能冒充
完整官方 Idea3。Idea1 旧产物未保存完整 fast states，可能需重放前缀。
候选答案嵌入排除选项字母和位置线索。

最低对照：直接读、查询优化（可从 3 步起，仅 validation 定设置）、重置/空记忆、
错误视频记忆、相应冻结初始化、full ICL。空记忆 delta 必须为零；测试搜索不改
记忆、候选排列不泄漏、收益是否来自当前视频。

## 七、ICL 和报告要求

历史 full ICL 为 21/57 (36.84%)、recent ICL 为 23/57 (40.35%)，必须从已有
逐题产物重算。入口：`docs/BENCH_V0_MCQ_BASELINES.md`、`data/bench_v0_baseline_v1/`、
`docs/IDEA01_BENCH_V0_MCQ_RESULTS.md`、`data/bench_v0_idea01_v1/`、
`scripts/evaluate_forecast_mcq.py`、`ttt_frame/forecast_mcq.py`。

旧 ICL 约 90 帧，Idea1 密集缓存约 3660 帧，不能只凭准确率宣布架构胜负。
尽可能比较相同输入预算；否则列清帧数、时间覆盖、监督数据、搜索次数与成本。
报告同题集正确数/准确率、逐题预测、相对 full ICL/直接读的配对得失、重置/错视频、
validation/test 角色、输入/计算预算、种子、源码版本、协议、模型来源与限制。

## 八、旧实验保护

优先新增版本化模块/runner，保护旧源码 SHA 协议。尤其不静默修改
`forecast_mcq.py`、`forecast_bench.py`、`forecast_features.py`、`spatial_videoqa.py`、
`spatial_model.py`、`spatial_memory.py`、`idea1_mcq_memory.py`、`idea1_mcq_readout.py`、
`forecast_residual.py`、`predictive_memory.py` 和旧缓存/运行/评测脚本。
如必须改共享实现，先审查协议依赖，并采用版本化方式。

## 九、RL 研究转向

旧会话只形成方案，尚未写 input。本次先保留原文，再按核验结果更新：
RL 控制写入、保留、遗忘、更新强度，以及检索、查询地址、细节获取，使用后续合法
观测到达后的延迟反馈。

每个方案写清状态、动作、状态转移、奖励来源及可得时间、更新参数、信用分配跨度、
部署时反馈。被动视频单选类别通常接近 contextual bandit；影响后续状态的记忆操作
更可能构成序列决策。不能把负 MSE 改名 reward 就声称新 RL 机制。
区分离线 RL 初始化、测试时在线 RL、推理搜索、伪标签自训练。加入同反馈 CE/MSE、
监督 gate、固定/随机策略、同预算 ICL/检索。无密集历史动作真值，不能默认语义奖励；
最终测试答案仅评分，不能给当前题在线奖励。文献只据原论文/官方实现核验。

其他 idea：Idea2 已有 Procrustes 和拒绝条件；Idea4 缺完整教师状态轨迹；Idea5
适合 RL 获取/写入策略但缺低/高分辨率配对验证。本次优先 Idea3。

## 十、交付与后续状态

交付代码、必要测试、真实视频评测、完整文档、逐题结果与复现命令；实际阻塞保留日志，
明确完成/未完成。提交前审查并只提交任务相关文件，随后推送。
以上为本次收到的交接内容；以下为实际完成情况。

### 本轮完成状态

- 已完成官方 Spatial 原生关联地址诊断的实现、全部 200 题视频前缀重放、validation
  选择、预测冻结和独立评分。不是 Idea1 适配，不宣称完整 outer-trained Idea3。
- 结果：直接读取 val8/32、test9/57；搜索 val6/32、test6/57；reset/冻结初始化
  val4/32、test3/57；错视频直接读 test10/57、搜索9/57；Qwen full ICL test21/57。
  搜索相对直接读改对3/改错6，相对full ICL改对3/改错18。本轮是探索性负结果。
- 200/200 题与旧 full ICL 的实际 90 帧像素、时间及采样一致。采用同帧预算，仍有
  checkpoint、读出和计算差异。所有因果限制和银标签限制保留。
- 正式运行 `runs/forecast_bench_v0/idea03_native_spatial_v2/`，GPU4 四 worker，
  938.914 秒，全部正常结束。其他 GPU 任务未终止；无付费 API、无依赖安装。
- 完成后再次检查：项目盘可用约 219G、数据盘约 340G；提交归档约 19.73 MB。
  GPU4 已释放本任务模型显存（检查时仅余约 919 MiB，其他进程状态可能继续变化）。
- [结果、协议与复现命令](IDEA03_BENCH_V0_RESULTS.md)；
  [逐题归档](../data/bench_v0_idea03_spatial_v1)；
  [旧基线核验](IDEA03_BASELINE_AUDIT.md)。归档含核心 SHA 清单与附加审计日志。
- 最终完整测试 **668 passed in 65.08s**。独立复算 200 题预测、39 组指标、144 组
  配对计数/ID 以及帧证据全部通过。旧冻结源码无变动。
- `ideax_input.md` 已转向 RL 的可检验问题并写入此次实际负结果。原文完整备份于
  `docs/archive/ideax_input.pre_rl_20261025.md`，SHA256 为
  `6ff75a6c34f27a7713bcf6152c3c306aa48e56725cad79c78c9a0541b4aa3859`。
  `ideax_output/` 与 `docs/SVCBENCH_TESTSET_README.md` 保持原样、不纳入本次提交。
- 提交和推送按用户授权执行，提交哈希以最终 Git 日志/交付消息为准。没有留下未完成
  的实验进程；完整端到端 Idea3 训练及 RL 机制实验属于后续研究，不作本轮完成声明。
