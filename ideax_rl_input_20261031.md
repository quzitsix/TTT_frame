# 给 IdeaX 的研究生成输入：RL 如何帮助 TTT 做第一人称视频 forecasting

日期：2026-10-31。项目：`https://github.com/quzitsix/TTT_frame`。
已完成实验基线提交：`5219bae450212ab9c56947a75e25d81f4ce1ea12`。
本文件用于生成下一轮研究提案，不表示 RL 已实现或已有正结果。
新服务器执行交接见 [迁移命令](docs/NEW_SERVER_RL_HANDOFF_20261031.md)。
旧研究输入保存在 `docs/archive/ideax_input.pre_migration_20261031.md`，旧提案
`ideax_output/` 保留原文；本轮产物另写 `ideax_rl_output_20261031/`。

## 1. 请研究什么

我们关心第一人称长视频中，**参数化记忆如何帮助预测未来动作，以及什么时候比 ICL
或显式检索更有效**。请围绕 RL 控制 TTT 的写入、保留、遗忘、检索、细节获取与更新
预算，提出可证伪、能在现有数据上执行的研究方案。不要预设 RL 必然优于直接监督。

核心问题：在相同可见视频、合法反馈和计算预算下，学习一个会改变后续记忆状态的
策略，能否提高 forecasting 或在相同准确率下节省成本？收益是否真正依赖多步信用
分配，还是固定近期策略、监督 gate、CE/MSE、推理搜索或普通 answer GRPO 即可解释？

请先提出至少八个内部候选并查重，最终输出五个有实质差异的方向，推荐一个最小首实验。
可以保留一条“RL 没必要”的反向方案。不要仅把负 MSE 改名为 reward，也不要把单次
答案采样的 GRPO 直接称为“学会管理参数记忆”。所有内容标清：实际结果、已实现、
待核验、机制假设、计划。

## 2. 已有结果：应依据什么、不能假定什么

先读 `docs/IDEAX_EXPERIMENT_SUMMARY_20261026.md` 和各 Idea 的结果文档。
旧 bench_v0 为 200 道六选一题：约 1800 秒历史 + 30 秒近期视频，预测 cutoff 后
约 5 秒动作。所有佩戴者按整天划分 train111（第1–4天）、val32（第5天）、test57
（第6–7天），窗口重叠。划分归档为
`data/bench_v0_baseline_v1/research_day_split.json`，禁止随机重分。

| 已完成方向 | 真实结果 | 对 RL 研究的约束 |
|---|---|---|
| Idea1：延迟视觉残差 fast MLP | 主 residual、recent/reset 均24/57；扩容量、普通预训练降低成熟视觉MSE但未稳定改善动作 | 先建立 memory 对答案的作用；不能由 MSE 下降推导语义奖励有效 |
| Idea2：坐标搬运预检 | test684对，0对通过门槛；独立对应误差恶化0.132% | 尚未进入完整预测集成；不是0准确率，不优先扩大此路线 |
| Idea3：原生Spatial地址搜索 | 直接读9/57，三步搜索6/57，错视频搜索9/57，reset3/57 | text-V目标未训练视觉语义对齐；关联目标增加不等于动作成功 |
| Idea4：函数响应记忆 | 已保存真实教师轨迹并训练学生；所有val/test选项与teacher/reset相同，均24/57 | 不是“缺轨迹”，而是尚无预测增量；512KiB状态、当前未融合写入更慢 |
| Idea5：细节/预算预检 | 相同90源帧，low15/57、mid21/57、high25/57；混合预算运动22/57，固定末块24/57 | 已有low/high配对；固定末块为事后对照。细节与近期预算有线索，未证明记忆条件策略有效 |

旧 Qwen full ICL 为21/57，recent ICL 为23/57。Idea1/4用约3660帧与train监督读出，
旧ICL约90帧且零样本，不能据此宣布架构胜负。Idea5六主配置按val选到uniform，
对应test16/57；不能看完test再宣布high或motion是事先选定的赢家。
Idea5与旧ICL同抽帧时刻但缩放和视频包装不同，不是只差一个策略的比较。

旧57题已反复探索、只有两个测试日、标签为银标签。新选项不会让旧录制日成为独立
未见测试集。需要新录制日/参与者及独立人工审计才能做确认性结论。

## 3. 新服务器数据：用户提供路径，尚待新机实读核验

| 数据 | 路径 | 已知信息与使用边界 |
|---|---|---|
| EgoLife原始视频及官方标注 | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/EgoLife/` | 核对源视频、时间单位、佩戴者/日期、标注范围与来源；“官方”不自动等于逐动作人工真值 |
| NextMe提供的EgoLife动作标注 | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/NextMe-meta/`，主要在 `captions/2_egolife/` | 核对caption生成方式、密度、完整时间区间、是否看未来、是否可部署获得 |
| 原始benchmark | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_v0/` | 用户称200条；核对输入、标注与旧归档SHA/ID/时间轴 |
| 重写选项MCQ | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_mixed_v1/` | 用户称185条；独立版本，需映射及重跑基线，不预设仍六选一或旧schema |
| 200 case的context训练/验证数据 | `/mnt/nas/users/wuyuhang/forecasting_mcq50_200_v1/data/<case_id>/` | case内部train/val不是全局train/val；查生成器与每题证据时间 |
| 未逐条审核的开放VQA候选池 | `/mnt/gpu02/disk1/wuyuhang/workspace/TTT/outputs/case_lora/20261022T174317Z/data/<case_id>/all_vqa.jsonl` | 先筛来源、时间泄漏、重复和质量；不能直接当真值奖励 |
| 最近20 case开放SFT/GRPO数据 | `/mnt/nas/users/wuyuhang/forecasting_open_grpo_v1/20261029_v1/data/<case_id>/` | 列出实际ID/样本数/录制日/生成器；“最近20个”不是合法训练split |

新增动作标注提供了潜在语义监督入口，因此不能继续简单声称“没有历史动作标注”。
但其质量、时间对齐和**部署时可得性**尚未确认，不能默认有密集在线真值奖励。

先生成版本化 `data_inventory`、`case_mapping`、`split_manifest`、`provenance_audit`。
审计进程可读私有元数据来建立划分，推理/在线更新进程只接收合法输入与发布队列。
至少记录以下字段，未知值明确为unknown并隔离相关训练样本：

```text
dataset_version, case_id, source_video_id/hash, wearer, recording_day
context_start, observation_cutoff, target_start, target_end, timestamp_unit
source_annotation, annotation_type(human/automatic/unknown), generator_revision
generator_evidence_start/end, includes_future, includes_final_answer/options
earliest_available_at, global_split, sample/content_sha256, duplicates/overlap_group
```

185条需用 ID + 源视频 + 佩戴者/日期 + 各时间区间确认与200条的对应关系，登记删除、
重命名、无法匹配和答案变更。如果确为子集，沿用原整天分组的交集并重新统计数量，
不能直接写111/32/57。选项及标签变了，候选文本特征、读出、预测和评分必须重做；
旧答案字母不能直接拿来评分。新loader另建版本，不能放松旧严格输入校验。

context目录中的训练文件，只允许在所属全局训练日用于共用policy/reward/readout/
slow模型训练。验证/测试case内合法前缀可用于该题独立TTT，但需先审计QA生成器是否
看过完整视频、未来、最终题答案或候选，且按证据结束与实际计算延迟发布，不能启动时
把整个case的QA一次性读入。每题结束后丢弃fast/policy/critic/optimizer等在线状态。
已经用过验证/测试日资料的checkpoint登记为污染版本，不能改名成“新test”消除污染。

## 4. 五条方案的范围与推荐首实验

以下是候选轴，允许调研后替换，但最终五条必须在机制上有区别：

1. **细节获取与写入的联合预算**：用当前低清观察、历史记忆摘要和剩余预算，决定
   skip / write_low / acquire_high_and_write；证明确实把预算花在对后续预测有用处。
2. **保留、遗忘与状态保护**：学习decay/reset/保护哪些分区，研究较早写入如何影响
   较晚再访和动作；与固定衰减、FIFO/显式bank、监督gate及无历史比较。
3. **语义对齐后的检索与停止**：先用训练日对齐读出，再控制读取地址、检索步数或
   停止；与直接读、等计算搜索及多次forward比较，防止重演Idea3代理目标失配。
4. **延迟反馈控制更新时机和强度**：依据已成熟的预测误差决定何时更新、步数/学习率；
   比较同反馈MSE/CE、可微unroll与RL估计器，检验多horizon信用分配的必要性。
5. **输出训练的反向验证**：SFT/answer GRPO、显式检索或context适配是否已解释收益？
   若单轮输出训练即可匹敌记忆controller，明确结论是无需复杂记忆RL。

每个候选必须逐项回答，不能只列术语：

| 项目 | 必须明确的内容 |
|---|---|
| 状态 | 只能用当前/过去观测、可读记忆摘要、已成熟损失、预算；大小和信息缺失是什么？ |
| 动作 | 取值范围、频率、no-op、合法动作mask、真实部署可执行性 |
| 状态转移 | 如何改变fast参数、可见信息、buffer、预算；视频本身是否外生固定？ |
| 奖励来源 | 人工标签/动作caption/teacher伪标签/后到视觉特征/内部代理分别是什么？ |
| 可得时刻 | 证据何时完整、计算延迟多久、哪些只离线可用、未成熟项如何处理？ |
| 更新参数 | policy/critic/fast W/slow W0/encoder/readout各自训练、冻结、重置时刻 |
| 信用跨度 | 一次操作影响多少后续chunk/预测；为何不是one-step bandit？ |
| 部署反馈 | 部署是否真的提供同类奖励？没有时哪些参数冻结、哪些声明不成立？ |
| 替代解释 | 同反馈监督gate、CE/MSE、固定策略或推理搜索如何反证该机制？ |

推荐首实验是小动作空间的**离线训练controller + 测试时冻结policy的TTT控制**。
先固定一个小参数记忆后端，不同时更换基座、读出和RL算法。保留相同低清源帧，
按预算区间约束高清获取与写入次数；高清内容只能在动作选定、计费后对策略可见。
允许访问已合法保存帧但必须计入buffer bytes，不能无限回看原视频而声称纯参数记忆。
先证明最强固定策略中的真视频memory优于reset/wrong-video，或在相同质量下节省成本。
若没有，不扩大到全Spatial或多卡长训练。

## 5. RL、GRPO及奖励的严格定义

| 类型 | 训练/测试行为 | 不能混淆的概念 |
|---|---|---|
| 训练阶段RL学策略 | 仅训练日重新rollout，测试冻结controller，fast状态可变 | 这是离线阶段训练，不自动等于只用固定日志的offline RL |
| 固定日志offline RL | 只用已有行为轨迹，明确行为策略和覆盖 | SFT/GRPO数据文件不一定含状态/动作/奖励轨迹，不能默认可用 |
| 测试时在线RL | 当前episode收到合法延迟反馈后更新policy/value | 无部署语义奖励时，不允许借官方离线标注或终题标签实现此声称 |
| 推理搜索 | 优化当前可计算目标，结束丢弃临时query | 有迭代不等于RL，搜索成本和代理目标也需对照 |
| 伪标签自训练 | 因果teacher产出目标，再做CE/MSE | teacher反馈不是环境真值，需算生成成本与错误 |

被动视频里，选择答案通常不改变未来真实视频；一次答案奖励接近contextual bandit。
写入/遗忘/获取会改变模型未来内部状态，才有潜在序列控制价值，但仍需实验说明。

如用GRPO，明确group采样的是**多个答案**还是**多条记忆操作轨迹**，如何从回报
得到token/操作级优势、如何处理零方差group、KL/reference及长轨迹信用分配，报告
group大小、rollout次数、token/FLOPs和训练预算。已有SFT/GRPO目录不是完整训练器。
不要把同一终题标签复制到1800秒内所有chunk作为密集语义奖励。

离线训练可用训练日终点动作标签作稀疏回报，但只有有限且重叠的episode；还可在训练
日前缀内构造多个预测锚点，标签区间不得越过全局训练边界，且不能把锚点当独立视频。
动作caption若描述区间[t0,t1]，最早在t1及生成延迟后可用；若generator需要全片，
则不能在片内提前使用。把动作caption变成reward要说明语义匹配、置信度与错误传播。

测试时终点预测必须冻结落盘，终题答案和cutoff后视频只由评分器使用。若实验允许
测试policy在线更新，回报必须来自cutoff前已真正成熟的视觉反馈/因果teacher；
该反馈与训练时终点语义奖励不同，需要分别报告。官方/NextMe注释只在明确存在同类
在线标注服务时才可作为部署奖励，否则仅用于离线训练和隔离评分。

严格因果伪代码（各方案需提供自己的具体实现）：

```text
offline: audit -> split by source days -> fit on train -> select on validation
freeze policy/critic/readout/slow initialization; register budget and test protocol
for each evaluation episode independently:
    reset memory, optimizer, buffers, pending predictions and optional online policy
    for each observation time t strictly before cutoff:
        receive only observations/evidence whose available_at <= t
        settle past predictions with newly matured evidence; never overwrite predictions
        optionally update online policy only in the explicitly declared legal-feedback arm
        observe low-res and permitted memory summary; choose a_t under budget mask
        acquire/pay for requested detail; apply memory action; debit compute/storage
        save intermediate forecast with timestamp/state/policy version and target time
        enqueue it; feedback cannot arrive in the same step from later observations
    save final answer/scores and freeze prediction artifact
    discard unearned rewards and episode state; do not adapt using final target
separate scorer: read frozen answers and private labels; paired analysis and examples
```

## 6. 最低实验矩阵、评估与停止条件

先分开三件事：获取更多/更清楚的信息、记忆本身、RL优化器。第一阶段少量训练case
做机制和预算验证，validation固定配置后才统一跑test；扩大再做至少三seed。

| 轴 | 必须有的对照 |
|---|---|
| 输入和读出 | recent/full ICL、同预算显式检索，基座/预处理/问题/可见选项一致 |
| 获取策略 | 固定区间末块/偏近期、统一中清、随机、简单变化或不确定性阈值 |
| 写入 | 不更新、固定等次数TTT、监督gate、RLcontroller |
| 反馈公平 | 相同训练标签/伪标签/成熟视觉反馈下CE/MSE、可微优化（可行时）、RL |
| memory因果性 | 当前视频、reset/冻结初始化、固定wrong-video、删关键历史、仅近期 |
| RL时段 | 冻结离线policy和合法在线policy分开；输出GRPO与记忆操作RL分开 |

wrong-video配对不能看答案，donor不得把相对目标时间尚未可得的观测带入目标episode。
候选编码不含选项字母/位置，候选重排只重排分数；开放式生成不提供MCQ选项，另立
协议，不能与有选项准确率直接当同任务比较。若用候选似然，要固定长度归一化等规则。

MCQ保存每题完整选项、实际选择、所有候选分数/概率、参考答案（仅评分后联结）、
无效输出、相对最强固定策略/ICL/direct的paired gains/losses、分日结果和种子。
开放回答保存原始生成文本、目标时间段、参考动作、评分器版本、人工盲审样本及
逐项verb/noun/语义匹配与歧义；不能只贴reward均值，也不能生成事后理由冒充模型推理。
训练reward model和评估judge分开，若共用或来自同一teacher，标明相关偏差。

预算表至少含：源帧数/时间覆盖/分辨率/视觉token、训练标签与样本数、额外teacher
调用、policy和search步数、写入梯度步数、训练/推理wall time、峰值显存、fast权重/
optimizer/KV/buffer/bank/workspace bytes。相同输入预算和相同总计算预算分别比较。

预先在validation设定主指标、最低增量或等质量节省门槛、主seed及三seed扩大条件：

- 真视频记忆不优于reset/wrong-video：不能声称记忆因果收益，先定位输入/读出。
- RL不优于最强固定近期策略及同反馈监督gate：停止扩大RL，保留更简单方法。
- reward升但forecasting不升：否定当前奖励对齐解释，不能只报告代理曲线。
- 收益仅来自更多高清/更多teacher/更多更新：归因为信息或预算，不能归为RL。
- 185新表必须重新跑全部基线，不把旧21/57插入新版准确率比较。
- 旧测试只有两天且反复使用，不做IID显著性夸张；确认性结果另留未使用源录制日。

## 7. 文献核验与方案查重

只引用原始论文/官方实现，记录日期、版本、作者、URL、实际阅读范围、状态/动作/
奖励与反馈时间、训练/部署差别，以及是否真正更新参数记忆。未读全文要直说。
2026-10-31通过原始arXiv摘要/元数据和官方仓库核验的起点如下，非完整综述：

- [Memory-R1](https://arxiv.org/abs/2508.19828)：RL管理与利用外部记忆的相关工作；
  “add/update/delete/no-op”本身不是新贡献。当前只核验摘要/元数据。
- [MEM1](https://arxiv.org/abs/2506.15841)：RL联合记忆与推理的相关工作；需进一步
  查全文明确状态表示与反馈，不默认等于视频fast weights。当前只核验摘要/元数据。
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)：GRPO的原始来源之一；数学输出
  优化的成功不保证视频记忆轨迹可用相同奖励。当前只核验摘要/元数据。
- [Spatial-TTT](https://arxiv.org/abs/2603.12255)与[官方实现](https://github.com/THU-SI/Spatial-TTT)：
  核验摘要和官方README。nano是官方SFT模型；本地参数状态边界不是官方完整混合协议。
- [EgoLife](https://arxiv.org/abs/2503.03803)：原始数据论文摘要/元数据；不能据此
  声称新机NextMe caption或生成VQA已经逐条人工审核。
- [NextMe-800](https://arxiv.org/abs/2610.01461v1)与[官方项目页](https://kkkkawayi.github.io/nextme-800/)：
  核验摘要、项目页与论文HTML第4/6节。文中的caption流程使用完整30秒窗口及多模态
  信息，层级摘要还会合并更长跨度；NextAct的1500点、next-1/10和best-of-three协议
  不等于本项目200/185条MCQ。新机EgoLife衍生标注是否沿用此流程仍待本地核验。

若caption沿用完整窗口生成，只有 `event_end <= cutoff` 仍不够，必须检查
`teacher_source_window_end <= cutoff`，包括全部OCR/语音/多模态与高层摘要的来源。
生成窗口末端的未来信息不能包装成较早事件的在线奖励。上述网络核验在web工具502后
通过原始arXiv页面/API和官方GitHub API完成，不是对本地数据质量的核验。

请补查active perception、budgeted sensing、learned optimizer、delayed feedback、
POMDP memory与offline RL。每条最终方案至少三个最接近方法和两个非RL/非TTT替代，
用“已有相同部分—本方案新增—删除新增后退化为何物”表格查重；未搜到不代表不存在。
若检索失败，保存失败与未核验项，不编造引用或“已全文阅读”。

## 8. 工程边界与最终交付

Linux服务器，用conda `meowbench`，Python3.11；先查GPU、进程、磁盘和现有模型，
不要在base安装包。不假定新机是H20或仍有8张4090。优先本地模型、缓存，无授权的
付费teacher不要自动调用。本输入不要求此刻启动大训练。

可复用：`ttt_frame/videoqa.py`（LoRA路径）、`spatial_memory.py`、`spatial_probe.py`、
`idea1_mcq_memory.py`、`predictive_memory.py`与Idea5获取实验；查各自协议，不能直接
混用Spatial关联写入与Idea1 MSE目标。Idea1的features、recent_predictor和shared_readout
主要在被Git忽略的runs中，需另迁移或重算。新选项不可复用旧text_features作新候选。
旧缓存记录绝对视频路径/mtime，旧协议校验源码SHA；新增版本化loader/runner和路径
映射，保留原归档字节，不能sed路径后冒充原协议。迁移命令文档给出实际可运行验收，
目前尚无本轮RL训练runner，不要编造现成训练命令。

最终目录：`ideax_rl_output_20261031/`，不要覆盖旧 `ideax_output/`：

```text
overview.md                  # 五方向差别、优先级、最小实验与停止RL条件
data_audit_plan.md           # 七类数据的schema/来源/时间/分组核验及待办
experiment_matrix.md         # 预注册指标、预算、baseline、val选择规则和失败门槛
idea_01/ ... idea_05/
    idea_card.md             # 六节：问题假设；文献增量；架构九项；奖励时序；实验成本；失败条件
    literature.md            # 原始出处、阅读范围、查重表、未核验项
implementation_plan.md       # 只引用真实现有模块；新增模块/CLI标记为待实现
```

每条idea给训练与部署两段因果伪代码，列清策略输出、fast状态更新和最终答案的连接。
首实验给能在一天内完成的分阶段预算估算，实测资源后修订，禁止把估算写成完成记录。
请特别回答：数据是否真的提供合法RL反馈、为什么用RL、何时不值得用RL、怎样从逐题
具体回答确认收益来自记忆而不是输入/监督/预算差异。
