# Idea2 / Idea4 / Idea5 实验会话（2026-10-26）

用户要求继续其他 idea 的实验，全部结束后写总结 Markdown，重点解释方向和尝试结果，
并检查具体题目的回答效果。本轮延续此前的实现、训练、评测及相关 Git 提交/推送授权。

## 启动核验

- 已实际执行 pwd、git status、GPU/进程/磁盘检查；工作目录 `/home/quzitsix/TTT_frame`。
- 基础提交 `8e1d524deabb7ed5eba5427833f633629ee0e2dc`。仅有原有未跟踪
  `docs/SVCBENCH_TESTSET_README.md` 和 `ideax_output/`，保留原样。
- 无遗留训练/评测进程。8 张 RTX4090D，各约49GB，启动检查全部空闲。
- 项目盘可用约218GB，数据盘仅约160GB、98%占用。新产物放项目 runs，限制缓存。
- Python 使用 conda `meowbench`。读取历史全局 Windows AGENTS，按用户明确的服务器
  环境要求执行；祖先及仓库没有额外 AGENTS。无需安装新依赖或调用付费模型。

## 本轮实验边界

沿用 bench_v0 全200题及 train111/validation32/test57 整天共同划分。
每题独立状态；只用已见视频作在线更新；test答案只评分，不作选择或奖励。
test为反复使用的探索性银标签题集。候选读出/输入/预算不同时明确说明。
旧源码及旧产物协议不改；只新增版本化模块、脚本、测试和归档。

三方向并行：

1. Idea2：真实相邻视觉 token 的低秩坐标变换检验。train拟合投影，validation选择
   设置，匹配分 fit/gate/audit，比较 identity/错匹配/错视频。门槛不成立时停止完整
   transport训练，不把代数等价当真实视频任务收益。使用 GPU0/1。
2. Idea4：明确命名为 Idea1 小MLP教师上的函数响应原型，采集真实前缀连续写入轨迹。
   检验 probe压缩、单步/20步响应传播、固定核与依赖状态的转移及forecast读出。
   教师辅助的warm-start诊断与完整前缀学生分开。使用 GPU2/3。
3. Idea5：同90帧、相同分chunk形式比较低/中/高分辨率和固定/随机/运动细节策略。
   先验证细节获取的前置条件；不冒充已训练四分支、依赖TTT记忆的完整selector。
   只用当前/过去缩略图作决策，记录实际tokens和成本。使用 GPU4/5。

根任务负责跨方向校验、全200题答案文本对照、确定规则选择具体例子和可见前缀回看，
最终总结及相关提交。GPU6/7暂不占用。例子只作评分后的诊断，不反馈给任何模型。

## 最终产物

全部实验已实际结束，各方向的范围/停止点均明示，不再有本任务GPU进程。
完成时复核8张GPU均回到约15MiB空闲占用；项目盘可用约214GB，数据盘约160GB。

- Idea2：200题×12帧对、三个匹配条件共7200条；test684对均eligible，0对接受；
  独立audit误差恶化0.132%。按预设门槛停止完整搬运，不伪造forecast答案。
- Idea4：采集全部200题真实教师轨迹，训练6个容量/核配置，完整前缀学生及warm/copy/
  oracle共10组均val14/32、test24/57；val/test答案逐题相同。学生状态512KiB，
  教师1MiB；本地未融合核写入更慢，不能声称部署加速。V1序列化重复存储已通过
  V2逐题clone重放修复；只移除本次产生且被替代的V1大缓存，保留协议/日志。
- Idea5：200题×6预定配置、另200题固定末块归因对照。test：低15、中21、高25、
  固定首块16、随机17、运动22、固定末块24。末块对照在主结果评分后追加，明示事后
  分析；支持细节/近期预算线索，未建立运动或TTT记忆条件选择的独立增量。
- 总报告：[IDEAX_EXPERIMENT_SUMMARY_20261026.md](IDEAX_EXPERIMENT_SUMMARY_20261026.md)。
  分方向报告 `IDEA02_BENCH_V0_RESULTS.md`、`IDEA04_BENCH_V0_RESULTS.md`、
  `IDEA05_BENCH_V0_RESULTS.md`，均含精确协议、结果、预算和复现命令。
- 归档 `data/bench_v0_idea02_v1/`、`bench_v0_idea04_v1/`、`bench_v0_idea05_v1/`。
  汇总 `data/bench_v0_ideax_summary_v1/` 保存200题×25组实际选项文字、配对、
  8个按规则选择的例子及6帧/例的可见前缀图；未打开future视频作样例回看。
- 独立复算了Idea2缓存/模型/分片，Idea4全部200题张量/预测及指标，Idea5全部200题
  主组预测/token/概率/指标与配对。核心与补充SHA清单覆盖归档。
- 全仓显式离线测试 **711 passed in 33.78s**。首次全测710通过、1旧CLIP项因网络
  探测超时失败；强制HF/Transformers离线后该项和全仓均通过。失败与成功日志均保留。
- 旧冻结源码没有修改；仅为新实验新增模块/runner/测试，更新路线图与RL输入的证据。
  原始 `ideax_output/` 和无关 `docs/SVCBENCH_TESTSET_README.md` 原样保留、不提交。
- 最终相关变更审查后按授权提交推送；具体提交哈希以Git日志与交付消息为准。
