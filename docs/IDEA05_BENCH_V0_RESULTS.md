# Idea5：视觉细节获取的分辨率与预算预检

日期：2026-10-26。正式运行目录为
`runs/forecast_bench_v0/idea05_acquisition_v1/`，逐题归档为
`data/bench_v0_idea05_v1/`。全部200题×6个预先固定配置已完成并独立评分。
高清输入改善了本轮探索性预测成绩；简单选择策略的效果不一致，尚未证明
记忆条件收益分解或TTT本身有效。

## 研究问题与范围

Idea5 完整提案希望利用当前缩略图和参数记忆，预测“额外高清会补充多少未来有用
证据”，并通过有/无记忆的四分支离线收益训练选择器。本轮先检验其必要前提：
同一组已观察帧中的高清细节是否改善 forecasting，以及简单因果选择能否以较少
视觉 token 保留这种改善。

本轮是冻结原始 Qwen3-VL-2B 的感知与获取预算预检。没有训练选择器，没有四分支
长期记忆收益，没有接入 TTT/GRU/bank 的固定状态在线递归，不能称为完整 Idea5。
ICL 仍保存完整选择后的视觉输入直至预测；固定参数记忆的丢帧部署条件没有被验证。
“动作正确率提高”也不能直接证明高清识别了某个物体或保留了某个事实。

## 数据、因果边界和六个配置

沿用 200 道 bench_v0 六选一未来动作题：约 1800 秒历史、30 秒近期、cutoff 后
五秒目标。整天划分 train111（第1–4天）、validation32（第5天）、test57（第6–7天）。
推理只读取公开题目、选项以及本题的 `context.mp4`/`input.mp4`。所有配置在查看
本轮标签前固定，不使用 validation 选择阈值或配置；分别保存全部结果。
冻结 200 题六配置预测及 SHA 后，独立评分调用才读取答案。

同轮六配置共享同一组 90 个实际帧：历史 30 帧（60秒间隔）、近期60帧（0.5秒间隔），
起点包含、终点不包含，时间戳只取请求时刻或更早。先统一解码为最长边448的RGB，
再派生低/中分辨率。每六帧组成一个视频段，历史5段、近期10段，共15段；角色边界
不混合，不补充额外帧。所有配置使用相同的15段顺序包装及相同问题/选项和 A–F
next-token 约束读出。

| 配置 | 输入分辨率和选择规则 |
|---|---|
| low112 | 全部90帧最长边112 |
| mid224 | 全部90帧最长边224；同轮中等分辨率对照 |
| high448 | 全部90帧最长边448；充分高清输入，不是使用答案的选择oracle |
| uniform_high | 每预算区间第一个六帧段高清，其余低清 |
| random_high | 每区间预先按题目ID和seed=0确定一个随机段高清 |
| motion_high | 顺序读取当前段低清缩略图，运动量超过0.1则高清；未触发时选区间最后一段 |

三个预算配置均使用同样的四个公开索引区间 `[0,4), [4,8), [8,12), [12,15)`，
每区间恰好一个高清段，总计24/90帧高清（26.67%）。motion 取当前段五个相邻帧
对的32×32灰度平均绝对变化（缩略图先经112下采样、数值除255）。选择发生在当前
六帧都已到达之后，不比较后续段的运动量，也不使用问题/选项、类别、答案或标注
时间。最后段兜底只用公开预算端点，单元测试验证修改未来缩略图不影响过去选择。

这不是卡片中“每个2fps时刻即时决定”的 token bucket：本轮使用稀疏固定候选和
六帧段决策，历史段跨度300秒、近期段跨度2.5秒；预算按段而非每秒计费。
该限制不应隐藏。实验 harness 为复用同一媒体会在CPU保留所有90张448图像；
策略函数本身只依赖当前/过去低清信息，但没有测真实丢帧系统的内存与延迟。

## 实际编码预算和原 ICL 比较边界

112 是输入缩略图尺寸；Qwen processor 根据 patch/merge 规则实际将其对齐到
128×128，而不是严格112×112。每段 grid 为 `(3,8,8)`，合并后48视觉token。
224 每段 grid `(3,14,14)`、147 token；448 为 `(3,28,28)`、588 token。
实际序列预算如下，不能把112/448像素比直接当 FLOPs 比：

| 配置 | 高清帧 | 合并后视觉token/题 | 相对全高清 |
|---|---:|---:|---:|
| low112 | 0 | 720 | 8.16% |
| mid224 | 0 | 2205 | 25.00% |
| high448 | 90 | 8820 | 100% |
| 任一预算策略 | 24 | 2880 | 32.65% |

预算策略使用的是CPU缩略图统计，不执行独立低清神经编码器；因此不能把这里的
成本当作完整“low encoder + learned selector + optional high encoder + memory”成本。
本次仍需解码每个候选图像，解码、预处理、前向、CPU缩略图与端到端实验耗时均记录。
共享的thumbnail统计计算也为 uniform/random 执行，作为本预检的共同开销；正式
部署这两个策略可以省去不需要的统计，但本轮不以未测优化报告速度。

旧 full ICL 的21/57保留为外部参考。时间戳/源文件审计为 **200/200一致**，但
新 mid224 与旧224的像素哈希为 **0/200一致**。旧基线将90帧合成一个视频输入，本轮是15段；
此外旧版从原图直接缩到224，本轮是先448再224，像素不保证相同。因此新 mid224
与旧 full ICL 的差异包含时间包装和缩放路径，不能解释为纯高清或纯包装效应。
同轮六组的时间点、题目、模型和包装保持一致，才是分辨率/选择策略的主比较。

## 结果、配对与运行成本

测试集已反复使用，且为银标签、同日窗口重叠；以下是探索性正确数、配对和NLL，
不作IID显著性结论。NLL 使用六个A–F logits归一化后的标签概率，不是无约束生成NLL。

| 配置 | Train111 | Validation32 | Test57 | Test准确率 | Test NLL |
|---|---:|---:|---:|---:|---:|
| low112 | 14/111 | 4/32 | 15/57 | 26.32% | 2.2813 |
| mid224 | 30/111 | 6/32 | 21/57 | 36.84% | 1.8826 |
| high448 | 37/111 | 8/32 | 25/57 | 43.86% | 1.7291 |
| uniform_high | 28/111 | 9/32 | 16/57 | 28.07% | 1.9334 |
| random_high | 32/111 | 7/32 | 17/57 | 29.82% | 1.9952 |
| motion_high | 36/111 | 8/32 | 22/57 | 38.60% | 1.7870 |
| 旧 full ICL（外部参考） | 31/111 | 6/32 | 21/57 | 36.84% | — |

所有配置在评分前固定，本轮没有“选出一个最优方法”再报test。尤其不能根据test
把 motion 说成 validation 选出的策略：**validation 的最高正确数是 uniform 9/32，
其 test 只有16/57；motion 是8/32，test22/57。** 若按最高validation正确数部署，
本次会选择 uniform。三种预算策略仅各一个seed/固定阈值，跨日样本也很少。

| Test57 配对（左侧方法 vs 右侧对照） | 改对 | 改错 | 净增正确数 |
|---|---:|---:|---:|
| high448 vs low112 | 15 | 5 | +10 |
| high448 vs mid224 | 7 | 3 | +4 |
| uniform_high vs low112 | 8 | 7 | +1 |
| random_high vs low112 | 8 | 6 | +2 |
| motion_high vs low112 | 12 | 5 | +7 |
| motion_high vs uniform_high | 9 | 3 | +6 |
| 新 mid224 vs 旧 full ICL | 1 | 1 | 0 |

新旧224虽同为21/57，实际有6题选择不同。高清在train/validation/test三个组均比
低清多答对，支持继续研究“细节有没有用”的前提；这还不能证明高分辨率改善了
持久记忆，且高清仍会把低清原本答对的题改错。motion 在test保留了部分高清收益，
但其validation不优于uniform，未达到完整提案的稳健策略收益标准。
同轮均匀中清 mid224 是必要的强对照：motion 比它仅多答对1题，却使用2880对2205
视觉token（多30.61%）；uniform/random的token更多而准确率更低。因此本轮不能
宣布自适应分辨率已经优于直接采用统一中清的整体预算取舍。

正式六组实验在 GPU4/5 各两个worker运行，墙钟 **532.233秒**。200题均值：
共享源视频解码6.524秒/题（p95 8.335）；共同CPU缩略图/选择0.064秒/题；
全部六配置实验10.236秒/题（p95 12.102）。并发逐题时间不能累加当墙钟/GPU-hours。

| 配置 | 预处理+前向均值/秒 | 同区间p95/秒 | 其中模型前向均值/秒 |
|---|---:|---:|---:|
| low112 | 0.220 | 0.467 | 0.134 |
| mid224 | 0.466 | 0.660 | 0.230 |
| high448 | 1.205 | 1.563 | 0.814 |
| uniform_high | 0.437 | 0.634 | 0.271 |
| random_high | 0.427 | 0.623 | 0.270 |
| motion_high | 0.433 | 0.773 | 0.276 |

这些区间不含共享解码，且在同卡两个worker并发下测量；没有单独端到端部署性能
测量，不能把视觉token减少67.35%解释为总时间或能耗减少67.35%。模型输入总token
范围为low1437–1476、mid2922–2961、high9537–9576、budget3597–3636。
单worker PyTorch allocated高水位最大5,855,858,688 bytes（约5.454GiB），不是各
配置独立重置后的显存占用，不包含全部CUDA开销和其他进程。

## 查看具体回答

以下按测试ID排序展示可解释的改对和改错例子；是已评分的事后案例分析，不是额外
模型选择。共同问题是“结合此前30分钟与最近30秒，预测接下来5秒的具体动作”。
表内文本为模型实际选中字母对应的公开选项，正确项来自银标签。

| 题目 | 银标签答案 | low112 | high448 | uniform_high | motion_high |
|---|---|---|---|---|---|
| egolife_023 | E：我接过了这个床单 | D：我整理了一下枕头 | E：我接过了这个床单 | D：我整理了一下枕头 | E：我接过了这个床单 |
| egolife_024 | F：我打开刚才的黑色软件 | F：我打开刚才的黑色软件 | B：我摘下头上的耳机 | B：我摘下头上的耳机 | B：我摘下头上的耳机 |
| egolife_093 | D：我喝了一口矿泉水 | C：我调整了一下耳机 | D：我喝了一口矿泉水 | D：我喝了一口矿泉水 | C：我调整了一下耳机 |
| egolife_096 | B：我拿出了茉莉花茶 | D：我拿出一个杯子 | D：我拿出一个杯子 | B：我拿出了茉莉花茶 | D：我拿出一个杯子 |

023展示高清与某种选择能改变动作判断，024展示增加细节不保证更正确；093说明
预算策略可能遗漏全高清/固定策略所保留的有用证据，096显示全高清也并非单调上界。
这些只是答案层面的观察，未逐帧人工确认视觉证据和真实后续动作，因此不声称已经
定位具体物体、帧或模型内部推理原因。

## 补充归因：固定选择区间最后一段

主六组评分后检查时刻表发现：motion 的800次高清选择只有330次达到运动阈值；
其余来自区间末尾兜底。后两个区间分别有173/200、181/200题选择最后一段。
因此“motion优于区间起点uniform”可能来自更近期的画面，不能先归因为运动信息。
为检验此解释，新增独立的 `deadline_high`：总是选择索引3、7、11、14，高清预算
同样24帧、2880视觉token。该配置明确是**事后探索性归因对照**，不包装为主六组
预先注册内容；全部200题预测再次冻结后才评分，原始六组产物保持不变。
该补充全部200题已完成，原始448源帧像素SHA逐题与主实验一致，单独冻结后评分：

| 事后固定最后一段 | Train111 | Validation32 | Test57 | Test准确率 |
|---|---:|---:|---:|---:|
| deadline_high | 36/111 | 10/32 | 24/57 | 42.11% |

Test57相对motion改对3题/改错1题；相对uniform改对10题/改错2题；相对全高清
改对2题/改错3题，少答对1题。其视觉token仍为2880（全高清的32.65%）。补充运行
墙钟370.181秒，预处理+前向均值0.419秒/题、p95为0.491秒，均不含解码。

**该控制不支持运动信息的独立收益。** 在本轮数据上，更简单的固定区间末尾策略
已达到或超过motion，提示近期证据的预算安排值得先研究。它并不证明所有收益都
来自近期性，也不能将这个事后新增配置视为原始六配置的validation赢家；需要在
未见录制日上重新固定协议检验。离完整Idea5还缺低清+记忆输入预测四分支长期收益、
同反馈训练的直接gain基线，以及TTT/显式bank/SSM同选择时刻表的比较。

补充逐题结果放在 `data/bench_v0_idea05_v1/supplement/deadline/per_case.jsonl`；
其 `scores.jsonl`、协议、预测冻结凭据、运行日志和结果由 `supplement_manifest.json`
校验，绑定原始核心manifest的SHA，核心六组归档没有被改写。

## 复现与产物

基础源码提交 `8e1d524deabb7ed5eba5427833f633629ee0e2dc`，新增源码、已有冻结
模块及本地模型的 SHA 均保存在嵌套协议中。模型为
`/data/quzitsix/models/Qwen3-VL-2B-Instruct`，没有新增模型训练或付费API调用。
Python使用conda `meowbench`。主干BF16，SDPA，启用原基线的线性patch等价执行。
旧冻结模块没有修改。

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python scripts/run_idea5_acquisition.py \
  --prepare --protocol runs/forecast_bench_v0/idea05_reproduce/protocol.json
# 启动前检查所指定GPU的当前任务；每worker要求至少8GiB空闲
conda run --no-capture-output -n meowbench python scripts/launch_idea5_acquisition.py \
  --protocol runs/forecast_bench_v0/idea05_reproduce/protocol.json \
  --output-dir runs/forecast_bench_v0/idea05_reproduce \
  --devices cuda:4 cuda:4 cuda:5 cuda:5 --threads 2
conda run --no-capture-output -n meowbench python scripts/evaluate_idea5_acquisition.py \
  --stage freeze --run-dir runs/forecast_bench_v0/idea05_reproduce
conda run --no-capture-output -n meowbench python scripts/evaluate_idea5_acquisition.py \
  --stage score --run-dir runs/forecast_bench_v0/idea05_reproduce
conda run --no-capture-output -n meowbench python scripts/audit_idea5_inputs.py \
  --run-dir runs/forecast_bench_v0/idea05_reproduce
conda run --no-capture-output -n meowbench python scripts/evaluate_idea5_acquisition.py \
  --stage archive --run-dir runs/forecast_bench_v0/idea05_reproduce \
  --output-dir data/bench_v0_idea05_reproduce
conda run --no-capture-output -n meowbench python -m pytest -q \
  tests/test_idea5_acquisition.py tests/test_evaluate_idea5_acquisition.py \
  tests/test_idea5_deadline.py
```

事后归因补充独立运行（不会改动主六组的protocol或预测）：

```bash
conda run --no-capture-output -n meowbench python scripts/run_idea5_deadline.py \
  --stage launch --run-dir runs/forecast_bench_v0/idea05_reproduce
conda run --no-capture-output -n meowbench python scripts/run_idea5_deadline.py \
  --stage score --run-dir runs/forecast_bench_v0/idea05_reproduce
```

本轮新增17项测试通过，覆盖因果缩略图决策、精确预算、未来时间戳拒绝、冻结预测
无答案访问、篡改拒绝、分片完整性及原始像素一致性检查。

逐题 `scores.jsonl` 保存六个答案、logits/概率、实际grid、输入token、选择时刻、
分辨率、源帧及选定像素哈希、前向耗时；`predictions.jsonl` 为评分前冻结预测；
`per_case.jsonl` 包含选项原文、银标签答案、各配置实际选择的文本及正确性。
这些模型输出只有约束选项字母，未生成解释；文档分析不得编造模型推理过程。
