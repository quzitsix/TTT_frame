# Idea 02：真实视频 key 坐标搬运预检

日期：2026-10-26。已完成 **200 题、2400 对自然相邻帧**的冻结特征实验，
包括正确匹配、打乱对应和错误视频三个条件，共 7200 条逐帧对记录。
**预检没有通过：独立 audit 对应上的误差没有改善，全部 2400 对都拒绝搬运。**
因此本轮按 Idea 卡片的停止条件停止完整记忆集成，没有生成新的 forecasting 答案，
不能将这一结果填成动作准确率为零，也不能宣称完整训练版 Idea2 已被否定。

## 方向与实际实现

原方向是：视角改变时，先把 fast function 的输入坐标和查询坐标同步变换，再写新信息，
减少同一内容换视角后的重复覆盖。原始卡片要求先验证“真实 key 的变化能否被局部
低秩正交变换描述”；至少改善 15%，且错误对应不应得到相同收益。

本轮使用冻结 Qwen3-VL-2B 视觉塔的原生合并空间 token。每张图为 448×448，保留
14×14=196 个 token，而不是将旧的 8 个 pooled token 上采样。既有冻结模块不修改，
新增 `ttt_frame/idea2_transport_audit.py` 与 `scripts/run_idea2_transport.py`。

视觉输出经固定种子 0 的正交随机投影得到 256 维，再做逐 token LayerNorm。
仅用 train111 的 **522,144 个视觉 token**拟合 PCA16，解释方差为 **55.501%**。
投影后的每行归一化为单位向量，因此实验检验的是**低维方向变化**；原 key 的范数
变化被去掉，不能据此验证一般特征变化。该 PCA 空间也不是完整卡片要求离线学习的
可变换 key 空间，更不是已接入官方 Spatial 层的 live key。

## 因果协议和样本

沿用第 1–4 天 train111、第 5 天 validation32、第 6–7 天 test57 的整天划分，
不重新随机划分题目。所有问题独立读取媒体，不携带其他题的在线记忆。

每题固定抽 12 对图像，全部在 observation cutoff 前，时间不由动作注释、问题、
选项或答案选择：历史片段的 150、450、750、1050、1350、1650 秒，近期片段的
0、5、10、15、20、25 秒；每对第二帧为第一帧之后 0.5 秒。近期时间加 1800 秒
才是完整前缀时间。解码器只返回 PTS 不晚于请求时间的帧，保存实际时间和像素 SHA。

推理与分析均不读取答案、future 视频、私有动作注释或音频。当前 200 个问题有窗口
重叠，因此 2400 帧对不是 2400 个独立 episode；两天 test 只作探索性诊断，不给
IID 显著性结论。这个机制实验不使用题目银标签，也没有人工物体身份或相机运动标注。

匹配在原 256 维归一化特征上进行：双向最近邻，cosine 至少 0.7。它只是自动特征
对应，不能把同位置背景、重复纹理或同类物体当成已确认的同一物体。

每帧对按 ID 和帧对编号确定性随机分成 **fit / gate / audit** 三组，比例约
1/2、1/4、1/4。至少 32 个匹配，fit 至少 16、gate/audit 各至少 8，fit 的最小/
最大奇异值比至少 `1e-4`。全部 matched 帧对都满足这些条件。三组互不重叠：

1. 在 fit 对应上解正则化正交 Procrustes：
   `min_R ||X R.T - Y||² + lambda ||R-I||², R.T R=I`。
2. 在 gate 对应上判断该变换是否至少降低 15% RMSE，否则返回 identity。
3. 在从未参与拟合或接受决策的 audit 对应上报告误差。

`lambda ∈ {0, 0.01, 0.1, 1, 10}` 仅按 validation 的 gate pooled RMSE 选择，
平局取较小值；不使用 validation audit 或 test。最终选择 `lambda=10`。
合并 RMSE 先按对应数累积 MSE 再开平方，不将不同帧对 RMSE 简单平均。

错误对照：`permuted` 将原匹配目标做非零循环移位，保证没有保留原对应；
`wrong_video` 将旧帧与另一天固定训练题的对应时刻新帧做同样的双向最近邻。
validation/test 的 donor 都来自更早 train 日；第一天的 train 控制可用第二天训练
素材，只用于错误匹配诊断。错误视频也可能存在相似纹理，因此不视为已知“无对应”真值。

## 结果：fit 改善没有迁移到独立对应

下表是所选 `lambda=10` 的 matched 条件。正数代表 RMSE 下降，负数代表恶化。

| 划分 | 帧对 / eligible | 接受数 | Fit 改善 | Gate 改善 | 独立 audit 改善 |
|---|---:|---:|---:|---:|---:|
| Train111 | 1332 / 1332 | 0 | +3.066% | -0.077% | -0.108% |
| Validation32 | 384 / 384 | 0 | +3.156% | -0.017% | -0.075% |
| Test57 | 684 / 684 | 0 | +2.988% | -0.079% | -0.132% |

所有 matched 帧对都 eligible（覆盖率 100%），没有因匹配不足而排除困难样本后
报告高改善率；接受比例却为 **0/684 test 帧对**。以下为 test 独立 audit：

| 条件 | Eligible / 全部 | Identity RMSE | Candidate RMSE | 候选改善 | Gate 后改善 |
|---|---:|---:|---:|---:|---:|
| 原始自动匹配 | 684/684 | 0.089144 | 0.089262 | -0.132% | 0.000% |
| 打乱对应 | 684/684 | 0.349523 | 0.348798 | +0.207% | 0.000% |
| 错误视频最近邻 | 643/684 | 0.104854 | 0.104333 | +0.497% | 0.000% |

Gate 后改善为零来自全部回退 identity，并非学到一个有用的新变换。
错误条件的小幅正增益也没有达到 15% 门槛，且不支持真实同物体坐标变化的特殊收益。
历史和近期两组的 matched audit 分别恶化 0.131% 和 0.134%，方向一致。

更弱的正则也未通过 validation：

| Lambda | Validation gate RMSE | 相对 identity 的变化 |
|---:|---:|---:|
| 0 | 0.097300 | 恶化 8.155% |
| 0.01 | 0.097083 | 恶化 7.914% |
| 0.1 | 0.095854 | 恶化 6.549% |
| 1 | 0.092374 | 恶化 2.680% |
| 10 | 0.089978 | 恶化 0.017% |

这不是“正则太强所以没有尝试变换”的单一解释：无正则在未拟合的 validation 对应
上反而更差。测试前预先记录的 go 规则还要求 >=50% eligible、独立 audit 的候选与
实际门控结果都改善 >=15%、错误控制无正增益；本轮明确不满足。

## 具体帧对的效果

以下样例取 test 按 ID 排序第一对，以及全 test audit 改善最好和最差的各一对，
是结果分析，不用于选择参数。时间为对应片段内秒数。

| 选择方式 | 题目 / 帧对 | 对应数 | Fit 改善 | Gate 改善 | Audit 改善 | 决策 |
|---|---|---:|---:|---:|---:|---|
| 第一对 | egolife_022，history 150→150.5 | 121 | +1.916% | +0.025% | +1.682% | 拒绝 |
| Audit 最好 | egolife_126，recent 20→20.5 | 89 | +5.148% | +0.128% | +2.626% | 拒绝 |
| Audit 最差 | egolife_158，history 1650→1650.5 | 93 | +4.011% | +0.685% | -2.776% | 拒绝 |

即使事后挑 audit 最好的帧对也远低于预设 15%。本轮检查的是地址变化假设，未运行
动作选择模型，因此这里没有可展示的 Idea2 “具体动作回答”。其余已跑 forecasting
方法的具体选项对比由总报告列出，避免将特征匹配误差冒充动作预测成功率。

## 解释与下一步边界

在当前冻结视觉表示、自动匹配、0.5 秒间隔及 train PCA16 的设置中，fit 拟合降低
误差，但不能泛化到独立对应；最有效的选择接近“不搬运”。这不足以支持把全体历史
fast weights 一起变换，也没有理由启动昂贵的完整 Idea2 训练作为本轮默认后续。

可能解释包括自动匹配混入错误对应、非全局正交变化、投影损失、表示已具有部分视角
不变性，或变化主要来自物体状态而非相机。数据没有身份/相机运动真值，不能唯一归因。
如果未来重开此方向，应先审计明确的同物体真实视角变化及 revisit、训练可变换 key，
并与独立 canonicalizer / invariant key 比较；本轮没有覆盖这些训练后的假设。

## 资源、协议与复现

单题 24 张已观察图像，共 4800 张；每张 196 native token，共 940,800 个 token。
这不是和 forecasting 的 90 帧输入相同的预算，而是稀疏相邻帧机制诊断。
特征缓存为 **965,085,400 bytes**，位于项目盘的 ignored `runs/`，未写入接近满载
的数据盘。存量 fast memory 未修改，没有额外在线状态持久化。

提取使用 GPU0/1，每卡两个 worker；四 worker 全部正常退出。完整分片启动墙钟
294.505 秒，其中 199 题新提取、1 题复用相同冻结协议的 smoke 缓存。
200 题单题区间累积解码 1048.237 秒、编码 77.746 秒，总计 1125.983 秒；并发区间
之和不是墙钟或 GPU-hours。单题 smoke 包含在逐题累计时间内。

视觉模型 fingerprint（权重内容和配置 SHA256）为：
`317f465c9107c75f736f43c6ad6296a2928670b47d97ecaacd7a9337b8a51b43`，
四 worker 相同且与既有 Idea1 冻结 encoder 匹配。模型来自本地
`/data/quzitsix/models/Qwen3-VL-2B-Instruct`。Python 使用 conda `meowbench`；
主干 bfloat16，投影/匹配 FP32，PCA 协方差 FP64。种子与随机分割规则写入协议。
基础源码 Git 为 `8e1d524deabb7ed5eba5427833f633629ee0e2dc`，新旧依赖文件 SHA
保存在 `protocol.json`。没有使用付费 API 或新增依赖。

核心产物：[data/bench_v0_idea02_v1](../data/bench_v0_idea02_v1)。
`pairs.jsonl` 含 7200 条逐帧对原始误差和拒绝原因，`per_case.jsonl` 含 200 题聚合和
像素/时间证据，`selection.json` 含 validation 网格，`train_pca.pt` 为训练日投影，
`feature_manifest.json` 保存所有特征缓存 SHA，`manifest.json` 校验归档文件。
归档脚本核验全部 200 个缓存、分片无重叠、模型一致、源媒体 size/mtime 和采样因果性。
原始视频未做额外整文件 SHA；逐帧像素 SHA 和特征文件 SHA 是本轮可重核的内容证据。

使用新目录复现（先确认 GPU 可用）：

```bash
cd /home/quzitsix/TTT_frame
PYTHONPATH=. conda run --no-capture-output -n meowbench python scripts/run_idea2_transport.py \
  --stage prepare --output-dir runs/forecast_bench_v0/idea02_reproduce
PYTHONPATH=. conda run --no-capture-output -n meowbench python scripts/run_idea2_transport.py \
  --stage extract --output-dir runs/forecast_bench_v0/idea02_reproduce --device cuda:0
PYTHONPATH=. conda run --no-capture-output -n meowbench python scripts/run_idea2_transport.py \
  --stage analyze --output-dir runs/forecast_bench_v0/idea02_reproduce
```

本次四 worker 的精确命令见归档 `provenance/launch.json`。归档验证器要求本次四分片
完整性；若按上面的单 worker 命令重跑，分析结果可直接读取，不能伪造四分片凭据。

相关测试 **24 passed**（新 preflight 11 项与旧 key transport 13 项）：

```bash
conda run --no-capture-output -n meowbench python -m pytest -q \
  tests/test_idea2_transport_audit.py tests/test_key_transport.py
```
