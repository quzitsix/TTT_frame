# Idea4：真实视频上的函数响应记忆预检

日期：2026-10-26。正式目录 `runs/forecast_bench_v0/idea04_response_v2/`。
本轮检验有限个函数响应能否代替 fast MLP 的参数状态，并将短程教师辅助诊断与
从零开始的完整视频前缀推理分开报告。

## 实现范围

使用已有 Idea1 residual MLP 作为教师：冻结 Qwen 视觉特征、随机输入矩阵/零输出
初始化，hidden128，四个 horizon（1/2/4/10 个半秒 step），学习率 0.05。
每题完整重放3660个合法可见帧，生成最后20次连续成熟更新的真实教师响应轨迹。
这是 **Idea1 教师的函数空间原型**，没有实现完整官方 Spatial-TTT 或原卡片的
端到端可学习探测点/视觉语言 adapter，不把原型结果当完整 Idea4 的上限。

教师重放与旧实验的200题逐题张量核对通过：teacher/recent 最大绝对误差均为
2.02656e-6，所有对应 MCQ 预测完全一致。差异来自批量计算的浮点舍入。
原冻结源码不修改；所有新增代码、模型、特征和划分身份均有 SHA256 记录。

探测点从 train111 的标准化视觉 key 中确定性 farthest-first 选取128个；前32/64/128
构成嵌套容量组。32维随机投影固定 seed4，不在测试集上拟合；探测点按 train 数据
选择后冻结，没有把测试视频历史作为动态 bank 存入状态。

学生存储 `S[h,j,:]≈f_W(P_j)`，初始响应为零。核与读出为：

```text
phi(x) = normalize(x @ random_projection)
c = log(1 + RMS(S))
tau(S) = exp(clamp(log_tau + tau_state*c, -4, 4))
K_S(x,y) = exp(-||phi(x)-phi(y)||² / tau(S))
r(S,q) = K_S(q,P) solve(K_S(P,P)+0.05I, S)
e_i = v_i - r(S,k_i)
S' = S + sigmoid(logit_eta + eta_state*c) * mean_i K_S(P,k_i)e_i
```

核矩阵为正定 RBF 加正 ridge，使用 Cholesky 求解。固定核模型学习两个标量；
状态相关模型额外学习 `tau_state/eta_state`，共四个标量。两个模型都沿用相同
冻结投影、probe、目标和训练步数。所有核与 residual 都使用写入前的 S。
八个空间 token 同时计算再平均，是卡片顺序 token 更新的显式近似；没有把它称为
精确 NTK 转换，也没有声称标量核可表达一般跨输出通道的非线性梯度。

## 因果边界、训练与对照

沿用 train111（第1–4天）、validation32（第5天）、test57（第6–7天），所有佩戴者
共同按天划分。每题独立 reset；每个更新只在对应目标帧真正到达后发生。没有读取
`future.mp4`、未来描述或测试答案。完整前缀包含1800秒历史+30秒近期，2fps、
每题3660帧、8×256冻结特征，与 Idea1 同输入，和90帧零样本 full ICL 不是公平
输入/监督预算比较。共享动作读出已经使用 train111 标签、validation 选择，属于
额外监督；本轮没有重新拟合该读出。

每个容量的 fixed/conditioned 学生各训练80步、batch4、Adam lr0.03、梯度裁剪1。
前40步从真实教师 probe 响应出发展开2次写入，后40步展开20次；损失只匹配 train
教师 probe 响应。不是持续 teacher forcing：每段内部状态由学生自身递推。
教师初始态、完整连续写入和目标全部来自同一合法视频前缀；没有更大模型/caption
教师。训练窗口只覆盖各题末尾10秒，不能假定覆盖所有长期状态分布。

validation 仅按20次更新后未参与响应损失的视觉 query 误差选择容量，固定与状态
相关模型分别选择；不读取 validation/test 动作答案选模型。测试 query 是最后
可见帧的空间 token，用于诊断读出；不是不可见未来帧，也没有反向修改学生。
训练集最后 token 可能属于 train probe 候选池，严格未见输入的判断以 val/test 为准。

| Arm | 含义 | 部署是否需要教师 |
|---|---|---|
| teacher | 完整 Idea1 residual MLP 重放 | 本身是参数更新教师 |
| recent_only | 冻结近期预测器，零残差/reset | 否 |
| fixed_full / conditioned_full | 从 S=0 读完整3660帧的前向响应学生 | 否 |
| fixed_warm20 / conditioned_warm20 | 从 cutoff 前20次更新时教师精确响应初始化，然后自己递推 | 是，仅作短程诊断 |
| fixed_copy / conditioned_copy | 保持同一 warmstart 响应，20次均不更新 | 是 |
| fixed_oracle / conditioned_oracle | 最终教师精确 probe 响应，经学生核读出 | 是，静态插值诊断 |

“oracle”只表示响应来自精确教师，不是数学上的误差下界；核插值有偏，别的响应
偶然产生更小的未见 query 误差并不矛盾。完整学生推理不调用教师，也不做在线梯度。
没有人工身份/回忆真值，因此本轮不报告 delayed-recall accuracy。

## 结果

**全部200题已完成，预测冻结后独立评分。函数响应学生保持了教师的选项预测，但
没有建立动作收益或速度优势；不进入完整 VLM 集成。** 固定核和状态相关核均由
validation 未见视觉 query 的20步误差选中 `m=128`，没有使用动作答案作此选择。

| 方法 | Validation32 | 探索性 test57 | Test 准确率 |
|---|---:|---:|---:|
| Idea1 residual 教师 | 14/32 | 24/57 | 42.11% |
| recent_only / 零残差 reset | 14/32 | 24/57 | 42.11% |
| fixed_full，从零处理完整前缀 | 14/32 | 24/57 | 42.11% |
| conditioned_full，从零处理完整前缀 | 14/32 | 24/57 | 42.11% |
| fixed/conditioned warm20（教师辅助） | 14/32 | 24/57 | 42.11% |
| fixed/conditioned copy（教师辅助，不更新） | 14/32 | 24/57 | 42.11% |
| fixed/conditioned oracle（教师辅助插值） | 14/32 | 24/57 | 42.11% |

所有 validation/test 题目的选项字母也完全一致，而非仅正确数相同；完整学生与教师
全部200题的预测相同。相对教师和 recent-only 的 test 配对得失均为0/0。因此不能
用42.11%声称学生保留了对任务有效的长期记忆：当前教师本身也没有超过零残差对照。
full ICL 的21/57仅作跨实验背景，不能忽略3660 vs90帧以及 train监督读出的差异。

下表均为20次更新后的**标准化残差空间** MSE，与 cutoff 后动作准确率不同：

| 容量 | Fixed val 未见 query | Conditioned val 未见 query |
|---|---:|---:|
| 32 probes | 0.00868940 | 0.00867762 |
| 64 probes | 0.00727366 | 0.00727278 |
| 128 probes | 0.00622046 | 0.00622255 |

| 选定128 probes 的诊断 | Validation | Test |
|---|---:|---:|
| fixed warm20：probe 状态误差 | 0.00003318 | 0.00003621 |
| fixed warm20：未见 query 误差 | 0.00622046 | 0.00681355 |
| conditioned warm20：未见 query 误差 | 0.00622255 | 0.00681778 |
| fixed copy：未见 query 误差 | 0.00653390 | 0.00723409 |
| exact probes + fixed 核：未见 query 误差 | 0.00647294 | 0.00710967 |
| 教师该 query 残差信号均方 | 0.01730605 | 0.01938896 |

fixed 模型在1/2/20次更新后的 val probe 状态 MSE 为
`7.09e-7 / 1.86e-6 / 3.32e-5`，未见 query MSE 为
`0.006534 / 0.006519 / 0.006220`。因此没有观察到20步外推误差爆炸，但“拟合 probe
很准”也不等于“任意 query 读得准”。即使给精确教师 probe，test 未见 query 仍有
约36.7%信号能量的误差；这个数是具体核/投影的插值诊断，不是函数响应表示的理论下界。
状态相关核在选定容量的 validation 短程 query 指标没有超过固定核。

从零处理完整前缀后，**原始特征空间**最终预测相对教师的 MSE：fixed 为
val0.00098926/test0.00101567，conditioned 为0.00097545/0.00099963，recent-only
为0.00329872/0.00364938。学生确实更接近教师的连续输出；状态相关核有小幅数值改善，
但它没有改变任何验证/测试动作答案，不能包装成 forecasting 收益。

结论限于这个单种子、小参数核、随机投影、固定 train probe 和末尾10秒训练窗口。
未见 query 插值、语义读出敏感性与全前缀状态覆盖仍是不同瓶颈。没有运行完整 GRU/
Mamba、无教师训练和人工 recall 主实验；本轮不够作架构优劣或等预算非劣性结论。
test57已复用、为银标签且同日窗口重叠，以上均为探索性比较。

## 成本与归档

教师 fast state 四 horizon 共1 MiB。响应状态为四个 `m×256` FP32：m32=128 KiB、
m64=256 KiB、m128=512 KiB。探测点、随机投影与训练标量属于固定模型参数；Cholesky
矩阵、系数和临时激活是额外工作空间。近期预测器及成熟目标 FIFO 都是共同开销。
本次 runner 为复现读取整段特征缓存，单题输入缓存约28.6 MiB，不能把它隐去后声称
整个程序只有512 KiB状态。学生的算法状态固定大小，缓存不是算法长期记忆。

完整教师轨迹约2.5GB，位于 git 忽略的 runs 中；归档保存其逐题 SHA、训练模型和
probes，以及预测、正确性、诊断和阶段审计。V1 只发生过一次序列化存储问题：
张量 view 保存了 batch 底层存储；V2 改为逐题 clone，重新冻结协议和重放。V1 大缓存
在 V2 核验通过后移除，V1 日志和协议保留；没有混用两个版本的训练/预测。

实际阶段成本：200题教师重放/轨迹写盘26.42秒；六模型训练24.57秒；完整学生预测、
所有 warm/copy/oracle 控制及全部容量诊断合计166.06秒。不同阶段包含的工作不同，
不能直接除出部署速度。分别使用 GPU2/GPU3，RTX4090D；只使用本地缓存，无付费API。

另对一个预先固定的 train 题20次真实成熟写入作同步单题模块计时，2次预热、5次
重复取均值的中位数，每次均为四 horizon、八 token 更新：

| 模块 | 毫秒/次写入 | 持久 fast 状态 |
|---|---:|---:|
| 手工更新的 Idea1 MLP | 0.145 | 1 MiB |
| fixed RBF response m128 | 1.434 | 512 KiB |
| conditioned RBF response m128 | 1.410 | 512 KiB |

学生状态减半，但本地未融合核求解实现约慢10倍，**没有速度收益**。固定核原型每次
重新求解 Gram，并未缓存其固定分解；缓存可以改善该对照，但不能由未运行优化宣称
已有加速。计时不含视频编码、近期预测器、query、教师训练；峰值内存包含共驻三个
模型及轨迹，不当作单模块独占显存。完整读出和端到端部署延迟尚未单独测量。

归档：[data/bench_v0_idea04_v1](../data/bench_v0_idea04_v1)，约2.42MB。
`per_case.jsonl/csv` 保存具体预测和正确性；`predictions.jsonl` 为评分前冻结分数；
`forecast_errors.jsonl` 保存逐题连续输出误差；`diagnostics.json` 保存各容量/划分的
1/2/20步指标；`models/` 和 `probes.pt` 可直接复用，manifest覆盖全部归档SHA。

## 复现

所有 Python 命令使用 conda `meowbench`。先检查 GPU 空闲情况，再指定可用设备。
以下输出目录须不存在；每次阶段审计不读取答案。

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python scripts/run_idea4_response.py \
  --stage prepare --output runs/forecast_bench_v0/idea04_reproduce --device cuda:2
conda run --no-capture-output -n meowbench python scripts/run_idea4_response.py \
  --stage teacher --output runs/forecast_bench_v0/idea04_reproduce --device cuda:2
conda run --no-capture-output -n meowbench python scripts/audit_idea4_response.py \
  --run-dir runs/forecast_bench_v0/idea04_reproduce --before fit
conda run --no-capture-output -n meowbench python scripts/run_idea4_response.py \
  --stage fit --output runs/forecast_bench_v0/idea04_reproduce --device cuda:2
conda run --no-capture-output -n meowbench python scripts/audit_idea4_response.py \
  --run-dir runs/forecast_bench_v0/idea04_reproduce --before forecast
conda run --no-capture-output -n meowbench python scripts/run_idea4_response.py \
  --stage forecast --output runs/forecast_bench_v0/idea04_reproduce --device cuda:3 --batch 32
conda run --no-capture-output -n meowbench python scripts/audit_idea4_response.py \
  --run-dir runs/forecast_bench_v0/idea04_reproduce --before score
conda run --no-capture-output -n meowbench python scripts/run_idea4_response.py \
  --stage score --output runs/forecast_bench_v0/idea04_reproduce
conda run --no-capture-output -n meowbench python scripts/benchmark_idea4_memory.py \
  --run-dir runs/forecast_bench_v0/idea04_reproduce --device cuda:3
conda run --no-capture-output -n meowbench python scripts/summarize_idea4_response.py \
  --run-dir runs/forecast_bench_v0/idea04_reproduce --output-dir data/bench_v0_idea04_reproduce
conda run --no-capture-output -n meowbench python -m pytest -q \
  tests/test_idea4_response_memory.py tests/test_run_idea4_response.py
```

测试覆盖与原教师更新等价、候选排列等变、PSD、状态条件作用、参数不变、active mask、
梯度有限、逐题张量存储紧凑和轨迹篡改拒绝。
本轮专用测试结果：**8 passed in 1.21s**。
