# 工作报告 — TTT_frame

## 2026-09-11：生成式参数记忆接口与 MEOWBench 接入

本节记录新增实现与直接运行所得结果；下方原报告仍是 09-08 的 LaCT 实验历史。

**本轮交付。** 新增 `VideoTTTMemory`，接受一个或多个按时间排序的视频路径；
顺序 PyAV 解码、按 chunk 抽帧；冻结 VLM 生成观察和临时 QA；只更新语言注意力
q_proj/v_proj 的 LoRA 参数。封存后清理优化器、梯度及生成位置缓存，支持在新进程
加载 `adapter.safetensors` 回答选择、数值或开放问题。读取阶段不传入视频、caption
或历史 QA。实现独立于 benchmark，不改原 LaCT 更新公式。

这是 **LoRA 测试时自蒸馏基线**，有显式的临时文字训练目标和隐式的查询时参数
存储；不是原生连续视觉表征记忆，也不是 LaCT/Spatial-TTT 的生成式复现。
目前没有训练跨场景人物识别、关系保持 gate 或长期抗遗忘策略。

**接入方式。** `meowbench.adapters.ttt_lact --backend lora` 复用既有协议；
bench 侧新增 `scripts/run_ttt_pilot.py`，在同一组 suite items/sessions 上运行
`blind / memory / base-read`，记录预测、撤销状态、逐环境计数、参数大小、耗时与
成对分数差。`base-read` 仍执行训练，但回答时关闭 LoRA。训练核心不读取 suite
标注；gold 仅由 benchmark scorer 消费。其他对话进行中的真实数据适配文件
没有被搬入 TTT 仓库。

### 本机直接验证

硬件：RTX 4060 Laptop；环境：conda `claude`，torch 2.11.0+cu126、transformers
4.57.6、PEFT 0.14.0、PyAV 17.0.0、Accelerate 1.13.0。模型来自本机完整缓存，
未下载新权重。本轮没有访问服务器或完整真实第一人称 suite。

| 检查 | 结果 | 能说明什么 |
|---|---|---|
| 原 LaCT + 新 VideoQA 单元测试首轮 | 24 passed | 原机制回归、assistant 标签掩码、LoRA 更新、基础权重不变、reset、封存、checkpoint、一段视频解码 |
| bench 桥接/协议/staging/runner | 55 passed, 1 skipped | 桥接遵循生命周期及视频撤销协议；跳过项是已有平台条件测试 |
| SmolVLM2-256M 实际模型摄入 | 1 chunk、2 个更新步骤、caption fallback | 图像到文字再到 LoRA 的真实路径可执行；不能推断感知质量 |
| Qwen3-VL-2B 实际模型摄入 | 1 chunk、1 抽样帧、2 条临时 QA、2 更新步骤 | 已执行真实视觉 teacher 与文本梯度训练 |
| Qwen 单独进程加载参数回答 | 加载与生成成功，但出现无视频拒答 | 证明持久化接口可用；**没有证明有效回忆** |

随后新增了多 session 累积写入回归；最后一次 VideoQA 专项验证为 **10 passed**，
产物为 `TTT_frame/runs/videoqa_final.xml`。原 LaCT 的 15 项已在前述联合验证中通过。

Qwen 这一配置：bf16 基础模型、FP32 LoRA，rank=4、alpha=8、lr=0.0002、
steps=2、max_side=224、teacher_new_tokens=256。临时训练的首次/末次步骤损失
为 **5.4137 → 5.0884**；保存 **802,816 个 LoRA 参数，3,211,264 bytes**。
计数中的 1 帧是因为 demo 只有约 2 秒，而抽样 chunk 为 30 秒，并非解码失败。
默认真实任务使用 rank=16、steps=12；这个小配置仅用于本机链路检查。

训练损失下降后，独立进程询问 “What objects were visible in the video?”，仍生成
“I don't have access to any video content” 一类回答；输出受 48-token smoke 预算
限制。应把它当作需要改进的行为证据，不能把损失下降当作记忆已建立。

### MEOWBench 的三组端到端 smoke

使用现成 `fixtures/demo`，每组同样的前 2 道合成问题；完整执行 handshake、
env_begin、ingest、ingest_end、视频撤销、query、env_end。该 fixture 的问题
和答案属于 harness 测试，不是有语义效度的真实视频能力评测。

| 模式 | 协议成功 | 实际抽帧 | 更新步骤 | 回答 | fixture 分数 |
|---|---:|---:|---:|---|---:|
| blind | 2/2 | 0 | 0 | 两题均 E/信息不足 | 0/2 |
| memory | 2/2 | 1 | 2 | 两题均 E/信息不足 | 0/2 |
| base-read | 2/2 | 1 | 2 | 两题均 E/信息不足 | 0/2 |

memory 的 `enforcement=revoked`，`revocation_contested=false`；源 fixture 视频
仍存在且大小为 7,696 bytes。Windows 本次 `fd_audit_available=false`，因此不把
“未发现打开句柄”表述为完整操作系统句柄审计已通过。单元测试另覆盖了删除输入
后 checkpoint 恢复及解码器关闭文件的行为。

本次 memory 摄入计时约 20.43 秒，两题端到端查询分别约 376/366 ms；base-read
摄入约 58.08 秒，耗时变化很大。运行时有其他任务，且没有统一预热，**这些仅是
运行日志，不构成吞吐或速度优劣结论**。新版额外提供峰值 CUDA allocated 字节
计数；首轮 smoke 发生在该计数加入前，不能补写一个未测得的峰值。

本地原始产物（被 `.gitignore` 排除，未上传视频/权重）：

- `TTT_frame/runs/videoqa_qwen_smoke/memory/`：权重与配置、摄入计数。
- `TTT_frame/runs/videoqa_smoke/memory/`：SmolVLM 参数快照。
- `TTT_frame/runs/videoqa_validation.xml`：TTT 测试报告。
- `meowbench/runs/ttt_bridge_validation.xml`：bridge/协议回归报告。
- `meowbench/runs/ttt_qwen_smoke_20260911/`：每组 `predictions.jsonl`、
  `summary.json`、`report.json`、`ttt_metrics.jsonl`，以及 `comparison.json`。

### 运行中修正的问题与下一步

1. `local_files_only=True` 在 processor 子组件路径上仍出现元数据请求。现在先将
   repo ID 解析为已缓存 snapshot 目录，再加载，且不让 PEFT 保存逻辑探测远程
   embedding 配置。离线验证时可额外设 `HF_HUB_OFFLINE=1`。
2. 默认 pytest 临时目录受 Windows 沙箱权限限制；使用仓库 runs/ 下专属临时
   目录重新验证。中文路径经过子进程默认编码会损坏；pilot 显式设 UTF-8，
   同样环境重跑协议测试后通过。未通过回滚 staging 的方式掩盖问题。
3. 增加 parameter-only query 的状态检查、失败摄入隔离、跨家庭恢复初始权重、
   gradient-checkpointing 梯度检查，以及 Qwen 临时位置状态清理。
4. 下一步提供已准备的真实 suite 路径，在服务器用 README 中的 pilot 命令
   跑 8–30 道自动评分问题。先检查 teacher 是否观测到证据，再比较冻结模型、
   参数记忆和笔记 baseline；不要用测试答案训练 LoRA。
5. 搬家后人物–物品关系保持仍未被检验。应分别测感知正确率、即时参数读出、
   干扰后保持、场景切换和人物重识别，避免把前一环节失败当成后者结论。

---

**期间** 2026-09-08（单日）· **仓库** github.com/quzitsix/TTT_frame
**姊妹仓库** `meowbench`（quzitsix/test_1）有独立的 `docs/WORK_REPORT.md`，覆盖
benchmark harness 侧。本报告只讲 TTT 侧。

**证据标注**：本报告每个数字标了来源。
✅ = 我自己在本机跑出来的　🔬 = 后台并行分析产出、我独立复核过
📋 = 后台分析产出、我未逐条复核（当线索读，不当结论读）

---

## 一句话总结

**做出了一个可用的 TTT 快速权重记忆（并修了上游官方代码的一个真 bug），但最重要的
产出是一个否定结论：原本想问的「搬家后还记得物品-人物关系吗」这个问题，用现有
fixture 问不出来** —— 曾报出的效应，其符号完全由一个任意的预处理选择决定，已撤回。

真正学到的东西是：**在这套设定里，表征几何比记忆机制重要得多。**

---

## 一、交付物

| | |
|---|---|
| 代码 | 1133 行（记忆 373 / 实验 472 / 测试 288） |
| 测试 | ✅ 15 个，全过（1 个无 CLIP 缓存时跳过） |
| 提交 | 5 个，历史干净（无 benchmark 代码） |
| 依赖 | 仅 torch（实验额外需 transformers + pillow） |
| 可移植性 | ✅ 纯 PyTorch，无 Triton/CUDA 扩展；Windows 笔记本与 Linux 8×4090 同一份代码 |

```
ttt_frame/lact.py                     记忆本体，write/read 显式分离
ttt_frame/_lact_upstream.py           LaCT 官方原版（保留可 diff）
scripts/exp_scene_change.py           主实验，--full 跑 2×2
scripts/exp_centering_sensitivity.py  证明 gap 符号由预处理决定
scripts/fetch_weights.sh              离线环境取权重
tests/test_lact.py                    15 测试，含上游 bug 回归
docs/HANDOVER.md                      交接文档
docs/TTT_PROTOTYPE.md                 早期诊断日志（部分已过时）
```

---

## 二、做了什么

### 1. 移植 LaCT 并修了上游 bug

来源：`github.com/a1600012888/LaCT`（*Test-Time Training Done Right*,
arXiv 2505.23884，MIT，517 stars）。选它的 `minimal_implementations/` 因为它是
纯 PyTorch、只依赖 einops，Windows 上能直接跑。

**发现的 bug**：`bidirectional_lact_layer.py` 把正交化后的**梯度**赋给了**权重**：

```python
if use_muon:
    w0 = zeropower_via_newtonschulz5(dw0)   # 应该是 dw0 = ...
w0 = w0 + dw0
```

先前状态被销毁 → 那一层根本无法携带记忆。而 `use_muon=True` 是**默认值**。

✅ **验证方法**：把 learning rate 设为 0，此时更新在数学上必须是恒等变换：

| | lr=0 是否 no-op | 与 `f_w(q)` 偏差 |
|---|---|---|
| 上游 `use_muon=False` | ✅ | 0.000 |
| 上游 `use_muon=True`（默认） | ❌ | **20.416** |
| 修正后 | ✅ | 0.0003 |

同仓库其他所有实现（causal 层、`lact_ar_video`、Triton kernel）都写 `dw0 = ...`，
确认是那一个文件的笔误。已固化为 `test_zero_lr_is_a_noop`。

**为什么这个 bug 对本项目致命**：整个实验问的就是"快速权重留不留得住东西"，
上游写法**保证答案是「留不住」**，且与科学无关。

### 2. 加了定向关联 API

`write(tokens, values=...)`。identity 投影下 k 和 v 否则是同一个向量，记忆只能返回
你查询它的东西 —— 对绑定实验（object → person）毫无用处。

### 3. 三个实验脚本

- **`exp_scene_change.py`** — 主实验：在 A 家写入绑定，插入 N 次中性填充，再查询。
  `--full` 跑 2×2（filler 来源 × query 取景），带置信区间和塌缩检测。
- **`exp_centering_sensitivity.py`** — 专门证明 SAME/MOVED 的 gap 符号由中心化参考集
  决定。**这个脚本存在的目的就是防止那个 gap 再被当成发现。**
- **`fetch_weights.sh`** — 服务器无外网时探测镜像并验证离线可加载。

---

## 三、做了哪些实验

### 实验 0：对照组（必须先过，否则后面全是噪声）

✅ 三个对照全过：
- **纯关联召回**：写 8 个 (k,v) 再读回 → **1.000**，跨所有 lr（0.01–1.0）和 head 配置
- **置换对照**：写 object i → person perm[i]，记忆返回 **perm 而非 identity**
  → 读出确实在用记忆
- **无写入 baseline**：失败（随机水平）→ 成功需要写入

### 实验 1：绑定能不能跨越视频撤销

✅ 用文本 embedding 绕过感知：写入三条归属事实，撤销，查询 → **3/3 正确**。
write → revoke → read 这条链是通的。已固化为测试。

### 实验 2：搬家场景（主实验）——**结论已撤回**

在 A 家写 4 条 object→person 绑定，插入 N 次中性填充（空房间），再查询。
两个 arm：填充来自同一家（SAME）或另一家（MOVED）。

✅ 跨机器复现：4060 与 4090 数字误差内一致（8 fillers: 0.933/0.967 vs 0.942/0.983）。

✅ **但 2×2 拆解显示它是混淆的**（n=100，16 fillers）：

| | query in A | query in B |
|---|---|---|
| **filler from A** | 0.662 ± 0.033 | 0.362 ± 0.025 |
| **filler from B** | 0.845 ± 0.027 | 0.812 ± 0.031 |

filler 主效应 **+0.316**，query 主效应 **−0.166**，方向相反。
原 SAME = (A,A) = 0.662，MOVED = (B,B) = 0.812 —— 报出的 "+0.15" 是两者部分抵消
的残差。

✅ **而且 nf=8 时交互 +0.513 比两个主效应都大**：线索漂移在 filler B 下**完全免费**
（0.975→0.975），在 filler A 下**要命**（0.925→0.412）。平均成一个"漂移代价"没有意义。

### 实验 3：gap 的符号由中心化参考集决定 —— **这是撤回的根据**

✅ 中心化需要一个参考集，问题本身没规定用哪个。四个都站得住，各 40 trials：

| 参考集 | filler_A/obj | filler_B/obj | gap@8 | gap@16 |
|---|---|---|---|---|
| narrow（只含在用的） | 0.262 | 0.456 | **+0.037** | **−0.038** |
| broad（跨两家 36 物体） | 0.145 | 0.316 | −0.019 | −0.181 |
| home-A only（在线可算） | 0.529 | 0.157 | −0.062 | −0.281 |
| 不中心化 | 0.861 | 0.850 | 0.000 | 0.000 |

**符号在参考集之间变号，narrow 内部还随 filler 数变号。**

✅ 一个独立复分析用 narrow 参考报出 SAME **0.775** vs MOVED **0.631**（gap +0.144）
—— 与我的方向相反。我跑了它的代码，**逐位复现了它的数字**。两边计算都没错，
**算的是不同的东西**。真正驱动结果的是中间两列：中心化改变填充与存储 key 的重叠度，
保持率跟着重叠度走。home-A 参考下重叠排序翻转，结果也翻转。

**结论：现有 fixture 回答不了搬家问题。刺激几何主导一切。**

### 实验 4：为什么"中性"填充会擦除绑定 —— 机制归因

🔬 **决定性对照**（后台分析发现，我独立复核）：随机单位向量填充造成**同样的权重旋转**
（`cos(W_bind, W_8)` = 0.857 vs 房间图 0.880），却 16 次写入后仍保持 **1.000**，
房间图 2 次就崩。**旋转量不是机制。**

🔬 真正的机制是 **key 混叠**：✅ 实测原始 CLIP `|mu| = 0.969`、`mean(cos²(x,mu))
= 94.0%` —— 每个 embedding 94% 的能量在同一个方向上。所以在快速权重 key 空间里
✅ `cos(k_obj, k_filler) = 0.7105`（分析报 0.7103，一致），一张空房间图几乎是
每个物体 key 的副本。

🔬 剂量-响应在 key 空间余弦上单调，真实房间图正落在曲线上，无残留"图像性"。

✅ **中心化的效果**：

| 填充写入数 | 0 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| 原始 CLIP | 1.000 | 0.750 | 0.250 | 0.250 | 0.250 | 0.250 |
| 均值中心化 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.750 |

保持窗口从 ~1 次拉到 ~8 次。**这不是贡献，是 CLIP 检索的标准做法**（锥形效应
2203.02053/2405.18570，中心化补救 2112.12777，2022 年就有文献）。

### 实验 5：损伤只在填充充当 KEY 时发生

✅ 我独立复核过这一组：

| 条件 | f=0 | f=1 | f=2 | f=4 | f=8 |
|---|---|---|---|---|---|
| `k=filler, v=filler`（原样） | 1.000 | 0.750 | 0.250 | 0.250 | 0.250 |
| `k=filler, v=zeros` | 1.000 | 1.000 | 1.000 | **1.000** | **1.000** |
| 🔬 `k=random, v=filler` | 1.000 | 1.000 | 1.000 | 0.958 | 0.958 |

填充作为 value 携带什么是次要的。

### 实验 6：绑定没被擦除 —— 是读出塌缩了

✅ 我独立复核：读出向量之间的余弦随填充上升（0.798 → 0.956 → 0.974），
而**读出前减去均值就把准确率从 0.250 救回 1.000**（nf=2 和 nf=4 都是）。

📋 分析进一步报告：匈牙利匹配到 4 次写入仍 1.000、8 次 0.750；
`cos(read, 正确人物)` 几乎不动（0.536 → 0.576）。

**信息活过了现成读出评为随机的那些写入。**

### 实验 7：一个格子在"假装随机"

✅ `0.250 ± 0.000`，30 seed 零方差。看实际输出：`predictions=[3,3,3,3]` ——
记忆饱和，四个物体全返回同一个人。4 选 1 蒙对 1 个 = 0.250，**数值上与随机
完全一致**。准确率单独看**区分不出塌缩和瞎猜**。

已修：`run_trial` 同时返回 distinct 预测数，`--full` 标记并单列塌缩格子。
✅ 全 12 格审计：只有 (filler A, query B) 塌，nf=8 就开始（d=1.65），nf=32 彻底（d=1.00）。

### 实验 8：Muon 让学习率失效

✅ 我自己验的：Newton-Schulz 丢弃更新幅度，所以 `|step|` 在 lr 从 1e-4 到 100 之间
**恒定在 ~3.63**；关掉 muon 则 0.03 → 15.5。**开着 muon 调学习率等于什么都没调。**

📋 反方向的意外：`use_muon=False` 时更新小 ~470 倍（`|dW|/|W|` 0.00057 vs 0.269），
**零填充时就已经 0.250** —— 什么都没写进去。Muon 是让写入发生的前提。

已固化为 `test_muon_makes_the_learning_rate_nearly_inert`（声明近似不变，
bf16 + 5 次有限迭代留 ~2% 残差）。

### 实验 9：超参扫描 —— 全是平的

📋 head_dim 16→512、inter_multi 1→4、lr 三个数量级：都不改变保持窗口。
**没有任何超参能买到保持力，只有表征能。**

---

## 四、修掉的缺陷

**六个属于会伪造结果而不是报错的那一类** —— 这类最贵，因为它给你一个看起来像
发现的数字。

| # | 缺陷 | 后果 |
|---|---|---|
| ① | 上游 Muon 赋值错误 | 快速权重被梯度覆盖，**默认配置下无法携带记忆** |
| ② | 读出穿过随机 `W_o` | 读出落在任意旋转空间，三个 arm 报出**逐位相同**结果 |
| ③ | 选项 E 靠句子长度取胜 | CLIP 对**任何**问题都选 E（含"天空什么颜色"），14 题全答 E |
| ④ | 塌缩格子伪装成随机 | `0.250 ± 0.000` 被当成无害地板效应 |
| ⑤ | SAME/MOVED 一次改两件事 | 两个反向效应混合，报出的 gap 是抵消残差 |
| ⑥ | gap 符号由预处理决定 | 整个主结论作废 |
| ⑦ | 离线时测试卡 331 秒 FAILED | `except: skip` 等不到（重试 5 次指数退避超过 timeout） |
| ⑧ | Windows `torch.compile` 炸 | inductor 去调 MSVC，`cl is not found` |
| ⑨ | `git add -A` 污染仓库 | TTT 代码扫进 benchmark 仓库（两次），最终导致仓库分离 |

---

## 五、五次「合理推断」全错

这是本期最有价值的元教训。**这个系统反直觉，不要推理，要测。**

| 我的推断 | 实测 |
|---|---|
| 同屋填充与 key 重叠更多所以覆盖更狠 | **反了** —— 重叠更多的伤害更小 |
| 换家破坏线索这个 confound 已排除 | **没排除** —— 我只验证了最近邻，不足以证明检索不受影响 |
| 线索漂移是恒定代价 | **不是** —— filler B 下免费，filler A 下要命 |
| 填充自相干性是元凶 | **不是** —— 0.943 vs 0.919 差异撑不起 +0.316 |
| 旋转量决定损伤 | **不是** —— 随机填充同样旋转、零损伤 |

---

## 六、可发表性的诚实判断

📋 后台分析的结论，我认同：

**「LaCT 2 次写入就崩」不可发表。** 机制是教科书内容（2505.19488 有闭式解
`SNR⁻¹ ~ N/d_k`；2102.11174 在 2021 年就推断出来并发明 delta rule 来修；
2605.05066 证了信息论界），而触发因素是没中心化的 CLIP —— 锥形效应和中心化补救
从 2022 年就有文献。这么发会犯和 **SR-TTT (2603.06642) 当初必须撤稿**同一类的错误。

「快速权重会随流遗忘」也已被至少四篇论文陈述：LongVU-TTT (2608.25729) 原话称其
快速权重"表现为 temporal aggregation state 而非可靠的 long-horizon episodic
memory"、Mem3R (2604.07279)、Elastic TTT (2604.07350)、Gated DeltaNet (2412.06464)。

**可能有价值的**：`base_lr` 在 muon 下失效（检索未见人报过）、LaCT 特定的保持窗口
量化（原文没给）。但都需要在真实设定下重做。

---

## 七、下一步

### 优先级 1：两个正交的机制修复

**(a) 内层损失换成 delta/回归形式。** 📋 Gated DeltaNet (2412.06464 §3.1) 明确点名
负点积是弱的。`L = -f_W(k)ᵀv` 无界无不动点 —— 存好了还继续往 v 推，所以重复近重复
写入单调放大一个方向，正是实测到的塌缩。回归梯度 `(f_W(k) - v)` **自限**。
它攻**读出膨胀**那一半，中心化攻**key 几何**那一半，两者正交。

**(b) Elastic TTT 的 Fisher 锚定**（2604.07350）：唯一专门针对 LaCT 这个失效模式
的已发表方法。📋 分析测到 2 fillers 时 +0.688（配对 t=32.0）。从最便宜的 MAS
（`F = EMA(|Δθ|)`）开始 —— 该论文称 importance estimator 几乎不影响结果。

**建议先做 (a)** —— 机制性修复而非调参。

### 优先级 2：换几何可控的刺激设计

直接在 embedding 空间构造受控向量（指定与 key 子空间夹角），让"场景变化"和
"表征重叠"可以独立扫。现有 fixture 再加 trial 只是把几何巧合测得更精确。

### 不要做

- 不要在现有 fixture 上加 trial 确认 SAME/MOVED（符号由预处理决定）
- 不要开着 muon 调 `base_lr`（等于没调）
- 不要用未中心化 embedding 测任何东西
- 不要改 `_lact_upstream.py`（价值就在原封不动）

---

## 八、环境备注

- **服务器无外网访问 huggingface.co**。`hf-mirror.com` 可用（实测 307）。
  先跑 `bash scripts/fetch_weights.sh`。
- **Linux 上可开 `MEOW_TTT_COMPILE=1`**；Windows 上因无 MSVC 必须关。
- **两个仓库依赖相容**，可装同一 conda 环境。交集 `transformers>=4.57,<5`，
  下界是硬的（`qwen3_vl` 4.57.0 才进 auto mapping）。
- `pytest` 在 TTT_frame 目录跑 15 个测试；benchmark 那 329 个要在 meowbench 目录跑。
