# Predictive-TTT 实验总报告：A、B、C

本文汇总当前仓库中预测型 TTT 的三个实验。A 是 FHO-STA 短期预测边界；B 是 HomeSentinel 与 SVCBench 的流式状态维护；C 是受控搬移/重排机制探针。所有结果都保留了逐条 JSONL，报告中的小数可以由对应脚本重新生成。

## 结论先行

实验 A 目前没有可报告的 Ego4D 数值：服务器没有可用的 FHO-STA 视频和标注，不能用不存在的数据填结果。

实验 B 的 HomeSentinel 快速结果显示，显式视觉 ICL 的 lexical token-F1 高于已有 Spatial-TTT 子集，但两者视觉帧预算不同，不能据此判断 ICL 优于 TTT。完整视频采样的 ICL 为 0.2001，已有 Spatial-TTT 为 0.1660，无记忆读出为 0.1730；这更像是“当前 TTT readout 没有体现稳定增益”的信号。

SVCBench 只下载了一个约 14.8 MB 的视频和标注，没有下载约 80 GB 的完整数据集。为了完成一次可复现的真实模型 smoke，保留了 47 个原始 query points 中每四个取一个的 13 点轨迹。ICL 与 Spatial-TTT 的点准确率都为 0.1250；Spatial-TTT 的 MoC 较高（0.8750），但 UDA 较低（0.0625），说明预测轨迹更单调，却没有更好地跟随真实计数变化。

实验 C 的 symbolic probe 不是视觉 VLM 结果。它把位置、owner、搬移者、状态和下一步动作写入固定维度关联记忆，12 个 episode、60 个 query、3 次 trial 共 2160 行。LaCT 在 relocated 条件为 0.8833，在 same-scene control 为 0.5611；显式表为 1.0/1.0。这个结果说明关联记忆机制能保留绑定，但不证明视觉编码或真实家庭场景已经有效。PNG 卡片视觉 follow-up 的两集结果也很弱：ICL token-F1 0.0444，Spatial-TTT 0.0196。

## A：FHO-STA 短期预测边界

当前状态是 **blocked by data availability**，不是模型失败。`docs/FHO_STA_EXPERIMENT_A_RESULTS.md` 保留了完整配置与复现命令，但本机没有 Ego4D FHO-STA 的可读 annotation/video，因此没有生成 noun/verb accuracy，也没有填入估计数字。等 Ego4D 数据可用后，应固定相同观测边界和帧预算，再比较 blind、ICL-4f/8f 与 Spatial-TTT-4s8f/8s4f，并按 horizon 分层。

## B：长历史与结构化 counting

### B1 HomeSentinel

评测使用 `/data/HomeSentinel/asuka` 的现有视频和 benchmark queries，选取 `video_cutoff_idx <= 10` 的 15 个 event/home 查询、10 个视频。ICL runner 只取每个观察视频 2 帧，历史上限为 32 帧；quick 版只在每个视频的前 60 秒取样，full-video 版在整个已观察视频上均匀取样。两者都使用 Qwen3-VL-2B native video processor、max-side 448、cuda:2。

同一批 query 在已有 Spatial-TTT run6 中筛出对应的 `memory` 与 `without_memory` 行：run6 使用 60 s/4 frames 的视频块，在完整 asuka stream 上写入 Spatial fast weights。它的可用性和上下文范围更大，因此与 ICL 不是严格等帧比较。

| condition | n | mean token-F1 | exact | gold substring |
|---|---:|---:|---:|---:|
| ICL，first 60 s quick | 15 | 0.2148 | 0 | 0 |
| ICL，full-video sampling | 15 | 0.2001 | 0 | 0 |
| Spatial-TTT memory，run6 子集 | 15 | 0.1660 | 0 | 0 |
| Frozen/no-memory readout | 15 | 0.1730 | 0 | 0 |

这些分数是开放式回答的 token overlap 诊断，不是人工语义准确率。ICL full-video 相比 Spatial-TTT 的 paired mean token-F1 差为 +0.0340；Spatial-TTT 相比 frozen readout 为 -0.0070。结论应是：当前实验没有显示稳定的 TTT 增益，下一步应先做等帧数、等观测边界的 Spatial-TTT 对照，再调 writer/readout。

详细表格和逐题结果见 [`docs/EXPERIMENT_B_RESULTS.md`](./EXPERIMENT_B_RESULTS.md)。runner 是 [`scripts/evaluate_homesentinel_icl.py`](../scripts/evaluate_homesentinel_icl.py)。

### B2 SVCBench structured counting

通过官方 [SVCBench repository](https://github.com/buaa-colalab/SVCBench) 和 [dataset card](https://huggingface.co/datasets/buaaplay/SVCBench) 的公开文件，只选择 `simulated/0030.mp4`。下载内容为：

- `videos/simulated/0030.mp4`：约 14.8 MB；
- `vcbench_eval.jsonl`：约 1.2 MB；
- 两个问题，O2-Unique 33 点、E1-Action 14 点，共 47 个原始点。

报告的 quick smoke 每 4 个点取 1 个，实际评估 13 点（O2-Unique 9 点、E1-Action 4 点），max-side=224、chunk=4 s/8 frames。所有 query 都检查 `timestamp <= query_time`，没有把未来帧送入模型。ICL 只看最近 8 s/8 frames；Spatial-TTT 按时间增量写入 prefix 后用 text-only answer 读取 fast weights。

| condition | questions | valid/queries | coverage | exact count accuracy | GPA | MoC | UDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICL recent window | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.5625 | 0.3542 |
| Spatial-TTT full prefix | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.8750 | 0.0625 |

这是小样本、低分辨率 smoke，不代表完整 SVCBench。曾启动 448 px/47 点版本，但单点成本过高且没有产生可用结果，已停止；不能把它写成“完整轨迹已完成”。SVCBench runner 支持把 `--query-stride` 改成 1 进行完整评测。

## C：受控搬移与重排

### C1 symbolic mechanism probe

`scripts/prepare_experiment_c.py` 生成一个无需外部下载的受控 fixture：每个 episode 有三个物品、三个住户、初始位置、一个搬移动作、搬移者、状态变化和下一步动作；搬移前后插入 filler 事件。默认生成 12 episodes、204 张 PNG event cards、60 个问题。`scripts/evaluate_experiment_c.py` 使用 64 维稳定 semantic code，比较：

- `oracle`：结构化状态上界；
- `explicit_retrieval`：固定大小的显式 key/value 表；
- `lact`：仓库中的 LaCT fast-weight 关联记忆；
- `none`：不写入、不保留状态的 blind control。

在 `scene_drift_strength=0.35`、3 trials 下：

| arm | filler-only | same-scene | relocated |
|---|---:|---:|---:|
| oracle | 1.0000 | 1.0000 | 1.0000 |
| explicit retrieval | 0.5333 | 1.0000 | 1.0000 |
| LaCT | 0.2389 | 0.5611 | 0.8833 |
| none | coverage 0 | coverage 0 | coverage 0 |

LaCT 的 relocated − same-scene 差值按 query type 为：location +1.0000、mover 0、owner -0.0556、state 0、next_action +0.6667。这个正负混合结果说明它确实在更新状态，但 scene key、关系写入和 readout 仍然没有形成稳定的空间记忆。C1 不是视觉实验，必须与 C2 分开解释。

### C2 PNG-card visual plumbing check

同一 manifest 的事件被渲染为文字较多的 PNG cards，再用 Qwen3-VL-2B 输入 2 个 episode、共 10 个 query。ICL 使用完整 card history；Spatial-TTT 先写入相同帧，再丢弃视频并 text-only 读取。

| arm | queries | mean token-F1 | exact | gold substring |
|---|---:|---:|---:|---:|
| ICL visual | 10 | 0.0444 | 0 | 0 |
| Spatial-TTT | 10 | 0.0196 | 0 | 0 |

两种视觉 arm 都没有得到 exact answer，说明当前 PNG card 只适合检查数据流和读出接口，不能模拟真实家庭视觉。下一版 C 应将相同事件渲染成 photorealistic/3D 场景或使用真实受控视频，并保留当前的 same-scene、relocated、filler-only paired protocol。

## 复现命令

### HomeSentinel B

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 --sample-seconds 60 \
  --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10.jsonl --overwrite

conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 --sample-seconds 0 \
  --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10_fullvideo.jsonl --overwrite
```

### SVCBench B

```bash
SVC_ROOT=/data/quzitsix/svcbench_subset bash scripts/download_svcbench_subset.sh
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n meowbench python scripts/evaluate_svcbench.py \
  --annotations /data/quzitsix/svcbench_subset/subset/svcbench_subset.json \
  --video-root /data/quzitsix/svcbench_subset \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 --max-side 224 --query-stride 4 \
  --output runs/svcbench_0030_stride4.jsonl
```

### C symbolic and visual

```bash
conda run --no-capture-output -n meowbench python scripts/prepare_experiment_c.py \
  --output-dir data/experiment_c --episodes 12 --seed 20261015
conda run --no-capture-output -n meowbench python scripts/evaluate_experiment_c.py \
  --data data/experiment_c/manifest.json --output runs/experiment_c_final \
  --trials 3 --report docs/EXPERIMENT_C_RESULTS.md

# Visual proxy, two episodes per arm
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n meowbench python scripts/evaluate_experiment_c_visual.py \
  --data data/experiment_c/manifest.json --episodes 2 --max-frames 16 \
  --arm icl --device cuda:0 --output runs/experiment_c_visual_icl.jsonl
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n meowbench python scripts/evaluate_experiment_c_visual.py \
  --data data/experiment_c/manifest.json --episodes 2 --max-frames 16 \
  --arm spatial --device cuda:0 --output runs/experiment_c_visual_spatial.jsonl
```

## 当前决策与下一步

1. 不把 A 的空结果或 C1 symbolic 分数写成真实视觉能力。
2. B 的下一轮优先做等帧预算 Spatial-TTT：同一 2 frames/video、同一 query cutoff，避免把更多 Spatial 写入帧误认为 memory gain。
3. C 的下一轮优先换成真实可见的物品和房间渲染，随后再扩大 SVCBench 的 query stride=1；两者都应保留 no-write/frozen control。
4. 如果 TTT 仍然不超过 frozen/ICL，应先检查 Spatial checkpoint 的 readout scale、写入层覆盖范围和跨 chunk position offset，再增加模型尺寸或训练量。
