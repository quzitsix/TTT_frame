# HomeSentinel/asuka 的 Spatial-TTT 评测

`/data/HomeSentinel/asuka` 是目前本机拥有完整媒体的 HomeSentinel 子集。它有 52 段按时间排列的 `indoor_video.mp4` 和 138 道开放题：`owner` 52 道、`home` 42 道、`event` 44 道。评测入口是 [`scripts/evaluate_homesentinel_spatial.py`](../scripts/evaluate_homesentinel_spatial.py)。

评测只读取视觉视频，不把 `merged_captions.json`、`memory_materials`、题目答案或 evidence 送进模型。脚本按 `/data/HomeSentinel/video_order.json` 中的 `asuka` 顺序逐段处理视频；每段视频结束后调用 `finish_ingest()`，再回答该 cutoff 对应的问题。`video_cutoff_idx=-1` 的长期问题只在第 52 段处理完后回答，带有限 cutoff 的 episodic 问题在对应段之后立即回答，因此不会看到未来视频。

每道题有两个输出臂：

- `memory`：在同一条 Spatial fast-weight 状态上写入历史，再用 `context("read")` 回答；
- `without_memory`：使用相同的基座和当前位置，但在 `context("base")` 中绕过 Spatial fast weights，作为读出对照。

两臂都不保留视频帧、视觉 embedding 或历史 attention KV；长期状态只有 Spatial 的 fast weights 和其更新运行态。`without_memory` 不是一个完全没有视频前向的模型，它与记忆臂使用相同的基座、处理器和位置偏移，区别是回答时不读 fast weights。这个控制可以检验 fast-weight 读出是否改变答案，但不能替代一个独立的未看视频模型基线。

## 当前配置

当前全量作业使用 Qwen3-VL-2B-Instruct、官方 Spatial-TTT nano slow checkpoint、单卡 `cuda:0`、`bfloat16`、每 60 秒最多 4 帧（每帧最长边 448）、最多生成 128 token。60 秒/4 帧是速度基线：它比 4 秒/8 帧稀疏，不能直接当作最密集采样结果。视频总时长约 33,300 秒，预计全量写入约 30–45 分钟，具体取决于每段视频的帧数和当前 GPU 负载。

启动全量评测：

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python -u \
  scripts/evaluate_homesentinel_spatial.py \
  --data-root /data/HomeSentinel/asuka \
  --order-path /data/HomeSentinel/video_order.json \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 \
  --chunk-seconds 60 --frames-per-chunk 4 --max-side 448 \
  --max-new-tokens 128 \
  --output runs/homesentinel_asuka_spatial_60s4f/predictions.jsonl
```

全量运行需要一条不会被终端关闭的作业托管方式，例如：

```bash
systemd-run --user --unit=hs-spatial-ttt --same-dir --collect \
  --working-directory=/home/quzitsix/TTT_frame \
  /home/quzitsix/miniconda3/envs/meowbench/bin/python -u \
  scripts/evaluate_homesentinel_spatial.py \
  --device cuda:0 --chunk-seconds 60 --frames-per-chunk 4 \
  --max-new-tokens 128 \
  --output runs/homesentinel_asuka_spatial_60s4f/predictions.jsonl

journalctl --user -u hs-spatial-ttt -f
```

小规模检查可以只跑第一个视频和 event 题；脚本会自动排除需要完整 52 段历史的 `cutoff=-1` 问题：

```bash
conda run --no-capture-output -n meowbench python -u \
  scripts/evaluate_homesentinel_spatial.py \
  --category event --max-videos 1 --limit 1 \
  --device cuda:0 --chunk-seconds 60 --frames-per-chunk 4 \
  --max-new-tokens 64 --output /tmp/homesentinel_spatial_pilot.jsonl
```

## 输出和分数

`predictions.jsonl` 每道题两行，包含 `query_id`、category、question、gold、cutoff、实际回答的视频索引、arm、答案和耗时。gold 只用于写出后的诊断分数，不进入 prompt。`summary.json` 汇总每个 arm 的总数、按 category 和 cutoff 类型的平均 token F1、规范化 exact 和 gold substring 命中数。

开放式答案的词法分数只能作为可重复的诊断：同义词、复数、顺序和解释文字都会影响结果，不能把 token F1 直接解释成语义准确率。应同时抽查答案，并分别报告 `owner/home` 长期事实与 `event` cutoff 事件；尤其不要把 `without_memory` 的常识性猜测当作视觉记忆成功。

## 已知限制

官方 Spatial-TTT checkpoint 训练于通用视频任务，不是针对 HomeSentinel 的家庭记忆标注。60 秒/4 帧会丢失短动作和细小物品，且 fast weights 容量固定，长期写入可能发生遗忘。该脚本目前不支持从中断点恢复；如果进程被终止，需要从头建立同一条 fast-weight 历史。全量结果生成后，应记录 `summary.json` 的配置和耗时，并将 `predictions.jsonl` 与 summary 一起保留。
