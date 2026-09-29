# 按时间顺序查看 `supermemory-9`

`media_index.json` 是视频片段的时间索引。`sm-7a3e7974f1304cc9c19b` 对应
`supermemory-pilot-v2` 的 18 个连续片段，覆盖 `0--1080` 秒。项目提供的
[`scripts/watch_supermemory.py`](../scripts/watch_supermemory.py) 会读取索引、按
`start_sec` 排序，并把 `session_id` 解析成实际的 mp4 文件；它不会把片段送进
模型，也不会改动任何 memory。

在服务器上先确认片段顺序和文件是否存在：

```bash
python scripts/watch_supermemory.py list
```

如果要把同样的顺序交给摄入命令，可以只输出解析后的路径：

```bash
python scripts/watch_supermemory.py paths > /tmp/supermemory_q9_ordered.txt
mapfile -t VIDEO_ARGS < /tmp/supermemory_q9_ordered.txt
```

此时 `"${VIDEO_ARGS[@]}"` 就是按时间排序的 18 个视频参数。

逐段播放：每个片段在一个 `ffplay` 窗口中播放，窗口关闭后自动进入下一段。按
`q` 可以退出当前播放；`--fullscreen` 和 `--mute` 是可选项。

```bash
python scripts/watch_supermemory.py play --fullscreen
python scripts/watch_supermemory.py play --start 5 --end 8
```

如果希望拖动进度条或反复跳转，可以先拼接成一个文件。默认使用 stream copy，
若源片段编码参数不完全一致，再加 `--reencode`：

```bash
python scripts/watch_supermemory.py concat \
  --output /tmp/supermemory-9-full.mp4 \
  --overwrite

python scripts/watch_supermemory.py concat \
  --output /tmp/supermemory-9-full-reencoded.mp4 \
  --reencode --overwrite
```

默认路径针对本服务器：

```text
media index: /data/quzitsix/meow-releases/supermemory-pilot-v2/media_index.json
media root:  /data/quzitsix/meow-releases/supermemory-pilot-v2/media
memory id:   sm-7a3e7974f1304cc9c19b
```

如果媒体被复制到别处，可以显式替换这三个参数：

```bash
python scripts/watch_supermemory.py list \
  --media-index /path/to/media_index.json \
  --memory-id sm-7a3e7974f1304cc9c19b \
  --media-root /path/to/media
```

这套索引对应的顺序是：`0--60`、`60--120`、……、`1020--1080` 秒。观看时建议
记下“发生时间、对象、动作、最后位置”，后续用这些带时间和动作的事实提问，比
“厨房里有什么”更能区分模型是否真的读出了视频 memory。

## 4 秒 / 8 帧的可恢复摄入和测试

完整 1080 秒视频使用高密度采样时耗时较长。项目提供
[`scripts/build_spatial_memory.py`](../scripts/build_spatial_memory.py)，每完成一个
60 秒源片段就保存一次最新参数状态；中断后重复同一条命令会从最新片段继续：

```bash
conda run --no-capture-output -n meowbench python scripts/build_spatial_memory.py \
  --output runs/q9_full_history_spatial_official_4s8f \
  --progress runs/q9_full_history_spatial_official_4s8f.progress \
  --device cuda:0 --chunk-seconds 4 --frames-per-chunk 8
```

完成后使用 [`scripts/test_spatial_memory.py`](../scripts/test_spatial_memory.py) 做三路
对照。它会依次测试 4 秒/8 帧 memory、原来的 60 秒/16 帧 memory，以及同一官方
checkpoint 的 `--without-memory` 控制：

```bash
conda run --no-capture-output -n meowbench python scripts/test_spatial_memory.py \
  --device cuda:2 --max-new-tokens 96 --concise \
  --question 'Where did I put the empty red mesh bag?'
```

如果有空闲 GPU，可以把三个条件同时运行，缩短评测墙钟时间：

```bash
conda run --no-capture-output -n meowbench python scripts/test_spatial_memory.py \
  --parallel --devices cuda:1,cuda:3,cuda:4 \
  --max-new-tokens 96 --concise \
  --question 'Where did I put the empty red mesh bag?'
```

如果要复现发布包里的原题，直接传入题目的 `item_id`。脚本默认读取
`/data/quzitsix/meow-releases/supermemory-pilot-v2/items.jsonl`，按原始顺序
渲染选项，并在终端显示金标字母；金标和证据不会发送给模型：

```bash
conda run --no-capture-output -n meowbench python scripts/test_spatial_memory.py \
  --item-id supermemory-9 \
  --parallel --devices cuda:3,cuda:4,cuda:5 \
  --max-new-tokens 16
```

这里的 4 秒/8 帧步骤是更密集的 fast-weight 在线摄入，不会更新官方 slow weights；
若要训练 slow weights，仍需带答案的 QA batch 和 `SpatialOfflineTrainer`。
