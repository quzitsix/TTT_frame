# 官方 Spatial-TTT 权重：SuperMemory Q9 全视频实验

本次实验使用 `supermemory-pilot-v2` 中 `sm-7a3e7974f1304cc9c19b` 的 18 个连续片段，覆盖 0--1080 秒。视频片段按 `media_index.json` 的时间顺序输入，没有把视频拼成一个新文件。

## Checkpoint

官方发布的是完整的 [`THU-SI/Spatial-TTT-nano`](https://huggingface.co/THU-SI/Spatial-TTT-nano) 模型，而不是可以单独替换到原始 Qwen 上的 slow-only 文件：

```text
/data/quzitsix/models/Spatial-TTT-nano/model.safetensors
size: 5018692144 bytes
sha256: 5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947
```

它包含训练过的语言模型和 Spatial-TTT 的 slow 参数。摄入视频时，slow 参数保持不变，视频只更新每层的 fast weights。保存目录中的 `spatial.safetensors` 是加载后的 TTT slow 参数，4/8/16 帧实验的文件哈希一致（`d5043ec0...`）。

## 三个视频记忆目录

| 目录 | 每 60 秒采样 | 总帧数 | 视觉 token | 摄入时间 | 官方 checkpoint |
|---|---:|---:|---:|---:|---|
| `runs/q9_full_history_spatial_official_4f` | 4 | 72 | 7,056 | 114.1 s | 是 |
| `runs/q9_full_history_spatial_official_8f` | 8 | 144 | 14,112 | 227.0 s | 是 |
| `runs/q9_full_history_spatial_official_16f` | 16 | 288 | 28,224 | 454.9 s | 是 |

三份记忆都使用 21 个 TTT 层、`chunk_size=2648`、`window_size=2648`、`num_heads=4`、`base_lr=1e-3`。16 帧版本最接近官方训练时的最低帧数设置，但仍使用本项目的 448 像素缩放和 60 秒分块。

## 复现命令

16 帧版本的摄入命令如下；18 个视频路径按 `media_index.json` 顺序展开：

```bash
mapfile -t VIDEO_ARGS < <(python scripts/watch_supermemory.py paths)
conda run --no-capture-output -n meowbench python -m ttt_frame.spatial_videoqa ingest \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --video "${VIDEO_ARGS[@]}" \
  --save runs/q9_full_history_spatial_official_16f \
  --chunk-seconds 60 --frames-per-chunk 16 --max-side 448 \
  --device cuda:2 --dtype bfloat16
```

读取时，省略 `--without-memory` 才会进入 Spatial-TTT read 分支：

```bash
conda run --no-capture-output -n meowbench python -m ttt_frame.spatial_videoqa ask \
  --memory runs/q9_full_history_spatial_official_16f \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --device cuda:2 --dtype bfloat16 \
  --question 'Where did I throw the empty red mesh bag earlier?'
```

加入 `--without-memory` 可以得到同一官方模型的 frozen/readout 对照。

## 读取结果

对问题 `Where did I throw the empty red mesh bag earlier?`：

```text
Spatial-TTT + 16f memory:
I can confirm that the empty red mesh bag was thrown away in the trash can.

同一 checkpoint + --without-memory:
I can confirm that I've seen the empty red mesh bag before. It was located in
the area of the room with a high level of activity, specifically in the center
of the room.
```

这说明 fast-weight read 分支确实改变了读出，并且带记忆的回答更接近金标中的白色垃圾桶位置。不过它没有生成“near the kitchen island”这一完整细节，仍不能算稳定的空间定位能力。

对开放问题 `What objects did I interact with in the kitchen?`，16 帧版本生成了 microwave、coffee maker、toaster、kettle、cutting board 等重复列表；无记忆版本也生成相似的厨房常见物体列表。因此这个问题主要反映官方模型的厨房先验，不能作为记忆准确率指标。

Q9 多选题的金标是 A。按 `items.jsonl` 中评测器实际使用的完整句式（每个选项都以 “You threw ...” 开头）运行时，16 帧版本输出 C，`--without-memory` 输出 A；因此这次 Spatial-TTT 读出反而把正确的基础模型答案改成了错误的 C。把选项缩短成 [`docs/TEACHER_PILOT_RESULTS.md`](TEACHER_PILOT_RESULTS.md) 中的片段句式后，两种模式都会输出 A，说明当前读出对提示词很敏感。多选题单次命中不能证明视频记忆被正确检索。

## 协议限制

官方 Spatial-TTT 的 slow/语言模型是在带 dense scene-description 监督的训练协议中得到的。本项目的直接视觉摄入只使用固定文本 `Observe the visible environment.`，没有把 teacher QA 或 caption 写入视频状态；因此存在明显的写入分布差异。这里的“重新训练”指测试时对 fast weights 的在线更新，不是用这段家庭视频反向训练 slow weights。
