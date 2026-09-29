# Q9 全视频 4 秒 / 8 帧 Spatial-TTT 运行

## 运行结果

输出目录：`runs/q9_full_history_spatial_official_4s8f`

```text
sessions:       18
chunks:         270
frames:         2160
visual_tokens:  211680
input_tokens:   225000
ingest_seconds: 3843.0
position_offset: 28440
```

配置仍然使用官方 `Spatial-TTT-nano` 完整 checkpoint、21 个 TTT 层、4 个 fast
heads、`chunk_size=2648` 和 `base_lr=1e-3`。相对于原来的 60 秒/16 帧版本，变化只
是视频采样密度：每 4 秒取 8 帧。这个过程更新的是本次视频的 fast weights，不是
官方 slow weights 的反向训练。

可恢复构建命令是：

```bash
conda run --no-capture-output -n meowbench python scripts/build_spatial_memory.py \
  --output runs/q9_full_history_spatial_official_4s8f \
  --progress runs/q9_full_history_spatial_official_4s8f.progress \
  --device cuda:0 --chunk-seconds 4 --frames-per-chunk 8
```

进度目录只保留最近一次已完成片段。命令中断后重复执行即可续跑；完成 18 个片段
后，最终 memory 目录才会被复制出来。

## 直接测试命令

三种条件可以同时放到三张空闲卡：

```bash
conda run --no-capture-output -n meowbench python scripts/test_spatial_memory.py \
  --parallel --devices cuda:3,cuda:4,cuda:5 \
  --max-new-tokens 96 --concise \
  --question 'Where did I throw the empty red mesh bag earlier?'
```

脚本输出的三个标签分别是：

1. 4 秒/8 帧 full-history memory；
2. 60 秒/16 帧 full-history memory；
3. 同一官方 checkpoint 的 `--without-memory`。

本次已实测：

```text
4s/8f memory:  trash can
16f/60s memory: trash can
without-memory: on floor
```

因此这个具体问题上 memory 改变了答案，且方向更接近视频证据。另一个问题
`What objects did I interact with in the kitchen?` 的三路回答仍然是常见厨房物体列表，
不能据此声称模型已经学会完整物体清单。`What color onion did I handle ...?` 也出现过
memory 与控制条件不一致但都不可靠的情况，所以正式评测应使用人工核对的时间证据题集。

## 八张 GPU 的使用方式

单条 Q9 轨迹的 fast-weight 更新有时间依赖：第 `t+1` 个视频块必须读取第 `t` 块的
状态，因此不能把相邻块直接分给不同 GPU 后再拼接。当前用一张卡顺序构建是正确的
协议。其余 GPU 可以用于：

| 任务 | 建议 GPU 分配 | 是否可并行 |
|---|---|---|
| 4s/8f、16f/60s、无 memory 三路问答 | `cuda:3,4,5` | 可以；测试脚本已支持 |
| 不同视频或不同采样密度的独立 memory | 每条轨迹一张卡 | 可以 |
| 外部 teacher 的帧包分析 | 剩余 GPU 分片 | 可以 |
| 同一个视频的单条 fast-memory 轨迹 | 一张卡，按时间顺序 | 不能直接拆分 |
| slow-weight 离线训练 | 每张卡处理独立 episode，梯度同步 | 需要后续 DDP/梯度累积 |

当前服务器上 `cuda:2`、`cuda:6` 可能有其他任务，启动并行测试前应先用：

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

确认设备空闲后再替换 `--devices`。每个 2B 进程约占 6 GB 显存，4090 的容量足以
同时运行多个条件；GPU 利用率仍可能受视频解码和 processor 限制，后续可以增加 CPU
预取或预先缓存帧包。
