# 实验 A：FHO-STA 短期预测边界

这个实验先回答一个小问题：在预测时刻之前的第一人称视频前缀中，Spatial-TTT 是否比当前帧和上下文学习更容易读出下一次人—物交互。

评测严格按时间截断。对每个 FHO-STA 标注，只解码 `frame` 或 `clip_frame` 之前的帧；`time_to_contact` 只用于评分和按 0.5、1.0、1.5、2.0 秒分层，不能进入写入过程。

## 首轮条件

固定 Qwen3-VL-2B 和同一组四选一 noun/verb 问题：

| 条件 | 观察输入 |
| --- | --- |
| `blind` | 观测边界的单帧（只在视觉处理器内部重复到 3 帧） |
| `icl_4f` | 最近 4 秒、4 帧，保留在问题上下文 |
| `icl_8f` | 最近 4 秒、8 帧，保留在问题上下文 |
| `icl_8s4f` | 最近 8 秒、4 帧，保留在问题上下文 |
| `spatial_8s4f` | 最近 8 秒、4 帧写入 Spatial-TTT fast weights，随后丢弃视频再提问 |
| `spatial_4s8f` | 最近 4 秒、8 帧写入 Spatial-TTT fast weights，随后丢弃视频再提问 |

脚本报告 noun accuracy、verb accuracy，以及每个 horizon 的分数。它是检查前缀边界和记忆读取的 smoke test，不是官方 FHO-STA detector/mAP 提交；官方指标需要另外接入框和检测评测器。

## 只下载少量数据

Ego4D 数据受许可和 AWS 凭据控制，凭据只需在服务器本地配置，不要写入仓库或发送给我。官方 CLI 支持按 benchmark 和 UID 过滤下载；脚本默认只下载 FHO-STA 标注和选中的 `clips`，不会请求完整 `full_scale`、LMDB 或 STA checkpoint。

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python -m pip install -q ego4d  # 已安装时可跳过
aws configure --profile ego4d

export EGO4D_DIR=/data/quzitsix/ego4d_fho_subset
export EGO4D_VERSION=v2_1
export EGO4D_AWS_PROFILE=ego4d
export FHO_LIMIT=16
export FHO_SEED=0
bash scripts/download_fho_sta_subset.sh
```

`FHO_LIMIT=16` 是首轮小测试；脚本会按 time-to-contact 近似分层，并生成：

```text
$EGO4D_DIR/fho_sta_subset_16/fho_sta_val_subset.json
$EGO4D_DIR/fho_sta_subset_16/clip_uids.txt
$EGO4D_DIR/fho_sta_subset_16/video_uids.txt
$EGO4D_DIR/v2/clips/
```

如果某个 FHO clip 在当前版本的 clips manifest 中不可用，可改为只下载这些样本对应的 canonical videos：

```bash
FHO_MEDIA_DATASET=full_scale bash scripts/download_fho_sta_subset.sh
```

此时评测根目录和帧字段改成 `$EGO4D_DIR/v2/full_scale` 与 `frame`。脚本会拒绝空 UID 文件，避免 CLI 意外退化成下载全部视频。

## 先做结构检查

下载完成后先运行 dry-run。它不会加载模型，只检查标注、视频路径、FPS、观测帧、TTC 和条件矩阵；只要仍有选中样本缺视频就返回退出码 2。

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_fho_sta.py \
  --annotations "$EGO4D_DIR/fho_sta_subset_16/fho_sta_val_subset.json" \
  --video-root "$EGO4D_DIR/v2/clips" \
  --frame-field clip_frame \
  --limit 16 --dry-run
```

## 运行首轮评测

服务器上若使用第 2 号物理 GPU，直接运行：

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_fho_sta.py \
  --annotations "$EGO4D_DIR/fho_sta_subset_16/fho_sta_val_subset.json" \
  --video-root "$EGO4D_DIR/v2/clips" \
  --frame-field clip_frame \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:2 --dtype bfloat16 \
  --limit 16 --max-new-tokens 8 \
  --output runs/fho_sta_smoke_16.jsonl
```

输出包括逐题 JSONL 和同目录的 `summary.json`。每一行保留问题选项、模型原始回答、解析出的选项、gold、TTC、观测帧和耗时，便于之后定位模型是否读到了未来信息或只是猜常见类别。

如果使用 `full_scale` fallback，替换为：

```bash
  --video-root "$EGO4D_DIR/v2/full_scale" --frame-field frame
```

## 如何解释首轮结果

先看同帧预算的差值：`spatial_4s8f - icl_8f` 和 `spatial_8s4f - icl_8s4f`。再按 TTC 分层，看收益是否在更近的交互边界消失。若 Spatial-TTT 与 ICL 几乎相同，说明这组数据只证明视觉识别能力，尚未证明 fast weights 带来了可读的短期预测记忆；此时应保留逐题输出，检查 clip 截断、选项先验和 Spatial checkpoint 的 readout 是否正确。

官方数据下载入口见 [Ego4D CLI 文档](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md)，任务定义和评测说明见 [FHO Short-Term Anticipation](https://github.com/EGO4D/forecasting/blob/main/SHORT_TERM_ANTICIPATION.md)。
