# 实验 A：FHO-STA 短期预测（ICL 与 Spatial-TTT）

状态：**待运行**。本文件由 `scripts/report_fho_sta_results.py` 生成。

本实验只使用观测边界之前的帧。ICL 将前缀保留在问题上下文中；Spatial-TTT 按时间顺序写入 fast weights，提问阶段丢弃视频后只读取参数状态。指标是封闭集合 noun/verb 选择题准确率，不能等同于官方 FHO-STA detector mAP。

## 实验设置

- 图像最长边（`max_side`）：由各 run 的元数据记录；旧结果若没有该字段则显示 unknown。
- ICL/TTT 必须使用相同观测边界和帧预算；`spatial_4s8f` 对比 `icl_8f`，`spatial_8s4f` 对比 `icl_8s4f`。
- 预测距离按 FHO 标注的 `horizon_bin` 分层。只在真实运行产生结果后填写数值。

## Run 状态

| run | 输入 | max_side | items/rows | model | 状态 |
|---|---|---:|---:|---|---|
| side256 | `/home/quzitsix/TTT_frame/runs/fho_sta_smoke_256.jsonl` | unknown | unknown/unknown | `unknown` | 未找到结果文件 |
| side448 | `/home/quzitsix/TTT_frame/runs/fho_sta_smoke_448.jsonl` | unknown | unknown/unknown | `unknown` | 未找到结果文件 |
| side672 | `/home/quzitsix/TTT_frame/runs/fho_sta_smoke_672.jsonl` | unknown | unknown/unknown | `unknown` | 未找到结果文件 |

## Overall noun/verb accuracy

| run | condition | task | n | correct | accuracy | errors | mean latency (s) |
|---|---|---|---:|---:|---:|---:|---:|
| — | — | — | — | — | — | — | — |

## Accuracy by horizon

| run | horizon | condition | noun | verb | action joint | n (noun/verb) |
|---|---:|---|---:|---:|---:|---:|
| — | — | — | — | — | — | — |

## Paired Spatial-TTT minus ICL

| run | TTT | matched ICL | task | horizon | pairs | TTT acc | ICL acc | Δ | wins/ties/losses |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| — | — | — | — | — | — | — | — | — | — |

## 解释边界

- 只有在相同帧数、分辨率、观测边界下的 paired Δ 才用于判断 TTT 是否优于 ICL；不同窗口只能作为采样预算分析。
- `action joint` 要求同一 UID 的 noun 与 verb 都答对；若只有 summary 没有逐题 JSONL，则无法计算该项和 paired Δ。
- 小样本结果只用于 smoke test。不要将 closed-set accuracy 写成官方 FHO-STA mAP。

## 复现实验

```bash
conda run --no-capture-output -n meowbench python scripts/evaluate_fho_sta.py \
  --annotations "$EGO4D_DIR/fho_sta_subset_16/fho_sta_val_subset.json" \
  --video-root "$EGO4D_DIR/v2/clips" --frame-field clip_frame \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:2 --dtype bfloat16 --limit 16 --max-new-tokens 8 \
  --output runs/fho_sta_smoke_16.jsonl

python scripts/report_fho_sta_results.py \
  --run smoke16=runs/fho_sta_smoke_16.jsonl \
  --max-side 448 \
  --output docs/FHO_STA_EXPERIMENT_A_RESULTS.md
```

当前没有可读取的 FHO-STA JSONL/summary 结果。配置 Ego4D 凭据并运行上面的评测命令后，再重新执行报告命令；本次输出没有填入任何估计分数。
