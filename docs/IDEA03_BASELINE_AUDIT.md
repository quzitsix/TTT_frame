# Idea 03 前置审计：已有逐题基线与复用边界

核验日期：2026-10-25。核验时 Git HEAD 为
`2cd12bfcc72d68694e027f87fb78d6337cc387c3`。本页只复核已固定产物，
不据此选择 Idea 03 超参数；新实验仍须先固定预测，再独立读取 test 答案评分。

## 从逐题记录重新计数

使用 `/data/share/raw/bench_v0/answers.jsonl` 的 `correct_answer` 和
`runs/forecast_bench_v0/research_day_split.json` 的固定 ID。
按天、全部佩戴者共同划分：train111 为第 1–4 天，validation32 为第 5 天，
test57 为第 6–7 天；三个集合互斥并覆盖 200 题。

| 已有方法 | Validation /32 | Test /57 | Test 准确率 |
|---|---:|---:|---:|
| 原始 Qwen 无视频 | 6 | 9 | 15.79% |
| 原始 Qwen recent ICL | 4 | 23 | 40.35% |
| 原始 Qwen full ICL | 6 | 21 | 36.84% |
| 官方 checkpoint，无视频 / bypass | 5 | 11 | 19.30% |
| 官方 checkpoint，recent ICL / bypass | 6 | 11 | 19.30% |
| 官方 checkpoint，full ICL / bypass | 6 | 10 | 17.54% |
| Spatial，本题 fast 更新 | 5 | 12 | 21.05% |
| Spatial，slow-only 初始化 | 3 | 13 | 22.81% |
| Spatial，写入后 bypass | 4 | 11 | 19.30% |
| Idea 01，共享旧读出 recent | 14 | 24 | 42.11% |
| Idea 01，共享旧读出 residual | 14 | 24 | 42.11% |
| Idea 01，共享旧读出 reset | 14 | 24 | 42.11% |
| Idea 01，共享旧读出 plain delayed | 10 | 25 | 43.86% |

原始 Qwen full 与 recent 在 test57 有 15 题选择不同：full 独对 4 题、recent
独对 6 题，净少对 2 题。validation32 有 7 题不同：full 独对 2 题、recent
独对 0 题。

注意 `data/bench_v0_baseline_v1/full_icl_predictions.jsonl` 指官方 checkpoint
的 bypass ICL；用户关心的 **21/57** 来自 `qwen_full_icl_predictions.jsonl`。
这两个结果不能混用。

从 `data/bench_v0_idea01_scaling_v1/shared_n111_predictions.jsonl` 重新计数：
所有 9 个 u1 residual 配置为 test24/57；3 个 random_u4 residual 为23/57；
12 个对应 residual_no_update 均为24/57；共享 recent 为 val14/32、test24/57。
这些数值与旧文档一致。旧读出和 scaling 新共享读出的训练/validation 行为不同，
虽然主 test 数相同，仍应保留各自协议。

## 产物与完整性

- 基线原始日志：`runs/forecast_bench_v0/mcq_baseline_v1/eval/*_shard*.jsonl`，
  8 个文件，1800 行，1800 个唯一 `(id, arm)`，全部 `status=ok`；与九份精简
  `*_predictions.jsonl` 的1800个选择逐一相等。
- 精简单方法预测格式为 `{"id":"egolife_001","prediction":"B"}`。
  原始日志还含 `arm`、`protocol_fingerprint`、`label_logits`、`label_probabilities`、
  `sampling`、`decode_seconds`、`elapsed_seconds`、`write_report` 等字段。
- scaling 精简预测每行含 `id`、`research_split` 和 `predictions` 字典，方法键形如
  `h128_random_u1__residual`。
- 基线归档 manifest 的18个文件、Idea 01 归档的17个文件、scaling归档的52个文件，
  大小及 SHA-256 全部与 manifest 一致。
- 原始基线协议指纹：`8b089206266dd191c2f329cadaa3581d7ce77db96596fb48b63e7595fdbf1da5`。
- 公共输入 SHA：`7a3b6eb61eef3111f409bd2781ec1634ce9f2bad3a66a2dd6919ef049bf29df8`。
- 固定 split 文件 SHA：`e23dfe7584e6a7268187a56e3997510281b576c3aab4892d3c25ae1b6e479603`。

已核对下列旧协议中的执行源码 SHA 全部匹配当前文件：基线10项、Idea 01
forecast5项、Idea 01 readout2项、scaling h128_random_u1 forecast2项。
应新增模块/runner，避免修改以下冻结依赖：

```text
scripts/evaluate_forecast_mcq.py    ttt_frame/forecast_mcq.py
scripts/evaluate_fho_sta.py         ttt_frame/forecast_bench.py
                                  ttt_frame/forecast_features.py
                                  ttt_frame/spatial_videoqa.py
                                  ttt_frame/spatial_model.py
                                  ttt_frame/spatial_memory.py
                                  ttt_frame/vision_attention.py
                                  ttt_frame/vision_patch_embed.py
scripts/run_idea1_mcq_forecasts.py  ttt_frame/idea1_mcq_memory.py
scripts/run_forecast_residual.py   ttt_frame/forecast_residual.py
                                  ttt_frame/predictive_memory.py
scripts/evaluate_idea1_mcq.py       ttt_frame/idea1_mcq_readout.py
scripts/run_idea1_scaling.py        ttt_frame/idea1_scaling_memory.py
```

## 输入预算和复用边界

| 项目 | 原始 Qwen full ICL | 原始 Qwen recent ICL | Idea 01 dense |
|---|---|---|---|
| 历史/近期采样 | 30 + 60 帧 | 0 + 60 帧 | 3600 + 60 帧 |
| 时间覆盖 | 1800s + 30s | 最后30s | 1800s + 30s |
| 最后可见时刻 | 1829.5s | 1829.5s | 1829.5s |
| 最长边 / 音频 | 224 / 无 | 224 / 无 | 224 / 无 |
| bench 标签监督 | 无 | 无 | train111 监督读出；val32 选正则 |
| 查询地址搜索 | 无 | 无 | 无 |
| 表征 | 原生视频 token + decoder | 同左 | 每帧8×256冻结视觉特征 |

ICL context 取 `0,60,...,1740s`，recent 取 `1800,1800.5,...,1829.5s`。
原始 Qwen完整ICL日志平均 forward/读出约0.371s，共同90帧解码平均约12s；
Spatial同90帧写入平均约3.266s、只读回答约0.064s。这是旧并发运行观测，
不能当作新实现实测。官方 checkpoint 替换视觉塔和语言模型，其与原始Qwen
性能差异不只是在线更新。

Idea 01的特征目录为 `runs/forecast_bench_v0/idea01_mcq_v1/features/`，约5.6GiB；
抽查 `egolife_022.pt` 为 float32 `[3660,8,256]`，携带请求/实际时间和媒体指纹。
缓存来自原始Qwen视觉塔，**不是官方Spatial checkpoint的视觉/语言key-value**，
不能当作真正Spatial写入缓存。已有 `forecasts/` 约48MiB，保存最终预测、stats和
metadata，**不保存完整episode fast权重**；查询搜索需要重放或新增state导出。
scaling各配置forecast目录合计约3.0GiB，旧产物应继续保留。

可复用的冻结组件：

```text
runs/forecast_bench_v0/idea01_mcq_v1/forecasts/recent_predictor.pt
runs/forecast_bench_v0/idea01_mcq_v1/text_features.pt
runs/forecast_bench_v0/idea01_mcq_v1/mcq/shared_readout.pt
```

文本缓存每题为 `[6,2048]`，由原始Qwen冻结decoder独立编码
`问题：{question}\n候选动作：{candidate}`；无选项字母、位置、其他候选或答案输入。
共享旧读出是float64的 `1024×2048` ridge系数，并保存train-only视觉/文本归一化。
官方Spatial地址若需要语义映射，不能直接假定这套表征与其内部坐标对齐。

## 快速重算命令

```bash
cd /home/quzitsix/TTT_frame
conda run --no-capture-output -n meowbench python - <<'PY'
import json
from pathlib import Path

def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

splits = json.loads(Path('runs/forecast_bench_v0/research_day_split.json').read_text())['splits']
gold = {r['id']: r['correct_answer'] for r in rows('/data/share/raw/bench_v0/answers.jsonl')}
assert sum(map(len, splits.values())) == len(set().union(*map(set, splits.values()))) == 200
for directory in ('data/bench_v0_baseline_v1', 'data/bench_v0_idea01_v1'):
    for path in sorted(Path(directory).glob('*_predictions.jsonl')):
        records = rows(path)
        pred = {r['id']: r['prediction'] for r in records}
        assert len(records) == len(pred) == 200 and set(pred) == set(gold)
        print(path.stem, {s: f'{sum(pred[i] == gold[i] for i in ids)}/{len(ids)}'
                          for s, ids in splits.items()})
PY
```

这是既有预测的审计命令，读取test答案仅用于评分。新方法的训练、选择和查询搜索
不能调用它获取当前题答案。test57已被反复查看，只有两个录制日，标签为银标签；
本轮后续比较均属探索性。
