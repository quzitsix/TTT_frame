# Experiment B: long-history and structured counting

## HomeSentinel quick B

The ICL file contains **15** event/home queries (IDs shared with the Spatial-TTT run: **15**).
Scores are the existing open-ended lexical diagnostics (token F1, exact normalized match, and gold-substring count). They are not human-validated semantic accuracy.

| condition | n | mean token F1 | exact | gold substring |
|---|---:|---:|---:|---:|
| ICL visual (quick) | 15/15 | 0.2148 | 0 | 0 |
| Spatial-TTT memory | 15/15 | 0.1660 | 0 | 0 |
| Frozen readout | 15/15 | 0.1730 | 0 | 0 |
| ICL visual (full-video) | 15/15 | 0.2001 | 0 | 0 |

The quick ICL control samples **2 frames from the first 60 seconds of each observed video**, then caps the history at **32 frames**. The optional full-video ICL row uses the same frame count but samples over each complete observed video. The Spatial-TTT numbers come from the existing `homesentinel_asuka_spatial_60s4f_run6` run, whose full run uses 60-second/4-frame chunks over the 52-video stream. Therefore the quick row is a diagnostic comparison; the full-video row is the closer temporal control.

| query | ICL quick F1 | Spatial F1 | frozen F1 | ICL full-video F1 |
|---|---:|---:|---:|---:|
| `asuka_event_001` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `asuka_home_031` | 0.2069 | 0.2667 | 0.3226 | 0.2857 |
| `asuka_home_032` | 0.5600 | 0.1667 | 0.2308 | 0.3810 |
| `asuka_event_002` | 0.0714 | 0.2069 | 0.1446 | 0.1429 |
| `asuka_event_003` | 0.1333 | 0.2222 | 0.2222 | 0.1429 |
| `asuka_home_030` | 0.4545 | 0.2609 | 0.2609 | 0.5455 |
| `asuka_event_004` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `asuka_event_005` | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| `asuka_event_006` | 0.1111 | 0.1905 | 0.1905 | 0.0000 |
| `asuka_home_033` | 0.3529 | 0.0800 | 0.0952 | 0.2222 |
| `asuka_event_007` | 0.1818 | 0.2222 | 0.2500 | 0.2400 |
| `asuka_event_008` | 0.1538 | 0.2222 | 0.2222 | 0.1538 |
| `asuka_home_034` | 0.5000 | 0.2667 | 0.2703 | 0.3333 |
| `asuka_event_009` | 0.2857 | 0.2857 | 0.2857 | 0.1538 |
| `asuka_home_035` | 0.2105 | 0.1000 | 0.1000 | 0.4000 |

## SVCBench structured counting

This subset contains one generated video (`simulated/0030.mp4`), two questions, and 47 raw ordered query points. The reported smoke keeps every fourth point (13 evaluated points: 9 O2-Unique and 4 E1-Action) to make the two-condition run tractable. At each evaluated point, the runner enforces a strict temporal prefix (`timestamp <= query_time`). ICL receives a bounded recent 8-second/8-frame window; Spatial-TTT ingests all sampled prefix frames incrementally and answers from fast weights.

| condition | questions | valid/queries | coverage | exact count accuracy | GPA | MoC | UDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| ICL recent window | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.5625 | 0.3542 |
| Spatial-TTT full prefix | 2 | 13/13 | 1.0000 | 0.1250 | 0.1250 | 0.8750 | 0.0625 |

The metrics are a small-sample trajectory smoke test rather than a claim about the full SVCBench benchmark. Exact count accuracy is the direct pointwise measure; GPA rewards numerical closeness, MoC measures monotonicity of predictions, and UDA measures agreement of change direction.

## Reproduction

```bash
# HomeSentinel quick visual ICL (15 event/home queries in the reported run)
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 \
  --sample-seconds 60 --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10.jsonl --overwrite

# Same 15 queries, sampling over each complete observed video
conda run --no-capture-output -n meowbench python scripts/evaluate_homesentinel_icl.py \
  --category all --max-cutoff 10 --frames-per-video 2 \
  --sample-seconds 0 --max-history-frames 32 --device cuda:2 \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --output runs/homesentinel_b_icl_event_home10_fullvideo.jsonl --overwrite

# Selective SVCBench data (one video, no repository-wide download)
SVC_ROOT=/data/quzitsix/svcbench_subset bash scripts/download_svcbench_subset.sh
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n meowbench python scripts/evaluate_svcbench.py \
  --annotations /data/quzitsix/svcbench_subset/subset/svcbench_subset.json \
  --video-root /data/quzitsix/svcbench_subset \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \
  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \
  --device cuda:0 --dtype bfloat16 --max-side 224 --query-stride 4 \
  --output runs/experiment_b_svcbench_0030_stride4.jsonl
```

The SVCBench annotations and selective-download layout follow the [official SVCBench repository](https://github.com/buaa-colalab/SVCBench) and [dataset card](https://huggingface.co/datasets/buaaplay/SVCBench).
