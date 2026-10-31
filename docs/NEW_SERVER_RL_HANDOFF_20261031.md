# 新服务器迁移与 RL–TTT forecasting 继续实验

日期：2026-10-31。旧实验基线提交为
`5219bae450212ab9c56947a75e25d81f4ce1ea12`，仓库为
`git@github.com:quzitsix/TTT_frame.git`。本页命令在**新 Linux 服务器**执行，
工作区默认为 `~/TTT_frame`，Python 使用 conda `meowbench`，不向 base 安装依赖。
本页是迁移和开始工作的命令，不代表 RL 已实现或已获得收益。

本次核验发生在旧服务器。新服务器的数据位置由用户提供，尚未通过本会话远程登录
核验；新机 GPU、权限、目录结构、模型位置和数据字段需要执行下面的检查后确认。
完整研究生成输入是 [ideax_rl_input_20261031.md](../ideax_rl_input_20261031.md)，
已完成实验是 [五方向总结](IDEAX_EXPERIMENT_SUMMARY_20261026.md)。

## 1. 拉取代码并核对版本

以下每个代码块按顺序执行。新目录保护会在已存在目标时停止；不要删除已有目录来
绕过检查，也不要将输出写入原始数据目录。已经 clone 的工作区先检查差异，再自行
选择继续使用；不要在有本地修改的工作区直接覆盖文件。

```bash
set -euo pipefail
test ! -e "$HOME/TTT_frame"
git clone git@github.com:quzitsix/TTT_frame.git "$HOME/TTT_frame"
cd "$HOME/TTT_frame"
git switch main
git pull --ff-only origin main
git merge-base --is-ancestor 5219bae450212ab9c56947a75e25d81f4ce1ea12 HEAD
git log -1 --format='%H %s'
git status --short
source scripts/new_server_paths_20261031.sh
```

`main` 应包含本迁移文档及旧基线提交；不要 checkout 到旧基线后期待这些新文件
仍存在。网络环境不支持 GitHub SSH 时，可将 clone URL 换成
`https://github.com/quzitsix/TTT_frame.git`，私有仓库按已有账户权限认证。

路径文件导出项目、运行输出、模型和数据变量。模型路径是可配置的本地目标，
不是用户已经确认的新机模型位置；已有模型时先设置 `TTT_MODEL_ROOT`、
`TTT_QWEN_MODEL`、`TTT_SPATIAL_CHECKPOINT`，再 source。不要把路径文件当作数据审计。

## 2. 资源、路径和已有任务检查

```bash
pwd
git status --short
nvidia-smi
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
ps -u "$USER" -o pid,etime,args --sort=start_time
df -h "$TTT_PROJECT_ROOT" /mnt/gpu02/disk1 /mnt/nas
for TTT_PATH_NAME in \
  TTT_PROJECT_ROOT TTT_RUN_ROOT TTT_MODEL_ROOT TTT_QWEN_MODEL \
  TTT_SPATIAL_CHECKPOINT TTT_EGOLIFE_ROOT TTT_NEXTME_ROOT \
  TTT_NEXTME_CAPTIONS TTT_BENCH_V0 TTT_BENCH_MIXED \
  TTT_CONTEXT_DATA TTT_VQA_POOL TTT_OPEN_GRPO_DATA; do
  printf '%-26s %s\n' "$TTT_PATH_NAME" "${!TTT_PATH_NAME}"
done
for TTT_DATA_NAME in \
  TTT_EGOLIFE_ROOT TTT_NEXTME_ROOT TTT_NEXTME_CAPTIONS TTT_BENCH_V0 \
  TTT_BENCH_MIXED TTT_CONTEXT_DATA TTT_VQA_POOL TTT_OPEN_GRPO_DATA; do
  test -d "${!TTT_DATA_NAME}"
  test -r "${!TTT_DATA_NAME}"
done
```

用户提供的数据清单如下。目录名中的 `train`、`val`、`GRPO` 不自动证明其符合项目
划分、存在环境真值奖励或具备部署反馈。

| 数据 | 新机路径 / 解释 |
|---|---|
| EgoLife 原视频及官方标注 | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/EgoLife/` |
| NextMe 的 EgoLife 动作标注 | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/NextMe-meta/`，重点检查 `captions/2_egolife/` |
| 原始 200 题 benchmark | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_v0/` |
| 重写选项后的 185 题 MCQ | `/mnt/gpu02/disk1/wuyuhang/data/EgoLife/bench_mixed_v1/` |
| 200 个 case 的 context 训练/验证数据 | `/mnt/nas/users/wuyuhang/forecasting_mcq50_200_v1/data/<case_id>/` |
| 未逐条审核的 context 开放式 VQA 候选池 | `/mnt/gpu02/disk1/wuyuhang/workspace/TTT/outputs/case_lora/20261022T174317Z/data/<case_id>/all_vqa.jsonl` |
| 最近 20 个 case 的开放式 SFT/GRPO 数据 | `/mnt/nas/users/wuyuhang/forecasting_open_grpo_v1/20261029_v1/data/<case_id>/` |

先确认新机 GPU 型号、驱动和任务占用，再指定空闲 GPU。旧机 2026-10-31 的瞬时状态
是 8 张 RTX 4090 D，4/5 卡各约 38.5 GiB 已用；旧机根盘约 88 GiB 空闲、数据盘约
19 GiB 空闲。这不是新机配置，也不是预留资源。不要在旧机打包复制一份巨大的数据树。

## 3. 建立 Python 环境

旧机实测 Python `3.11.16`、torch `2.9.1+cu128`、torchvision `0.24.1`、
CUDA runtime `12.8`、驱动 `570.211.01`、transformers `4.57.6`。
README 中更早的环境描述不作为本次迁移版本依据。其他实测版本记录于
[运行依赖快照](environments/meowbench_runtime_20261031.txt)。它是关键依赖快照，
不是跨平台完整 lockfile；新机 GPU/驱动需要支持所安装的 CUDA 构建。

**仅在新机尚无 `meowbench` 时创建。** 先查看环境列表；已有可用环境先核验，
不要盲目重建或替换其 torch。

```bash
conda env list
conda create -n meowbench python=3.11 pip -y
conda run --no-capture-output -n meowbench python -m pip install \
  'torch==2.9.1' 'torchvision==0.24.1' \
  --index-url https://download.pytorch.org/whl/cu128
conda run --no-capture-output -n meowbench python -m pip install \
  -r docs/environments/meowbench_runtime_20261031.txt
conda run --no-capture-output -n meowbench python -m pip install --no-deps -e .
conda run --no-capture-output -n meowbench python -m pip check
```

如果已有合适 torch，跳过 torch 安装行。若新机驱动/GPU 不支持该 CUDA 构建，先选择
兼容构建并记录差异；不要仅为复现旧版本而改变共享服务器驱动。

```bash
conda run --no-capture-output -n meowbench python - <<'PY'
import platform
from importlib.metadata import version
import torch
print('python:', platform.python_version())
for name in ['torch', 'torchvision', 'transformers', 'peft', 'accelerate',
             'av', 'safetensors', 'pillow', 'numpy', 'pytest']:
    print(name, version(name))
print('cuda_runtime:', torch.version.cuda)
print('cuda_available:', torch.cuda.is_available())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
PY
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
conda run --no-capture-output -n meowbench python -m pytest -q \
  tests/test_predictive_memory.py tests/test_spatial_probe.py \
  tests/test_forecast_mcq.py tests/test_idea5_acquisition.py
```

该轻量命令本次在旧机实际得到 **68 passed in 2.02s**，不需要外部模型缓存，
不启动训练。它不验证新机真实视频是否可解码。

## 4. 按需从旧机拉取模型和未入 Git 的产物

**已有相同本地模型时跳过传输，直接核验哈希。** 以下 `TTT_OLD_SSH` 必须改成旧机
可 SSH 登录的地址；本会话没有新机登录地址，也没有实际执行传输。`rsync -a` 保留
时间等属性，不使用 `--delete`。首次目标目录必须不存在；若网络中断，确认目标为
本次未完成复制后，只重跑相应 rsync 行即可断点续传。

```bash
export TTT_OLD_SSH='quzitsix@OLD_SERVER_HOST'
test "$TTT_OLD_SSH" != 'quzitsix@OLD_SERVER_HOST'
test ! -e "$TTT_QWEN_MODEL"
test ! -e "$(dirname "$TTT_SPATIAL_CHECKPOINT")"
mkdir -p "$TTT_QWEN_MODEL" "$(dirname "$TTT_SPATIAL_CHECKPOINT")"
rsync -a --partial --info=progress2 \
  "$TTT_OLD_SSH:/data/quzitsix/models/Qwen3-VL-2B-Instruct/" \
  "$TTT_QWEN_MODEL/"
rsync -a --partial --info=progress2 \
  "$TTT_OLD_SSH:/data/quzitsix/models/Spatial-TTT-nano/" \
  "$(dirname "$TTT_SPATIAL_CHECKPOINT")/"
```

Qwen 完整目录约 4.0 GiB；Spatial checkpoint 约 4.7 GiB。Spatial 原生路径需要
第二项；只做 Qwen/LoRA 新原型时可暂缓 Spatial，但旧九组 MCQ 协议的 `--prepare`
仍会计算该 checkpoint 哈希，不能假设 `--family qwen` 可以省略它。

以下权重哈希来自旧实验归档，执行时会重新读取新机文件进行校验：

```bash
conda run --no-capture-output -n meowbench python - <<'PY'
import hashlib, os
from pathlib import Path
checks = [
    (Path(os.environ['TTT_QWEN_MODEL']) / 'model.safetensors',
     '7de1838c87a5349b016c26a1c3f7d2bc400a3d485f95ef39a7059ffd734977a0'),
    (Path(os.environ['TTT_SPATIAL_CHECKPOINT']),
     '5cc68664bf1de0edc509d0596f654d9bed883c49aa2a85baafa4d51c1acdb947'),
]
for path, expected in checks:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == expected, f'checkpoint differs: {path}'
    print('verified', path)
PY
```

若只迁移 Qwen，先从上述 `checks` 删除未迁移的 Spatial 项；这不改变旧协议。
另一 revision 的模型需要建立新协议，不能改写旧归档使其“通过”。除权重外，实际
run 协议还会核验 tokenizer、processor、config 等文件，因此应复制 Qwen 整个目录。

Git 已包含约 54 MiB 的逐题结果、报告和协议。`runs/` 没有任何 Git 追踪文件。
只需要继续研发时，不必复制全部历史 run。需要复用 Idea1 特征/读出时，可额外执行：

```bash
cd "$TTT_PROJECT_ROOT"
test ! -e runs/forecast_bench_v0/idea01_mcq_v1
mkdir -p runs/forecast_bench_v0/idea01_mcq_v1
rsync -a --partial --info=progress2 \
  "$TTT_OLD_SSH:/home/quzitsix/TTT_frame/runs/forecast_bench_v0/idea01_mcq_v1/" \
  runs/forecast_bench_v0/idea01_mcq_v1/
test ! -e ideax_output
mkdir ideax_output
rsync -a --partial \
  "$TTT_OLD_SSH:/home/quzitsix/TTT_frame/ideax_output/" ideax_output/
```

Idea1 完整 run 约 5.7 GiB，其中特征约 5.6 GiB；`recent_predictor.pt` 约 2.1 MiB、
`text_features.pt` 约 9.7 MiB、`shared_readout.pt` 约 17 MiB。原始 `ideax_output/`
约 160 KiB，仍未入 Git，复制它是保留旧提案原文。Idea2 run 约 931 MiB、Idea4 v2
约 2.6 GiB、scaling 约 3.3 GiB，只有明确要重放相应实验时才复制。

**复制旧缓存不等于新路径可直接运行旧 runner。** Idea1 缓存记录绝对媒体路径，
验证器会 `stat` 这些旧路径并比较 size/mtime；旧协议还绑定输入 manifest、源码和
张量哈希。不得 `sed` 旧 JSON、重写缓存、关闭验证来迁移。可选择重新提取到新 run，
或者后续新增独立的 relocation adapter：逐文件验证源视频 SHA、记录旧新路径与
时间属性，再以新协议引用原缓存。这个 adapter 当前**尚未实现**。

旧 public loader 会规范化 benchmark 根路径，但禁止 `samples/<case_id>/` 内视频
或子目录被软链接重定向到别处。不要把给媒体建软链接当通用迁移方案。

## 5. 原始 bench_v0 审计与固定划分核对

此阶段允许读取私有 metadata/答案做**离线数据审计**；审计文件不进入 inference
输入。原 benchmark 的固定 split 已入 Git，位置为
`data/bench_v0_baseline_v1/research_day_split.json`。下面在新路径重新审计，但断言
题集和日划分必须与归档一致，不随机重新划分。

```bash
cd "$TTT_PROJECT_ROOT"
test ! -e "$TTT_RUN_ROOT/migration_bench_v0_audit"
conda run --no-capture-output -n meowbench python scripts/prepare_forecast_bench.py \
  --root "$TTT_BENCH_V0" \
  --output-dir "$TTT_RUN_ROOT/migration_bench_v0_audit" \
  --make-research-split
conda run --no-capture-output -n meowbench python - <<'PY'
import json, os
from pathlib import Path
old = json.loads(Path('data/bench_v0_baseline_v1/research_day_split.json').read_text())
new_path = Path(os.environ['TTT_RUN_ROOT']) / 'migration_bench_v0_audit/research_day_split.json'
new = json.loads(new_path.read_text())
for key in ['splits', 'days', 'id_to_day', 'counts', 'inputs_manifest_sha256',
            'metadata_manifest_sha256']:
    assert new[key] == old[key], f'benchmark differs from old release: {key}'
print('same release and day split:', new['counts'])
print('relocated root:', new['root'])
PY

# The split audit does not open answers.jsonl.  Protect the old re-scoring
# reference separately before reusing frozen predictions.
conda run --no-capture-output -n meowbench python - <<'PY'
import hashlib
import os
from pathlib import Path
path = Path(os.environ['TTT_BENCH_V0']) / 'answers.jsonl'
digest = hashlib.sha256(path.read_bytes()).hexdigest()
expected = '4c1c36e0ba137f921c009eceb64c6586829700bc183faae3cef7f8ffc6f20b6d'
assert digest == expected, f'answer manifest differs: {path}'
print('answers.jsonl SHA256 verified:', digest)
PY
```

预期为 train111 / validation32 / test57。断言不通过时保留日志并查明版本差异，
不要调整旧结果或强行匹配；相同 case ID 不证明视频、选项或 cutoff 相同。上面的
answers SHA 只保护旧评分文件；它不证明视频字节与旧缓存相同。若要复用特征，仍需
对每个 source media 做字节 SHA/尺寸/时间戳审计，或在新 run 中重新提取。

185 题 `bench_mixed_v1` 需要新数据适配器、旧新 ID 映射、选项及时间覆盖核验，
再按原始录制日继承分组。**其 split 数量尚未核验，不能写成 111/32/57，也不能
直接把原 200 题预测字母拿来评分。** 原 loader 严格要求特定字段、六选项和媒体布局，
目前没有承诺可直接读取 mixed、context SFT/VQA 或 GRPO 数据。

## 6. 真实视频 8 帧 smoke 与全仓测试

先选空闲 GPU，把下例的 `cuda:0` 改成实际设备。以下仅验证模型、decoder、视频路径
和写盘，不训练 RL，不输出 forecasting 准确率；只使用合法前缀，标记为 partial cache。

```bash
test ! -e "$TTT_RUN_ROOT/migration_visual_smoke"
conda run --no-capture-output -n meowbench python scripts/cache_forecast_features.py \
  --root "$TTT_BENCH_V0" --ids egolife_001 \
  --model-path "$TTT_QWEN_MODEL" --device cuda:0 --dtype bfloat16 \
  --max-side 224 --batch-size 4 --sample-fps 2 \
  --context-tail-seconds 2 --limit-frames 8 \
  --output-dir "$TTT_RUN_ROOT/migration_visual_smoke"
conda run --no-capture-output -n meowbench python - <<'PY'
import os, torch
from pathlib import Path
path = Path(os.environ['TTT_RUN_ROOT']) / 'migration_visual_smoke/egolife_001.pt'
obj = torch.load(path, map_location='cpu', weights_only=True)
assert tuple(obj['features'].shape) == (8, 8, 256)
assert obj['complete_prefix'] is False
assert bool((obj['actual_prefix_seconds'] <= obj['requested_prefix_seconds'] + 1e-6).all())
assert bool((obj['requested_prefix_seconds'] < 1830).all())
print('8 causal frames decoded and encoded:', path)
PY
conda run --no-capture-output -n meowbench python -m pytest -q
```

旧机最终全仓为 **711 passed**；新机缺 CLIP cache 时，`test_lact` 中一个集成测试会
skip。不要把跳过写成全项通过。若要包含该项，可先复制原有完整 Hugging Face cache
目录（约 1.2 GiB，保留 blobs、refs、snapshots 和相对软链接），再重新测试：

```bash
TTT_CLIP_HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
test ! -e "$TTT_CLIP_HUB/models--openai--clip-vit-base-patch32"
mkdir -p "$TTT_CLIP_HUB/models--openai--clip-vit-base-patch32"
rsync -a --partial --info=progress2 \
  "$TTT_OLD_SSH:/home/quzitsix/.cache/huggingface/hub/models--openai--clip-vit-base-patch32/" \
  "$TTT_CLIP_HUB/models--openai--clip-vit-base-patch32/"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  conda run --no-capture-output -n meowbench python -m pytest -q
```

若专门设置过 `HF_HUB_CACHE`，将 `TTT_CLIP_HUB` 改为该 cache 根目录。显式离线变量
用于避免旧 CLIP processor 触发网络探测，并不妨碍本地模型加载。

## 7. 在新路径复核旧逐题汇总

这一步仅对已冻结预测重新做分析，不产生新预测，不是新独立测试。要求第 5 节原始
数据身份核验通过。保留归档 config 原文，仅另存一个使用新数据路径的分析配置。

```bash
test ! -e "$TTT_RUN_ROOT/migration_summary_config.json"
test ! -e "$TTT_RUN_ROOT/migration_summary_reproduce"
conda run --no-capture-output -n meowbench python - <<'PY'
import json, os
from pathlib import Path
config = json.loads(Path('data/bench_v0_ideax_summary_v1/config.json').read_text())
config['benchmark_root'] = os.environ['TTT_BENCH_V0']
config['split_manifest'] = str(Path(os.environ['TTT_RUN_ROOT']) /
                               'migration_bench_v0_audit/research_day_split.json')
target = Path(os.environ['TTT_RUN_ROOT']) / 'migration_summary_config.json'
target.parent.mkdir(parents=True, exist_ok=True)
with target.open('x') as stream:
    json.dump(config, stream, ensure_ascii=False, indent=2)
    stream.write('\n')
PY
conda run --no-capture-output -n meowbench python scripts/summarize_ideax_answers.py \
  --config "$TTT_RUN_ROOT/migration_summary_config.json" \
  --output-dir "$TTT_RUN_ROOT/migration_summary_reproduce"
```

应仍得到 200 题 × 25 方法的旧回答汇总；权重输入、原视频、labels 和新版 benchmark
都没有因此变成新的实验结果。

## 8. 给新会话的继续工作指令

将下面内容贴给新服务器上的工作代理，先完成数据与奖励审计，再实现最小方案。
目前仓库**没有可声称直接启动本轮 RL 训练的 runner**；因此本交接不提供虚构的
`train_rl.py` 命令。完成设计、数据隔离和实现后，实际训练命令必须随协议一并保存。

```text
请接续 ~/TTT_frame。先执行 pwd、git status --short，读取适用 AGENTS.md，
检查 GPU/进程/磁盘；这是 Linux，Python 使用 conda meowbench，不向 base 安装包。
读取 docs/NEW_SERVER_RL_HANDOFF_20261031.md、ideax_rl_input_20261031.md、
docs/IDEAX_EXPERIMENT_SUMMARY_20261026.md，source scripts/new_server_paths_20261031.sh。

先完成新机环境、原 bench_v0 身份和固定 day split 核对、8 帧视觉 smoke 与测试。
不要改写旧冻结源码/协议/结果，不覆盖 ideax_output；新模块、协议和输出独立版本化。
185 题 mixed 与原 200 题分别报告，先审计 ID/视频/选项/cutoff 对应，再继承录制日分组。
NextMe captions、context SFT/VQA 池及 20-case GRPO 数据先审计来源、时间戳、
泄漏、重复和 teacher/人工来源；目录名不能当作合法训练或 reward 的证明。

研究目标是 RL 能否帮助参数记忆/获取/写入决策改善 forecasting。
按 ideax_rl_input_20261031.md 生成独立 IdeaX 方案，优先建立同反馈的 SFT/CE/MSE、
监督 gate、固定/随机/近期策略、匹配预算 ICL/检索和 reset/wrong-video 对照。
区分训练日离线 RL、测试时有合法延迟反馈的在线 RL、推理搜索和伪标签自训练。
最终测试答案与 cutoff 后视频不能用于当前题训练、奖励、检索或选择超参数。
每题重置 fast state，测试 policy 是否冻结写入协议；不把全体 context 的训练数据
直接合并成 train，测试日的 context caption 也不能静默成为训练监督。

先实现最小可验证管线并给出真实可运行命令，再推进实际实验。保存冻结预测、逐题
选择及文字、奖励可得时刻、训练日志、协议/模型/源码哈希、预算和实测成本。
原 test57 已反复探索，标签为银标签；新选项重写也不会自动产生独立测试集。
```

本页已核对现存命令的参数与版本，轻量测试在旧机执行通过；环境安装、SSH 传输、
新机数据审计和新机 smoke 尚未执行。新服务器真实运行后，应把 GPU/依赖、数据审计
结果、失败日志及实际命令补入新的执行记录，避免把迁移说明写成已完成的迁移。
