# ChatGPT 订阅版 Teacher 桥接

ChatGPT Pro 的网页订阅和 OpenAI API 是两套服务。订阅网页本身没有可供本地脚本直接调用的 API key；本项目保留了已经验证过的 Codex CLI 后端，也支持你后来申请的 OpenAI-compatible token。token 只从环境变量读入，不会进入 prompt、日志或 memory checkpoint。

## 导出

```bash
conda run --no-capture-output -n meowbench python -m ttt_frame.spatial_videoqa export-teacher \
  --video /path/to/clip.mp4 --out /path/to/teacher_packet \
  --chunk-seconds 4 --frames-per-chunk 8 --max-side 448
```

输出目录包含 `frames/`、`manifest.json` 和 `teacher_prompt.md`。把 `frames/` 与 prompt 上传到 ChatGPT 网页，要求只返回 `ttt_frame.teacher_scene/1` JSON；不要让模型回答 benchmark 问题。manifest 中的源视频 SHA256 和帧时间戳用于防止混包和未来信息泄漏。长视频（例如 SuperMemory-V2 的多个 60 秒片段）应按片段导出，分析后按 `session_id` 顺序分别导入。

## Codex CLI teacher（已完成订阅版单片段测试）

服务器上的 `codex` CLI 使用已登录的 ChatGPT 会话，非 API token。它会在空的只读临时工作目录运行，禁用 shell、联网搜索、插件和多代理，只接收 packet 中的图片。调用会把结构化输出和 token 审计写入新目录：

```bash
conda run --no-capture-output -n meowbench python -m ttt_frame.codex_teacher \
  --packet runs/teacher_pilot/packet \
  --out runs/teacher_pilot/codex_new \
  --model gpt-5.5 --effort medium
```

`codex` 必须先显示 `Logged in using ChatGPT`。这条路径已经处理过 `clip-0cc0d7aed02ac9242576edd6.mp4`，结果保存在 `runs/teacher_pilot/codex/analysis.json`，因此重复评测时直接复用，不再消耗订阅额度。

## OpenAI-compatible API teacher

安装 API 客户端并把 token 放入环境变量。不要把 token 写在命令行、代码、Git 或 `memory.json` 中：

```bash
conda run --no-capture-output -n meowbench python -m pip install -e '.[teacher]'
export TTT_TEACHER_API_KEY='你的token'
export TTT_TEACHER_BASE_URL='https://starwithcoding.com'
```

裸 origin 会自动补成 `https://starwithcoding.com/v1`；如果服务商给出的文档明确要求其他路径，直接把完整路径写进 `TTT_TEACHER_BASE_URL`。使用同一 60 秒视频的完整 packet，一次请求分析全部 24 帧：

```bash
VIDEO=/data/quzitsix/meow-releases/supermemory-pilot-v2/media/clip-0cc0d7aed02ac9242576edd6.mp4
conda run --no-capture-output -n meowbench python -m ttt_frame.spatial_videoqa export-teacher \
  --video "$VIDEO" --out runs/teacher_pilot/packet_api \
  --chunk-seconds 30 --frames-per-chunk 12 --max-side 448
conda run --no-capture-output -n meowbench python -m ttt_frame.api_teacher \
  --packet runs/teacher_pilot/packet_api --out runs/teacher_pilot/api \
  --base-url "$TTT_TEACHER_BASE_URL" --model 你的模型名 \
  --response-mode json_schema --max-tokens 8000
```

`run.json` 保存 endpoint host、模型、耗时和服务端返回的 token 数，不保存 token；`json_schema` 被网关拒绝时可以明确改成 `--response-mode json_object`，仍会经过本地严格校验。默认不重试，避免在用量受限时重复计费。随后用 API 分析训练 LoRA：

```bash
conda run --no-capture-output -n meowbench python -m ttt_frame.videoqa teacher-ingest \
  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct --local-files-only \
  --device cuda:0 --dtype bfloat16 --packet runs/teacher_pilot/packet_api \
  --analysis runs/teacher_pilot/api/analysis.json --rank 8 --lora-alpha 16 \
  --steps-per-chunk 2 --max-length 768 --learning-rate 3e-4 \
  --save runs/teacher_pilot/lora_api
```

这条命令按 packet 的时间段逐段更新 LoRA，使 API teacher 与本地 teacher 使用相同的 `steps-per-chunk`。它不会把原始视频、帧或 teacher 文本放入 checkpoint。

## 导入与规范化

```bash
conda run --no-capture-output -n meowbench python -m ttt_frame.spatial_videoqa import-teacher \
  --packet /path/to/teacher_packet --analysis /path/to/chatgpt.json \
  --save /path/to/teacher_canonical.json
```

导入会验证 schema、packet/source SHA256、chunk 与 frame 引用、时间顺序、事件区间和字段大小；失败时不会生成输出。canonical JSON 同时保存原结构与 `canonical_text`，供后续 `SpatialVideoMemory.ingest_teacher_text`（或离线训练程序）写入 fast weights。导出的图片和原始 teacher 文本不属于参数 memory checkpoint，需单独保存审计。

Teacher JSON 顶层字段为 `schema`、`packet_manifest_sha256`、`source_sha256`、`segments` 和 `global_summary`。每个 segment 应包含 `chunk_index`、时间范围、`frame_refs`、`summary`、`observations`、`entities`、`events`、`relations` 和 `uncertainty`。所有描述必须是帧中可见证据；不确定内容写入 `uncertainty`，不要猜测。
