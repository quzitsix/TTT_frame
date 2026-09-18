# 单片段 teacher pilot

本次对照使用 `supermemory-pilot-v2` 的 Q9 证据片段：

```text
/data/quzitsix/meow-releases/supermemory-pilot-v2/media/clip-0cc0d7aed02ac9242576edd6.mp4
```

它是一个完整的 60 秒切片，使用 24 张按时间排序的帧（两段，每段 30 秒）。评测问题是：

```text
I'm prepping food at the island. Where did I throw the empty red mesh bag earlier?
A. Into the white trash can near the kitchen island.
B. Into the recycling bin next to the refrigerator.
C. Into an unspecified disposal container in the kitchen.
D. This question cannot be answered.
```

release 的金标是 A。当前已经实际运行的输出如下：

| 路径 | teacher / 记忆 | 输出 | 解释 |
|---|---|---|---|
| frozen base | 无写入 | D | 基础模型没有可靠读出该事件 |
| local LoRA v2 | 本地 Qwen 观察 + 临时 QA，2 steps/segment | C | 写入了“容器”类描述，但没有保留白色垃圾桶这一细节 |
| Codex CLI LoRA v2 | ChatGPT 登录的 Codex CLI，24 帧，2 steps/segment | D | teacher 对容器用途保持不确定，LoRA 没有改变该题读出 |
| Spatial smoke | Spatial-TTT，未加载训练 checkpoint，`ttt_scale_init=0.1` | D | 证明视觉 token→fast weight 路径可执行，不是能力评测 |

这个结果暂时不能证明外部 teacher 提高了回答质量。视频中 17.5 秒附近能看到一个带白色内衬的容器，但抽样帧不能直接确认“扔入”动作已完成；Codex teacher 因此保留了不确定性。后续 API teacher 会使用同一 packet 和同一 LoRA 配置，避免把采样密度、训练步数和问题换掉。

API teacher 的输出目录应为 `runs/teacher_pilot/api`，训练后的参数目录应为 `runs/teacher_pilot/lora_api`。对照命令见 [`TEACHER_BRIDGE.md`](TEACHER_BRIDGE.md)。
