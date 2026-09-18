# Third-party notices

## Spatial-TTT

The Spatial-TTT implementation in this project is adapted from the mathematical
update and Qwen3-VL integration in:

- Project: [THU-SI/Spatial-TTT](https://github.com/THU-SI/Spatial-TTT)
- Authors: Fangfu Liu, Diankun Wu, Jiawei Chi, Yimo Cai, Yi-Hsin Hung,
  Xumin Yu, Hao Li, Han Hu, Yongming Rao, and Yueqi Duan.
- Reference commit: `e2e33a62b6f92c33b7e24ff042be9737d05c5bdb`
- Reference files: `qwen-vl-finetune/models/ttt_operation.py`,
  `qwen-vl-finetune/models/causal_swa_lact.py`,
  `qwen-vl-finetune/models/causal_swa_lact_streaming_chunked.py`, and
  `qwen-vl-finetune/models/spatial_ttt.py`.
- Paper: [Spatial-TTT: Streaming Visual-based Spatial Intelligence with
  Test-Time Training](https://arxiv.org/abs/2603.12255), 2026.
- License: Apache License, Version 2.0, as provided by the reference commit's
  [`LICENSE`](https://github.com/THU-SI/Spatial-TTT/blob/e2e33a62b6f92c33b7e24ff042be9737d05c5bdb/LICENSE).
  An unmodified copy is included at
  [LICENSES/Spatial-TTT-Apache-2.0.txt](LICENSES/Spatial-TTT-Apache-2.0.txt).
  The upstream README badge says MIT, but its actual license file is Apache 2.0;
  this notice preserves the actual file.

Local adaptations (2026-09-14): a portable PyTorch implementation of the
SwiGLU fast-weight update; explicit write/read/reset and serialization of
long-lived parameter memory; integration with the locally installed Qwen3-VL;
and a video ingestion/query lifecycle that discards historical attention caches
and keeps questions from changing video memory. These are project adaptations,
not the unmodified official implementation or an assertion of reproduced
benchmark results. See [docs/SPATIAL_TTT.md](docs/SPATIAL_TTT.md).

## LaCT

Spatial-TTT's upstream `causal_swa_lact.py` acknowledges adaptation from
[a1600012888/LaCT](https://github.com/a1600012888/LaCT), specifically
`lact_llm/lact_model/layer_lact_swiglu.py`. This project's earlier LaCT
implementation also uses that project.

Copyright (c) 2025 Tianyuan Zhang, Hao Tan.

The MIT license is retained at [ttt_frame/LICENSE.LaCT](ttt_frame/LICENSE.LaCT).

The upstream Newton–Schulz helper also acknowledges
[KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
and [MoonshotAI/Moonlight](https://github.com/MoonshotAI/Moonlight) for its
Muon implementation and adaptation to batched matrices. Those attributions
are preserved here.
