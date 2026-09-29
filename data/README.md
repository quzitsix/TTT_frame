# Codex-derived evaluation questions

`q9_codex_mcq.jsonl` contains 19 small multiple-choice questions generated from
the saved Codex scene analyses for the `supermemory-9` kitchen recording. The
questions cover relative location, object relationships, and short temporal
orders. Correct letters are balanced across the four option positions
(`A=5, B=5, C=5, D=4`) so a fixed-letter answer is an explicit baseline.

Four rows are marked with `long_range_*` axes. They combine evidence from
different 60-second clips and are reported separately from local single-clip
questions when interpreting long-term memory.

The memory environment is the full 18-clip recording (`0--1080` seconds). Each
row records the Codex source file, source clip, and local/global evidence span.
Most rows use the 18-clip `analyses_sol` outputs; the more detailed red-onion
rows use `runs/teacher_pilot/codex_dense/analysis.json`, whose source clip is
`clip-0cc0d7aed02ac9242576edd6` (`240--300` seconds globally).

The Codex analyses are sampled-frame observations, not a continuous human
annotation. The questions therefore avoid uncertain identities and avoid
claiming that an action completed when the frames only show an object being
held or positioned. `answer` and `evidence` are evaluation metadata and must
not be included in the prompt sent to a model.
