#!/usr/bin/env python3
"""Build a Markdown comparison report for the FHO-STA smoke runs.

The evaluator writes one JSON object per condition/task and a summary JSON.  This
utility consumes either the JSONL path, a summary path, or a run directory and
creates a reproducible report comparing ICL with Spatial-TTT.  It never fills
missing scores with guesses: if no input run exists, the report says that the
experiment is waiting for a data run and prints the command to run it.

Examples
--------

    python scripts/report_fho_sta_results.py \
      --run smoke16=runs/fho_sta_smoke_16.jsonl \
      --output docs/FHO_STA_EXPERIMENT_A_RESULTS.md

    python scripts/report_fho_sta_results.py \
      --run dense=runs/fho_sta_dense_448.jsonl \
      --run sparse=runs/fho_sta_sparse_448.jsonl \
      --max-side 448 --horizons 0.5 1 1.5 2
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "FHO_STA_EXPERIMENT_A_RESULTS.md"
CONDITION_ORDER = (
    "blind",
    "icl_4f",
    "icl_8f",
    "icl_8s4f",
    "spatial_4s8f",
    "spatial_8s4f",
)
PAIRINGS = {
    "spatial_4s8f": "icl_8f",
    "spatial_8s4f": "icl_8s4f",
}
TASK_ORDER = ("noun", "verb")


@dataclass
class Run:
    label: str
    source: Path
    rows: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    aggregate: dict[str, Any] = field(default_factory=dict)

    @property
    def max_side(self) -> Any:
        for key in ("max_side", "image_max_side"):
            value = self.metadata.get(key)
            if value is not None:
                return value
        # JSONL rows from newer evaluators carry the setting even when the
        # companion summary was not copied alongside the predictions.
        values = {row.get("max_side") for row in self.rows if row.get("max_side") is not None}
        if len(values) == 1:
            return next(iter(values))
        return "unknown"


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    # Accept paths relative to the invocation directory and repository root.
    return candidate if candidate.exists() else ROOT / candidate


def _find_sources(path: Path) -> tuple[Path | None, Path | None]:
    """Return (jsonl, summary) for a file or an evaluator run directory."""

    if path.is_dir():
        summary = path / "summary.json"
        jsonls = sorted(path.glob("*.jsonl"))
        return (jsonls[0] if jsonls else None), (summary if summary.is_file() else None)
    if path.suffix.lower() == ".jsonl":
        summary = path.with_name("summary.json")
        return path, (summary if summary.is_file() else None)
    if path.name == "summary.json" or path.suffix.lower() == ".json":
        return None, path
    return None, None


def _normalise_row(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict) or not row.get("condition") or not row.get("task"):
        return None
    status = row.get("status", "ok")
    correct = row.get("correct")
    if status == "ok" and isinstance(correct, bool):
        value = correct
    elif status == "ok" and isinstance(correct, (int, float)):
        value = bool(correct)
    else:
        value = False
    result = dict(row)
    result["status"] = str(status)
    result["correct"] = value
    result["condition"] = str(row["condition"])
    result["task"] = str(row["task"])
    result["uid"] = str(row.get("uid", row.get("video_uid", "")))
    result["horizon_bin"] = row.get("horizon_bin", "unknown")
    return result


def _load_rows(jsonl: Path | None) -> list[dict[str, Any]]:
    if jsonl is None or not jsonl.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with jsonl.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {jsonl}:{line_number}: {exc}") from exc
            row = _normalise_row(item)
            if row is not None:
                rows.append(row)
    return rows


def _load_run(label: str, source: Path) -> Run:
    jsonl, summary_path = _find_sources(source)
    metadata: dict[str, Any] = {}
    aggregate: dict[str, Any] = {}
    if summary_path is not None and summary_path.is_file():
        payload = _json(summary_path)
        if isinstance(payload, dict):
            metadata = dict(payload)
            aggregate = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
            # Evaluator summary stores its JSONL path relative to the command
            # directory. Resolve it relative to the summary's parent as well.
            pointer = payload.get("predictions_jsonl")
            if jsonl is None and isinstance(pointer, str):
                raw_pointer = Path(pointer)
                candidates = [raw_pointer]
                if not raw_pointer.is_absolute():
                    candidates.extend((summary_path.parent / raw_pointer, ROOT / raw_pointer))
                pointed = next((item for item in candidates if item.is_file()), None)
                if pointed is not None:
                    jsonl = pointed
    rows = _load_rows(jsonl)
    run = Run(label=label, source=source, rows=rows, metadata=metadata, aggregate=aggregate)
    # A summary still contains useful n/correct counts and horizon buckets, but
    # cannot support UID-paired or joint-action metrics.  Represent those
    # aggregates as rows so the regular Markdown tables can display them while
    # retaining the status marker for the paired-metric guard.
    if not run.rows and run.aggregate:
        run.rows = list(_summary_rows(run))
    return run


def _summary_rows(run: Run) -> Iterable[dict[str, Any]]:
    """Expand an evaluator summary when the per-question JSONL is unavailable."""

    # This expansion is intentionally marked aggregate_only; it supports
    # horizon tables but cannot support paired item/action statistics.
    for key, metric in run.aggregate.items():
        if not isinstance(metric, dict) or "::" not in key:
            continue
        condition, task = key.split("::", 1)
        horizons = metric.get("horizons", {})
        if isinstance(horizons, dict) and horizons:
            for horizon, value in horizons.items():
                if not isinstance(value, dict):
                    continue
                yield {"condition": condition, "task": task, "horizon_bin": horizon,
                       "status": "aggregate", "correct": value.get("correct", 0),
                       "n": value.get("n", 0)}
        elif metric.get("n") is not None:
            yield {"condition": condition, "task": task, "horizon_bin": "all",
                   "status": "aggregate", "correct": metric.get("correct", 0),
                   "n": metric.get("n", 0)}


def _group_rows(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["condition"]), str(row["task"]), str(row.get("horizon_bin", "unknown")))].append(row)
    return groups


def _metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok"]
    aggregate = [row for row in rows if row.get("status") == "aggregate"]
    if not valid and aggregate:
        n = sum(int(row.get("n", 0) or 0) for row in aggregate)
        correct = sum(int(row.get("correct", 0) or 0) for row in aggregate)
        return {"n": n, "correct": correct,
                "accuracy": correct / n if n else None,
                "errors": 0, "mean_latency": None}
    errors = len(rows) - len(valid)
    correct = sum(int(bool(row.get("correct"))) for row in valid)
    elapsed = [float(row["elapsed_sec"]) for row in valid if isinstance(row.get("elapsed_sec"), (int, float))]
    return {"n": len(valid), "correct": correct,
            "accuracy": correct / len(valid) if valid else None,
            "errors": errors,
            "mean_latency": sum(elapsed) / len(elapsed) if elapsed else None}


def _format(value: Any, digits: int = 3) -> str:
    if value is None or value == "unknown":
        return "—" if value is None else str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _horizon_sort(value: str) -> tuple[int, float | str]:
    try:
        return (0, float(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _select_horizons(rows: list[dict[str, Any]], requested: list[float] | None) -> list[str]:
    values = {str(row.get("horizon_bin", "unknown")) for row in rows}
    if requested:
        wanted = {str(v).rstrip("0").rstrip(".") for v in requested}
        # Keep both `1` and `1.0` spellings when generated by old runs.
        selected = [value for value in values if value in wanted or value.removesuffix(".0") in wanted]
    else:
        selected = list(values)
    return sorted(selected, key=_horizon_sort)


def _action_metrics(rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok" or not row.get("uid"):
            continue
        key = (str(row["condition"]), str(row.get("horizon_bin", "unknown")), str(row["uid"]))
        by_key[key][str(row["task"])] = bool(row.get("correct"))
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for (condition, horizon, _uid), tasks in by_key.items():
        if not all(task in tasks for task in TASK_ORDER):
            continue
        item = result.setdefault((condition, horizon), {"n": 0, "correct": 0})
        item["n"] += 1
        item["correct"] += int(all(tasks[task] for task in TASK_ORDER))
    for item in result.values():
        item["accuracy"] = item["correct"] / item["n"] if item["n"] else None
    return result


def _paired(rows: list[dict[str, Any]], ttt: str, icl: str, horizon: str | None = None, task: str | None = None) -> dict[str, Any]:
    lookup: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    for row in rows:
        if row.get("status") != "ok" or (task and row.get("task") != task):
            continue
        if horizon is not None and str(row.get("horizon_bin")) != horizon:
            continue
        uid = str(row.get("uid", ""))
        if not uid:
            continue
        lookup[(uid, str(row.get("task", "")))][str(row["condition"])] = bool(row.get("correct"))
    pairs = []
    for values in lookup.values():
        if ttt in values and icl in values:
            pairs.append((values[ttt], values[icl]))
    if not pairs:
        return {"n": 0, "ttt_correct": 0, "icl_correct": 0, "delta": None,
                "wins": 0, "ties": 0, "losses": 0}
    ttt_correct = sum(int(a) for a, _ in pairs)
    icl_correct = sum(int(b) for _, b in pairs)
    wins = sum(int(a and not b) for a, b in pairs)
    losses = sum(int(b and not a) for a, b in pairs)
    return {"n": len(pairs), "ttt_correct": ttt_correct, "icl_correct": icl_correct,
            "delta": (ttt_correct - icl_correct) / len(pairs), "wins": wins,
            "ties": len(pairs) - wins - losses, "losses": losses}


def render(runs: list[Run], *, output: Path, max_side: int | None, horizons: list[float] | None) -> str:
    all_rows = [row for run in runs for row in run.rows]
    status = "已获得结果" if any(row.get("status") == "ok" for row in all_rows) else (
        "已获得聚合结果" if all_rows else "待运行"
    )
    lines = [
        "# 实验 A：FHO-STA 短期预测（ICL 与 Spatial-TTT）",
        "",
        f"状态：**{status}**。本文件由 `scripts/report_fho_sta_results.py` 生成。",
        "",
        "本实验只使用观测边界之前的帧。ICL 将前缀保留在问题上下文中；Spatial-TTT 按时间顺序写入 fast weights，提问阶段丢弃视频后只读取参数状态。指标是封闭集合 noun/verb 选择题准确率，不能等同于官方 FHO-STA detector mAP。",
        "",
        "## 实验设置",
        "",
        f"- 图像最长边（`max_side`）：{max_side if max_side is not None else '由各 run 的元数据记录；旧结果若没有该字段则显示 unknown'}。",
        "- ICL/TTT 必须使用相同观测边界和帧预算；`spatial_4s8f` 对比 `icl_8f`，`spatial_8s4f` 对比 `icl_8s4f`。",
        "- 预测距离按 FHO 标注的 `horizon_bin` 分层。只在真实运行产生结果后填写数值。",
        "",
        "## Run 状态",
        "",
        "| run | 输入 | max_side | items/rows | model | 状态 |",
        "|---|---|---:|---:|---|---|",
    ]
    if not runs:
        lines.append("| — | — | — | — | — | 待运行：没有提供 `--run LABEL=PATH` |")
    else:
        for run in runs:
            n_items = run.metadata.get("n_items", "unknown")
            n_rows = run.metadata.get("n_rows", len(run.rows) or "unknown")
            model = run.metadata.get("model_path", "unknown")
            state = "可汇总" if run.rows or run.aggregate else "未找到结果文件"
            side = run.max_side if run.max_side != "unknown" else (max_side if max_side is not None else "unknown")
            lines.append(f"| {run.label} | `{run.source}` | {_format(side)} | {n_items}/{n_rows} | `{model}` | {state} |")
    lines += ["", "## Overall noun/verb accuracy", "", "| run | condition | task | n | correct | accuracy | errors | mean latency (s) |", "|---|---|---|---:|---:|---:|---:|---:|"]
    if not all_rows:
        lines.append("| — | — | — | — | — | — | — | — |")
    else:
        for run in runs:
            rows = run.rows
            groups = _group_rows(rows)
            conditions = sorted({key[0] for key in groups}, key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else len(CONDITION_ORDER))
            for condition in conditions:
                for task in TASK_ORDER:
                    selected = [r for (c, t, _h), rs in groups.items() if c == condition and t == task for r in rs]
                    metric = _metric(selected)
                    lines.append(f"| {run.label} | {condition} | {task} | {metric['n']} | {metric['correct']} | {_format(metric['accuracy'])} | {metric['errors']} | {_format(metric['mean_latency'])} |")
    lines += ["", "## Accuracy by horizon", "", "| run | horizon | condition | noun | verb | action joint | n (noun/verb) |", "|---|---:|---|---:|---:|---:|---:|"]
    if not all_rows:
        lines.append("| — | — | — | — | — | — | — |")
    else:
        for run in runs:
            rows = run.rows
            hs = _select_horizons(rows, horizons)
            conditions = sorted({str(r["condition"]) for r in rows}, key=lambda c: CONDITION_ORDER.index(c) if c in CONDITION_ORDER else len(CONDITION_ORDER))
            actions = _action_metrics(rows)
            for h in hs:
                for condition in conditions:
                    noun = _metric([r for r in rows if r["condition"] == condition and r["task"] == "noun" and str(r.get("horizon_bin")) == h])
                    verb = _metric([r for r in rows if r["condition"] == condition and r["task"] == "verb" and str(r.get("horizon_bin")) == h])
                    action = actions.get((condition, h), {"n": 0, "accuracy": None})
                    lines.append(f"| {run.label} | {h} | {condition} | {_format(noun['accuracy'])} | {_format(verb['accuracy'])} | {_format(action.get('accuracy'))} | {noun['n']}/{verb['n']} |")
    lines += ["", "## Paired Spatial-TTT minus ICL", "", "| run | TTT | matched ICL | task | horizon | pairs | TTT acc | ICL acc | Δ | wins/ties/losses |", "|---|---|---|---|---:|---:|---:|---:|---:|---|"]
    if not all_rows:
        lines.append("| — | — | — | — | — | — | — | — | — | — |")
    else:
        for run in runs:
            rows = run.rows
            hs = _select_horizons(rows, horizons)
            for ttt, icl in PAIRINGS.items():
                for task in TASK_ORDER:
                    for h in ["all", *hs]:
                        p = _paired(rows, ttt, icl, None if h == "all" else h, task)
                        if not p["n"]:
                            continue
                        lines.append(f"| {run.label} | {ttt} | {icl} | {task} | {h} | {p['n']} | {_format(p['ttt_correct']/p['n'])} | {_format(p['icl_correct']/p['n'])} | {_format(p['delta'])} | {p['wins']}/{p['ties']}/{p['losses']} |")
    lines += ["", "## 解释边界", "", "- 只有在相同帧数、分辨率、观测边界下的 paired Δ 才用于判断 TTT 是否优于 ICL；不同窗口只能作为采样预算分析。", "- `action joint` 要求同一 UID 的 noun 与 verb 都答对；若只有 summary 没有逐题 JSONL，则无法计算该项和 paired Δ。", "- 小样本结果只用于 smoke test。不要将 closed-set accuracy 写成官方 FHO-STA mAP。", "", "## 复现实验", "", "```bash", "conda run --no-capture-output -n meowbench python scripts/evaluate_fho_sta.py \\", "  --annotations \"$EGO4D_DIR/fho_sta_subset_16/fho_sta_val_subset.json\" \\", "  --video-root \"$EGO4D_DIR/v2/clips\" --frame-field clip_frame \\", "  --model-path /data/quzitsix/models/Qwen3-VL-2B-Instruct \\", "  --spatial-checkpoint /data/quzitsix/models/Spatial-TTT-nano/model.safetensors \\", "  --device cuda:2 --dtype bfloat16 --limit 16 --max-new-tokens 8 \\", "  --output runs/fho_sta_smoke_16.jsonl", "", "python scripts/report_fho_sta_results.py \\", "  --run smoke16=runs/fho_sta_smoke_16.jsonl \\", "  --max-side 448 \\", "  --output docs/FHO_STA_EXPERIMENT_A_RESULTS.md", "```", ""]
    if not all_rows:
        lines += ["当前没有可读取的 FHO-STA JSONL/summary 结果。配置 Ego4D 凭据并运行上面的评测命令后，再重新执行报告命令；本次输出没有填入任何估计分数。", ""]
    return "\n".join(lines)


def _parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = _resolve(spec)
        return path.stem, path
    label, raw = spec.split("=", 1)
    if not label.strip() or not raw.strip():
        raise ValueError(f"invalid --run {spec!r}; expected LABEL=PATH")
    return label.strip(), _resolve(raw.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", default=[], metavar="LABEL=PATH",
                        help="evaluator JSONL, summary.json, or run directory; repeatable")
    parser.add_argument("--max-side", type=int, default=None,
                        help="record the image longest-side budget in the report; metadata still takes precedence")
    parser.add_argument("--horizons", type=float, nargs="+", default=None,
                        help="only include these horizon bins in horizon/paired tables")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    runs: list[Run] = []
    for spec in args.run:
        label, path = _parse_run_spec(spec)
        runs.append(_load_run(label, path))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(runs, output=args.output, max_side=args.max_side, horizons=args.horizons), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "runs": len(runs), "rows": sum(len(run.rows) for run in runs), "status": "ready" if any(run.rows for run in runs) else "waiting_for_runs"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
