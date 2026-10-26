#!/usr/bin/env python3
"""Post-scoring comparison of frozen MCQ answers, including literal option text.

This is an analysis tool, never an inference or hyperparameter-selection tool.
All requested arms must cover exactly the fixed split. Examples are deliberately
selected after scoring by a documented deterministic rule; they are not a new test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected JSON objects: {path}")
    return rows


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def predictions(path: Path, key: str | None, expected_ids: set[str]) -> dict[str, str]:
    result = {}
    for row in read_jsonl(path):
        sid = row["id"]
        if "arm" in row and key is not None and row["arm"] != key:
            continue
        if sid in result:
            raise ValueError(f"duplicate prediction ID {sid}: {path}")
        if "predictions" in row:
            if key is None:
                raise ValueError("multi-arm prediction files require an explicit key")
            value = row["predictions"][key]
            if isinstance(value, dict):
                value = value["prediction"]
        else:
            value = row["prediction"]
        if value not in tuple("ABCDEF"):
            raise ValueError(f"invalid prediction for {sid}: {value!r}")
        result[sid] = value
    if set(result) != expected_ids:
        raise ValueError(f"incomplete or extra prediction IDs: {path}")
    return result


def select_examples(rows: list[dict], arms: list[str], baseline: str,
                    comparisons: list[tuple[str, str]], limit: int) -> list[dict]:
    """First sorted ID per paired outcome, then both-correct/all-wrong cases."""
    selected: dict[str, list[str]] = {}
    ordered = sorted((row for row in rows if row["research_split"] == "test"), key=lambda r: r["id"])

    def take(candidates: list[dict], reason: str) -> None:
        if not candidates:
            return
        chosen = next((row for row in candidates if row["id"] not in selected), candidates[0])
        if chosen["id"] in selected or len(selected) < limit:
            selected.setdefault(chosen["id"], []).append(reason)

    for arm, control in comparisons:
        take([r for r in ordered if r["predictions"][arm]["correct"]
              and not r["predictions"][control]["correct"]], f"{arm} 改对 {control} 的错误")
        take([r for r in ordered if not r["predictions"][arm]["correct"]
              and r["predictions"][control]["correct"]], f"{arm} 改错 {control} 的正确答案")
    take([r for r in ordered if all(r["predictions"][a]["correct"] for a in arms)], "全部展示方法均正确")
    take([r for r in ordered if all(not r["predictions"][a]["correct"] for a in arms)], "全部展示方法均错误")
    if len(selected) < limit:
        take([r for r in ordered if r["predictions"][baseline]["correct"]], f"{baseline} 正确的补充例子")
    by_id = {r["id"]: r for r in rows}
    return [{**by_id[sid], "selection_reasons": reasons} for sid, reasons in selected.items()]


def escape(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("choose a new analysis output directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = Path(config["benchmark_root"])
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("analysis must not write into benchmark data")
    split_path = Path(config["split_manifest"])
    split = json.loads(split_path.read_text())
    assignments = [(sid, role) for role in ("train", "validation", "test") for sid in split["splits"][role]]
    ids = {sid for sid, _ in assignments}
    if len(ids) != len(assignments):
        raise ValueError("split IDs overlap")
    input_path, answer_path = root / "inputs_mcq.jsonl", root / "answers.jsonl"
    public_rows, answer_rows = read_jsonl(input_path), read_jsonl(answer_path)
    public = {row["id"]: row["input"] for row in public_rows}
    answers = {row["id"]: row for row in answer_rows}
    if len(public) != len(public_rows) or len(answers) != len(answer_rows) or set(public) != ids or set(answers) != ids:
        raise ValueError("public inputs, answers and split must have identical unique IDs")
    arms = [arm["name"] for arm in config["arms"]]
    if not arms or len(arms) != len(set(arms)):
        raise ValueError("need unique nonempty arm names")
    values = {}
    for arm in config["arms"]:
        path = Path(arm["path"])
        if arm.get("sha256") is not None and record(path)["sha256"] != arm["sha256"]:
            raise ValueError(f"frozen prediction checksum mismatch: {arm['name']}")
        values[arm["name"]] = predictions(path, arm.get("key"), ids)
    rows = []
    for sid, role in assignments:
        options = public[sid]["options"]
        options_by_label = {x["label"]: x["text"] for x in options}
        answer = answers[sid]
        if set(options_by_label) != set("ABCDEF") or len(options) != 6:
            raise ValueError("invalid public options")
        private_options = {x["label"]: x for x in answer["options"]}
        if {k: v["text"] for k, v in private_options.items()} != options_by_label:
            raise ValueError("answer option order/text disagrees with public input")
        correct = answer["correct_answer"]
        if correct not in options_by_label or answer["answer"] != options_by_label[correct]:
            raise ValueError("answer label/text mismatch")
        rows.append({"id": sid, "research_split": role, "day": split["id_to_day"][sid],
            "question": public[sid]["question"], "options": options,
            "reference": {"label": correct, "text": options_by_label[correct],
                          "category": answer.get("answer_category"), "silver_label": True},
            "predictions": {arm: {"label": values[arm][sid], "text": options_by_label[values[arm][sid]],
                                  "correct": values[arm][sid] == correct,
                                  "released_option_role": private_options[values[arm][sid]].get("role")}
                            for arm in arms}})
    metrics = {}
    for arm in arms:
        metrics[arm] = {}
        for role in ("train", "validation", "test"):
            subset = [r for r in rows if r["research_split"] == role]
            n_correct = sum(r["predictions"][arm]["correct"] for r in subset)
            metrics[arm][role] = {"n": len(subset), "correct": n_correct,
                "accuracy": n_correct / len(subset) if subset else None,
                "incorrect_released_option_roles": dict(Counter(r["predictions"][arm]["released_option_role"]
                    for r in subset if not r["predictions"][arm]["correct"]))}
    baseline = config["baseline"]
    comparisons = [tuple(pair) for pair in config.get("comparisons", [])]
    if baseline not in arms or any(a not in arms or b not in arms for a, b in comparisons):
        raise ValueError("unknown comparison arm")
    paired = {}
    for arm, control in dict.fromkeys([(a, baseline) for a in arms if a != baseline] + comparisons):
        key = f"{arm} vs {control}"
        paired[key] = {}
        for role in ("train", "validation", "test"):
            subset = [r for r in rows if r["research_split"] == role]
            gains = [r["id"] for r in subset if r["predictions"][arm]["correct"] and not r["predictions"][control]["correct"]]
            losses = [r["id"] for r in subset if r["predictions"][control]["correct"] and not r["predictions"][arm]["correct"]]
            paired[key][role] = {"gains": len(gains), "losses": len(losses), "net": len(gains)-len(losses),
                                 "gain_ids": gains, "loss_ids": losses}
    example_arms = config.get("example_arms", arms)
    if not example_arms or any(a not in arms for a in example_arms):
        raise ValueError("invalid example arms")
    examples = select_examples(rows, example_arms, baseline, comparisons, config.get("example_limit", 10))
    summary = {"schema": "ideax_answer_comparison_v1", "analysis_only": True,
               "selection_is_posthoc": True, "examples_are_generated_rationales": False,
               "metrics": metrics, "paired": paired, "example_ids": [x["id"] for x in examples],
               "example_rule": "For each configured comparison: first lexicographic unused test ID for gain and loss; then all-correct/all-wrong; no inference or tuning uses these examples.",
               "limitations": ["Reused exploratory test57, silver labels, overlapping windows, two test days.",
                    "Answer text is the literal selected MCQ option, not a freely generated explanation.",
                    "Released distractor roles are metadata judgements, not independently human-verified error types.",
                    "Methods have different inputs, checkpoints, supervision, state and compute; consult individual reports."]}
    lines = ["# Idea 实验：具体回答对照", "", "这是所有预测冻结并评分后的诊断。答案文字来自模型选择的原始选项，",
             "不是模型自由生成的解释；不根据例子修改设置。银标签仅指一个参考事件，不能穷尽未来。", "",
             "## 同一题集的结果", "", "| 方法 | Validation | Test | Test准确率 |", "|---|---:|---:|---:|"]
    for arm in arms:
        v, t = metrics[arm]["validation"], metrics[arm]["test"]
        lines.append(f"| {escape(arm)} | {v['correct']}/{v['n']} | {t['correct']}/{t['n']} | {t['accuracy']:.2%} |")
    lines += ["", "方法预算与实现范围不同，不能直接把此表解释为架构排名。", "", "## 具体题目", "",
              "例子按配置中的配对比较，依次选字典序最早、尚未展示的改对/改错测试题，再补全部正确/错误例子。",
              "这属于明确按结果选例的事后分析，完整200题见 `answers.jsonl` / `answers.csv`，配对ID见 `summary.json`。", ""]
    for example in examples:
        sid = example["id"]
        lines += [f"### {sid}（第{example['day']}天）", "", "选择原因：" + "；".join(example["selection_reasons"]), "",
                  "问题：" + example["question"], "", "**银标签参考：" + example["reference"]["label"] + "．" + example["reference"]["text"] + "**", "",
                  "| 选项 | 原文 |", "|---|---|"]
        lines.extend(f"| {o['label']} | {escape(o['text'])} |" for o in example["options"])
        lines += ["", "| 方法 | 实际选择的答案 | 与参考一致 |", "|---|---|---|"]
        for arm in example_arms:
            pred = example["predictions"][arm]
            lines.append(f"| {escape(arm)} | {pred['label']}．{escape(pred['text'])} | {'是' if pred['correct'] else '否'} |")
        lines.append("")
    output.mkdir(parents=True)
    for name, value in [("summary.json", summary), ("examples.json", examples), ("config.json", config)]:
        (output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (output/"answers.jsonl").write_text("".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n" for row in rows),encoding="utf-8")
    with (output/"answers.csv").open("w",encoding="utf-8",newline="") as stream:
        writer = csv.writer(stream,lineterminator="\n")
        writer.writerow(["id","research_split","day","reference_label","reference_text"] + [f"{a}_{field}" for a in arms for field in ("label","text","correct")])
        for row in rows:
            writer.writerow([row["id"],row["research_split"],row["day"],row["reference"]["label"],row["reference"]["text"]] +
                            [row["predictions"][a][field] for a in arms for field in ("label","text","correct")])
    (output/"concrete_answers.md").write_text("\n".join(lines),encoding="utf-8")
    manifest = {"analysis_only": True, "sources": [record(path) for path in dict.fromkeys([config_path,input_path,answer_path,split_path,Path(__file__)] + [Path(a["path"]) for a in config["arms"]])],
                "files": {p.name: record(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    result=build(args.config,args.output_dir)
    print(json.dumps({"examples":result["example_ids"],"test":{a:m["test"] for a,m in result["metrics"].items()}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
