#!/usr/bin/env python3
"""Export the complete, exploratory Idea 01 scaling comparison as PNG/SVG.

Uses scored aggregate metrics and public per-episode forecast diagnostics; it
does not read benchmark answers, videos, or candidate scores. Matplotlib is an
optional reporting dependency, not required by the experiment itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_plot_data(run_dir: Path) -> dict:
    import torch

    plan = json.loads((run_dir / "plan.json").read_text(encoding="utf-8"))
    aggregate = json.loads((run_dir / "mcq/report/aggregate_metrics.json").read_text(encoding="utf-8"))
    report = aggregate["by_head"][aggregate["primary_head"]]
    split = json.loads(Path(plan["split_manifest"]).read_text(encoding="utf-8"))
    test_ids = split["splits"]["test"]
    if not test_ids:
        raise ValueError("test split cannot be empty")
    trials = []
    for trial in plan["trials"]:
        name = trial["name"]
        row = {"name": name, "configuration": trial["configuration"]}
        for arm in ("residual", "residual_no_update"):
            score = report["by_arm"][f"{name}__{arm}"]["by_research_split"]["test"]
            if score["n"] != len(test_ids) or score["valid"] != len(test_ids):
                raise ValueError(f"incomplete test predictions: {name}/{arm}")
            row[arm] = {"correct": score["correct"], "mse": []}
        for sample_id in test_ids:
            artifact = torch.load(Path(trial["forecast_dir"]) / f"{sample_id}_forecasts.pt",
                                  map_location="cpu", weights_only=True)
            if artifact["sample_id"] != sample_id or artifact["trial"] != name:
                raise ValueError(f"forecast identity mismatch: {name}/{sample_id}")
            for arm in ("residual", "residual_no_update"):
                stats = artifact["stats"][arm]
                if stats["mature_count"] <= 0 or stats["mature_mse"] is None:
                    raise ValueError(f"missing matured prediction diagnostic: {name}/{sample_id}")
                row[arm]["mse"].append(stats["mature_mse"])
        for arm in ("residual", "residual_no_update"):
            row[arm]["mse"] = sum(row[arm]["mse"]) / len(test_ids)
        trials.append(row)
    pretraining = {"by_width": {}}
    for width in plan["widths"]:
        directory = run_dir / "pretraining" / f"h{width}"
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
        if (summary.get("complete") is not True
                or protocol["protocol"]["config"]["hidden_dim"] != width
                or summary["protocol_sha256"] != protocol["protocol_sha256"]):
            raise ValueError(f"incomplete or inconsistent pretraining summary: h{width}")
        pretraining["by_width"][str(width)] = summary
    return {"trials": trials, "test_cases": len(test_ids),
            "primary_head": aggregate["primary_head"], "pretraining": pretraining}


def render(data: dict, output_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    trials, n = data["trials"], data["test_cases"]
    x = np.arange(len(trials))
    labels = []
    for row in trials:
        config = row["configuration"]
        label = f"pre{config['pretraining_steps']} / u{config['inner_steps']}"
        labels.append(label)
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [1, 1, 1.1]})
    fig.suptitle(f"Idea 01 scaling: exploratory reuse of {n} test cases\n"
                 f"One shared train111 readout across all 12 configurations; seed 0", fontsize=15)
    colors = {"residual": "#2b6cb0", "residual_no_update": "#c8d3df"}
    for ax_index, metric in enumerate(("correct", "mse")):
        ax = axes[ax_index]
        for shift, arm, label in ((-0.19, "residual", "Online residual TTT"),
                                  (0.19, "residual_no_update", "Same initialization, no online updates")):
            values = [row[arm][metric] for row in trials]
            bars = ax.bar(x + shift, values, width=0.36, label=label,
                          color=colors[arm], edgecolor="white", linewidth=0.5)
            ax.bar_label(bars, fmt="%d" if metric == "correct" else "%.3f", fontsize=8, padding=3)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        for boundary in (3.5, 7.5):
            ax.axvline(boundary, color="#a0aec0", linewidth=0.8, linestyle=":")
        for center, hidden, params in ((1.5, 128, "0.262M"), (5.5, 512, "1.049M"), (9.5, 2048, "4.194M")):
            ax.text(center, 1.02, f"hidden {hidden} / {params} fast parameters",
                    ha="center", va="bottom", transform=ax.get_xaxis_transform(), fontsize=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.margins(x=0.015)
        if metric == "correct":
            ax.set_title("A. Action forecasting: all configurations, same 57 questions", loc="left", pad=26)
            ax.set_ylabel(f"Correct / {n}")
            ax.set_ylim(0, n)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.legend(loc="upper right", frameon=False)
        else:
            ax.set_title("B. Past-target visual prediction at test time (lower is better)", loc="left", pad=26)
            ax.set_ylabel("Standardized MSE")
            ax.set_ylim(0, max(row[arm][metric] for row in trials for arm in colors) * 1.2)
    ax = axes[2]
    palette = {"128": "#2b6cb0", "512": "#d97706", "2048": "#15803d"}
    for width, row in data["pretraining"]["by_width"].items():
        steps = sorted(map(int, row["snapshots"]))
        for kind, style, marker in (("plain", "--", "s"), ("residual", "-", "o")):
            loss = [row["snapshots"][str(step)]["validation"][kind]["mse"] for step in steps]
            ax.plot(steps, loss, color=palette[width], linestyle=style, marker=marker,
                    label=f"hidden {width}, {kind}", linewidth=1.7, markersize=5)
    ax.set_title("C. Offline visual-prediction training on 111 train prefixes", loc="left")
    ax.set_xlabel("AdamW steps per target (128 origins / step)")
    ax.set_ylabel("Fixed validation-pair MSE")
    ax.set_xticks((0, 200, 800))
    ax.grid(alpha=0.2)
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.set_ylim(0.45, 1.09)
    fig.text(0.07, 0.015, "u = gradient updates per already-matured feedback pair; pre = offline optimizer steps. "
             "B scores forecasts issued in the final 30 s only after observed targets arrive.\n"
             "Test labels are silver; repeated views and horizon-token pairs do not add independent test cases. "
             "Offline training is ordinary prediction pretraining, not meta-training.",
             fontsize=9, color="#4a5568")
    fig.tight_layout(rect=(0.03, 0.065, 0.99, 0.94), h_pad=2.6)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"idea01_scaling_results.{suffix}" for suffix in ("png", "svg")]
    for path in paths:
        fig.savefig(path, dpi=180, facecolor="white")
        if path.suffix == ".svg":
            path.write_text("\n".join(line.rstrip() for line in
                                      path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/forecast_bench_v0/idea01_scaling_v1"))
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    args = parser.parse_args(argv)
    data = load_plot_data(args.run_dir)
    for path in render(data, args.output_dir):
        print(path)
    print(json.dumps({"readout": data["primary_head"], "test_cases": data["test_cases"],
                      "trials": data["trials"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
