"""Aggregate evaluation JSONs into results/summary.json and a Markdown table.

Expected layout (written by scripts/run_experiments.sh):
    results/baseline_{val,test}.json
    results/finetuned_seed{N}_{val,test}.json

For every run the decision threshold is chosen on *val* (max macro-F1) and then
applied unchanged to *test*; metrics at the default 0.5 threshold are kept too.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from metrics import best_threshold, full_report

KEYS = ("accuracy", "macro_f1", "fight_recall", "fight_precision", "none_recall", "roc_auc", "pr_auc_fight")


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_summary(val_path, test_path):
    val = load(val_path)["predictions"]
    test_file = load(test_path)
    test = test_file["predictions"]
    threshold = best_threshold(val)
    tuned = full_report(test, threshold)
    default = test_file["metrics"]
    return {
        "test_at_0.5": {key: default[key] for key in KEYS} | {"confusion": default["confusion"]},
        "val_threshold": threshold,
        "test_at_val_threshold": {key: tuned[key] for key in KEYS} | {"confusion": tuned["confusion"]},
        "ci95_at_0.5": default["ci95"],
        "format_compliance": default.get("free_generation", {}).get("format_compliance"),
        "free_generation_accuracy": default.get("free_generation", {}).get("accuracy"),
        "by_source_at_0.5": default["by_source"],
        "throughput": default.get("throughput"),
        "latency": default.get("latency"),
    }


def mean_std(values):
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="results")
    parser.add_argument("--prefix", default="finetuned")
    args = parser.parse_args()
    results = Path(args.results)

    summary = {"baseline": run_summary(results / "baseline_val.json", results / "baseline_test.json")}
    seeds = sorted(
        int(match.group(1))
        for path in results.glob(f"{args.prefix}_seed*_test.json")
        if (match := re.search(r"seed(\d+)_test", path.name))
    )
    runs = {
        seed: run_summary(results / f"{args.prefix}_seed{seed}_val.json", results / f"{args.prefix}_seed{seed}_test.json")
        for seed in seeds
    }
    summary["finetuned_runs"] = {str(seed): run for seed, run in runs.items()}
    if runs:
        summary["finetuned_aggregate"] = {
            view: {key: mean_std([run[view][key] for run in runs.values()]) for key in KEYS}
            for view in ("test_at_0.5", "test_at_val_threshold")
        }
    (results / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def pct(value):
        return f"{100 * value:.2f}"

    lines = ["| Model | Accuracy | Macro-F1 | Fight recall | Fight precision | None recall | ROC-AUC |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    base = summary["baseline"]["test_at_0.5"]
    lines.append(
        "| Base Qwen3-VL-4B (zero-shot) | "
        + " | ".join(pct(base[key]) for key in KEYS[:-2])
        + f" | {base['roc_auc']:.4f} |"
    )
    if runs:
        agg = summary["finetuned_aggregate"]["test_at_0.5"]
        lines.append(
            f"| QLoRA fine-tuned (mean ± std, {len(runs)} seeds) | "
            + " | ".join(f"{pct(agg[key]['mean'])} ± {pct(agg[key]['std'])}" for key in KEYS[:-2])
            + f" | {agg['roc_auc']['mean']:.4f} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
