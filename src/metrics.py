"""Metrics shared by evaluation and reporting (no GPU dependencies)."""

import random
from collections import defaultdict

from sklearn.metrics import average_precision_score, roc_auc_score

from fight_data import score


def with_threshold(predictions, threshold):
    return [
        {**row, "prediction": "fight" if row["p_fight"] >= threshold else "none"}
        for row in predictions
    ]


def bootstrap_ci(predictions, threshold, resamples=2000, seed=0):
    """Percentile 95% CI of accuracy / macro-F1 / fight recall over image resamples."""
    rng = random.Random(seed)
    rows = with_threshold(predictions, threshold)
    samples = defaultdict(list)
    for _ in range(resamples):
        draw = [rows[rng.randrange(len(rows))] for _ in rows]
        metrics = score(draw)
        for key in ("accuracy", "macro_f1", "fight_recall", "none_recall"):
            samples[key].append(metrics[key])
    interval = {}
    for key, values in samples.items():
        values.sort()
        interval[key] = [values[int(0.025 * resamples)], values[int(0.975 * resamples) - 1]]
    return interval


def best_threshold(predictions):
    """Threshold maximising macro-F1 (ties -> closest to 0.5). Use on validation only."""
    candidates = sorted({row["p_fight"] for row in predictions} | {0.5})
    best = None
    for threshold in candidates:
        value = score(with_threshold(predictions, threshold))["macro_f1"]
        key = (value, -abs(threshold - 0.5))
        if best is None or key > best[0]:
            best = (key, threshold)
    return best[1]


def full_report(predictions, threshold=0.5):
    report = score(with_threshold(predictions, threshold))
    report["threshold"] = threshold
    gold = [1 if row["label"] == "fight" else 0 for row in predictions]
    probability = [row["p_fight"] for row in predictions]
    report["roc_auc"] = roc_auc_score(gold, probability)
    report["pr_auc_fight"] = average_precision_score(gold, probability)
    report["ci95"] = bootstrap_ci(predictions, threshold)
    if all("free_prediction" in row for row in predictions):
        free = [{**row, "prediction": row["free_prediction"]} for row in predictions]
        free_metrics = score(free)
        report["free_generation"] = {
            "accuracy": free_metrics["accuracy"],
            "format_compliance": sum(row["free_prediction"] != "other" for row in predictions)
            / len(predictions),
        }
    by_source = defaultdict(list)
    for row in with_threshold(predictions, threshold):
        by_source[row["source"]].append(row)
    report["by_source"] = {
        source: {
            "images": len(rows),
            "accuracy": sum(row["prediction"] == row["label"] for row in rows) / len(rows),
        }
        for source, rows in sorted(by_source.items())
    }
    return report
