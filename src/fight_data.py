import json
import random
from pathlib import Path

PROMPT = (
    "Does this image show people fighting, including a punch, kick, slap, or grapple? "
    "Reply with only one word: fight or none."
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_holdout(paths, test_ratio, rng):
    paths = list(paths)
    rng.shuffle(paths)
    test_count = max(1, int(round(len(paths) * test_ratio))) if len(paths) > 1 else 0
    return paths[test_count:], paths[:test_count]


def build_records(root):
    root = Path(root)
    rng = random.Random(42)
    fight_train = list_images(root / "datasets/action-net/action_net_v1/train/fighting")
    fight_test = list_images(root / "datasets/action-net/action_net_v1/test/fighting")
    punch_train, punch_test = split_holdout(
        list_images(root / "datasets/hii/facial_descriptors_dataset/Boxing-Punching"),
        0.2,
        rng,
    )
    kick_train, kick_test = split_holdout(
        list_images(root / "datasets/hii/facial_descriptors_dataset/Kicking"),
        0.2,
        rng,
    )
    negative_folders = [
        root / "datasets/negatives/Hugging",
        root / "datasets/negatives/HighFive",
        root / "datasets/negatives/Handshaking",
    ]
    negatives = []
    for folder in negative_folders:
        negatives.extend(list_images(folder))
    negative_train, negative_test = split_holdout(negatives, 0.2, rng)

    train = [(path, "fight") for path in fight_train + punch_train + kick_train]
    train += [(path, "none") for path in negative_train]
    test = [(path, "fight") for path in fight_test + punch_test + kick_test]
    test += [(path, "none") for path in negative_test]
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_split(root):
    root = Path(root)
    train, test = build_records(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("test", test)):
        output = data_dir / f"{name}.jsonl"
        with output.open("w", encoding="utf-8") as handle:
            for path, label in rows:
                handle.write(
                    json.dumps(
                        {
                            "image": str(path.resolve().relative_to(root.resolve())),
                            "label": label,
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
    summary = {
        "train": _count(train),
        "test": _count(test),
    }
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _count(rows):
    counts = {"fight": 0, "none": 0}
    for _, label in rows:
        counts[label] += 1
    counts["total"] = len(rows)
    return counts


def read_jsonl(path):
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def messages_for(image, label=None):
    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }
    if label is None:
        return [user]
    return [
        user,
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]


def parse_label(text):
    if not text or not text.strip():
        return "other"
    word = text.strip().lower().split()[0].strip(".,!?:;\"'")
    if word in {"fight", "none"}:
        return word
    return "other"


def score(rows):
    labels = ("fight", "none")
    matrix = {label: {pred: 0 for pred in ("fight", "none", "other")} for label in labels}
    correct = 0
    for row in rows:
        matrix[row["label"]][row["prediction"]] += 1
        if row["prediction"] == row["label"]:
            correct += 1
    total = len(rows)
    metrics = {
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "confusion": matrix,
    }
    for label in labels:
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in labels)
        true_positive = matrix[label][label]
        metrics[f"{label}_recall"] = true_positive / support if support else 0.0
        metrics[f"{label}_precision"] = true_positive / predicted if predicted else 0.0
        metrics[f"{label}_support"] = support
    return metrics
