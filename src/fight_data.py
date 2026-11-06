import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROMPT = (
    "Does this image show people fighting, including a punch, kick, slap, or grapple? "
    "Reply with only one word: fight or none."
)

LABELS = ("fight", "none")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}

# source name -> (label, folders). Action-Net ships its own train/test folders, but
# they contain re-uploads of the same photo, so both folders are pooled and split
# again below with near-duplicate grouping.
SOURCES = {
    "actionnet_fighting": (
        "fight",
        (
            "datasets/action-net/action_net_v1/train/fighting",
            "datasets/action-net/action_net_v1/test/fighting",
        ),
    ),
    "hii_boxing_punching": ("fight", ("datasets/hii/facial_descriptors_dataset/Boxing-Punching",)),
    "hii_kicking": ("fight", ("datasets/hii/facial_descriptors_dataset/Kicking",)),
    "hii_hugging": ("none", ("datasets/negatives/Hugging",)),
    "hii_highfive": ("none", ("datasets/negatives/HighFive",)),
    "hii_handshaking": ("none", ("datasets/negatives/Handshaking",)),
}

SPLIT_RATIOS = {"test": 0.2, "val": 0.1}
NEAR_DUPLICATE_BITS = 6
SPLIT_SEED = 42


def list_images(folder):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def image_fingerprints(paths):
    """64-bit perceptual hash per image (imagehash.ImageHash, subtract = Hamming)."""
    import imagehash
    from PIL import Image

    hashes = {}
    for path in paths:
        with Image.open(path) as image:
            hashes[path] = imagehash.phash(image.convert("RGB"))
    return hashes


def duplicate_groups(paths, max_bits=NEAR_DUPLICATE_BITS):
    """Union-find over exact (md5) and near (pHash) duplicates. Returns list of groups."""
    import numpy as np

    paths = list(paths)
    parent = list(range(len(paths)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_md5 = {}
    for index, path in enumerate(paths):
        digest = hashlib.md5(path.read_bytes()).hexdigest()
        if digest in by_md5:
            union(index, by_md5[digest])
        else:
            by_md5[digest] = index

    hashes = image_fingerprints(paths)
    bits = np.stack([hashes[path].hash.reshape(-1) for path in paths]).astype(np.uint8)
    for start in range(0, len(paths), 256):
        block = bits[start : start + 256]
        distance = (block[:, None, :] != bits[None, :, :]).sum(-1)
        for i, j in zip(*np.nonzero(distance <= max_bits)):
            a, b = start + int(i), int(j)
            if a < b:
                union(a, b)

    groups = defaultdict(list)
    for index, path in enumerate(paths):
        groups[find(index)].append(path)
    return list(groups.values())


def build_records(root):
    """Leakage-free split. Returns (splits, audit) with splits[name] = [(path, label, source)]."""
    root = Path(root)
    owner = {}
    for source, (label, folders) in SOURCES.items():
        for folder in folders:
            for path in list_images(root / folder):
                owner[path] = (source, label)
    paths = sorted(owner)
    groups = duplicate_groups(paths)

    conflicting = [group for group in groups if len({owner[p][1] for p in group}) > 1]
    clean_groups = [group for group in groups if len({owner[p][1] for p in group}) == 1]
    duplicates_removed = sum(len(group) - 1 for group in clean_groups)

    rng = random.Random(SPLIT_SEED)
    by_source = defaultdict(list)
    for group in clean_groups:
        group = sorted(group)
        source = Counter(owner[p][0] for p in group).most_common(1)[0][0]
        by_source[source].append(group)

    splits = {"train": [], "val": [], "test": []}
    for source in SOURCES:
        source_groups = sorted(by_source[source])
        rng.shuffle(source_groups)
        count = len(source_groups)
        n_test = int(round(count * SPLIT_RATIOS["test"]))
        n_val = int(round(count * SPLIT_RATIOS["val"]))
        assignment = {
            "test": source_groups[:n_test],
            "val": source_groups[n_test : n_test + n_val],
            "train": source_groups[n_test + n_val :],
        }
        for split, chosen in assignment.items():
            for group in chosen:
                # one representative per duplicate group: duplicates add no information
                representative = group[0]
                splits[split].append((representative, owner[representative][1], source))
    for rows in splits.values():
        rng.shuffle(rows)

    audit = {
        "raw_images": len(paths),
        "duplicate_groups": sum(1 for group in groups if len(group) > 1),
        "duplicates_removed": duplicates_removed,
        "label_conflict_groups_dropped": len(conflicting),
        "label_conflict_images_dropped": sum(len(group) for group in conflicting),
        "near_duplicate_hamming_bits": NEAR_DUPLICATE_BITS,
    }
    return splits, audit


def write_split(root):
    root = Path(root).resolve()
    splits, audit = build_records(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        with (data_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for path, label, source in rows:
                record = {
                    "image": path.relative_to(root).as_posix(),
                    "label": label,
                    "source": source,
                }
                handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    summary = {name: _count(rows) for name, rows in splits.items()}
    summary["audit"] = audit
    (data_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _count(rows):
    counts = {label: 0 for label in LABELS}
    sources = Counter()
    for _, label, source in rows:
        counts[label] += 1
        sources[source] += 1
    counts["total"] = len(rows)
    counts["by_source"] = dict(sorted(sources.items()))
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
    if word in LABELS:
        return word
    return "other"


def score(rows):
    """Confusion matrix + per-class precision/recall/F1. rows need 'label' and 'prediction'."""
    matrix = {label: {pred: 0 for pred in ("fight", "none", "other")} for label in LABELS}
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
    f1s = []
    for label in LABELS:
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        true_positive = matrix[label][label]
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
        metrics[f"{label}_recall"] = recall
        metrics[f"{label}_precision"] = precision
        metrics[f"{label}_f1"] = f1
        metrics[f"{label}_support"] = support
    metrics["macro_f1"] = sum(f1s) / len(f1s)
    return metrics
