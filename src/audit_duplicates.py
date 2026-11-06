"""Audit a train/test manifest pair for exact and near-duplicate images.

Near-duplicates are detected with a 64-bit perceptual hash (pHash). Two images
whose hashes differ by at most ``--max-distance`` bits are treated as the same
picture (re-encodes, resizes, small crops, watermarks).

Example:
    python src/audit_duplicates.py --root . --train data/train.jsonl --test data/test.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

from fight_data import image_fingerprints, read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--test", default="data/test.jsonl")
    parser.add_argument("--max-distance", type=int, default=6)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    train = [root / row["image"] for row in read_jsonl(root / args.train)]
    test = [root / row["image"] for row in read_jsonl(root / args.test)]
    md5 = {path: hashlib.md5(path.read_bytes()).hexdigest() for path in train + test}
    phash = image_fingerprints(train + test)

    train_md5 = {}
    for path in train:
        train_md5.setdefault(md5[path], path)
    exact = []
    near = []
    for path in test:
        if md5[path] in train_md5:
            exact.append((path, train_md5[md5[path]], 0))
            continue
        best = min(train, key=lambda other: phash[path] - phash[other])
        distance = phash[path] - phash[best]
        if distance <= args.max_distance:
            near.append((path, best, distance))

    def rel(path):
        return str(path.relative_to(root))

    report = {
        "train_images": len(train),
        "test_images": len(test),
        "max_distance": args.max_distance,
        "test_exact_duplicates_of_train": len(exact),
        "test_near_duplicates_of_train": len(near),
        "test_leaked_total": len(exact) + len(near),
        "test_leaked_fraction": (len(exact) + len(near)) / len(test) if test else 0.0,
        "pairs": [
            {"test": rel(a), "train": rel(b), "hamming": int(d)} for a, b, d in exact + near
        ],
    }
    summary = {key: value for key, value in report.items() if key != "pairs"}
    print(json.dumps(summary, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
