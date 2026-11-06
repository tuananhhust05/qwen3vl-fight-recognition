import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import fight_data  # noqa: E402
from fight_data import duplicate_groups, parse_label, score  # noqa: E402
from metrics import best_threshold, full_report, with_threshold  # noqa: E402


def _noise_image(path, seed, size=96):
    from PIL import Image

    rng = random.Random(seed)
    image = Image.new("RGB", (size, size))
    image.putdata([(rng.randrange(256), rng.randrange(256), rng.randrange(256)) for _ in range(size * size)])
    image.save(path)
    return image


def test_parse_label():
    assert parse_label(" Fight.") == "fight"
    assert parse_label("none") == "none"
    assert parse_label("I think so") == "other"
    assert parse_label("") == "other"


def test_score_confusion_and_f1():
    rows = [
        {"label": "fight", "prediction": "fight"},
        {"label": "fight", "prediction": "none"},
        {"label": "none", "prediction": "none"},
        {"label": "none", "prediction": "other"},
    ]
    metrics = score(rows)
    assert metrics["accuracy"] == 0.5
    assert metrics["fight_recall"] == 0.5
    assert metrics["fight_precision"] == 1.0
    assert metrics["none_precision"] == 0.5
    assert metrics["confusion"]["none"]["other"] == 1


def test_duplicate_groups_catch_resized_copy(tmp_path):
    original = _noise_image(tmp_path / "a.jpg", seed=1)
    original.resize((80, 80)).save(tmp_path / "a_resized.jpg")
    (tmp_path / "a_copy.jpg").write_bytes((tmp_path / "a.jpg").read_bytes())
    _noise_image(tmp_path / "b.jpg", seed=2)
    groups = duplicate_groups(sorted(tmp_path.glob("*.jpg")))
    names = sorted(sorted(path.name for path in group) for group in groups)
    assert names == [["a.jpg", "a_copy.jpg", "a_resized.jpg"], ["b.jpg"]]


def test_build_records_has_no_cross_split_duplicates(tmp_path, monkeypatch):
    sources = {}
    seed = 0
    for name, label in (("fa", "fight"), ("nb", "none")):
        folder = tmp_path / name
        folder.mkdir()
        for index in range(30):
            seed += 1
            image = _noise_image(folder / f"{index}.jpg", seed)
            if index % 3 == 0:  # every third image gets a re-encoded duplicate
                image.save(folder / f"{index} (2).jpg", quality=70)
        sources[name] = (label, (name,))
    monkeypatch.setattr(fight_data, "SOURCES", sources)
    splits, audit = fight_data.build_records(tmp_path)
    assert audit["duplicates_removed"] == 20
    assert audit["label_conflict_images_dropped"] == 0
    seen = {}
    for split, rows in splits.items():
        for path, _, _ in rows:
            stem = path.name.split(" ")[0].split(".")[0] + path.parent.name
            assert seen.setdefault(stem, split) == split
    assert sum(len(rows) for rows in splits.values()) == 60
    assert {source for rows in splits.values() for _, _, source in rows} == {"fa", "nb"}


def test_threshold_selection_and_report():
    predictions = [
        {"label": "fight", "p_fight": 0.9, "source": "s"},
        {"label": "fight", "p_fight": 0.4, "source": "s"},
        {"label": "none", "p_fight": 0.2, "source": "s"},
        {"label": "none", "p_fight": 0.1, "source": "s"},
    ]
    threshold = best_threshold(predictions)
    assert 0.2 < threshold <= 0.4
    assert all(row["prediction"] == row["label"] for row in with_threshold(predictions, threshold))
    report = full_report(predictions, 0.5)
    assert report["roc_auc"] == pytest.approx(1.0)
    assert report["accuracy"] == pytest.approx(0.75)
