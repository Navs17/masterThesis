import csv

import torch
from PIL import Image

from src.data import PillDefectDataset, build_task_sequence, get_transforms

MANIFEST_FIELDS = ["domain", "split", "label", "defect_type", "image_path"]


def _make_fake_processed_dataset(root):
    """Build a tiny synthetic data/processed/ tree + manifest.csv across two domains."""
    rows = []
    for domain in ["fake_pill", "fake_capsule"]:
        for split, label in [("train", "non_defective"), ("val", "non_defective"), ("test", "defective")]:
            rel_path = f"{domain}/{split}/{label}/img.png"
            abs_path = root / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 32), color=(255, 0, 0)).save(abs_path)
            rows.append(
                {"domain": domain, "split": split, "label": label, "defect_type": label, "image_path": rel_path}
            )

    manifest_path = root / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return manifest_path


def test_dataset_loads_correct_label_and_shape(tmp_path):
    manifest_path = _make_fake_processed_dataset(tmp_path)
    transform = get_transforms(train=False, image_size=64)

    ds = PillDefectDataset(manifest_path, tmp_path, "fake_pill", "test", transform=transform)
    assert len(ds) == 1

    image, label = ds[0]
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, 64, 64)
    assert label == 1  # defective


def test_dataset_filters_by_domain_and_split(tmp_path):
    manifest_path = _make_fake_processed_dataset(tmp_path)

    pill_train = PillDefectDataset(manifest_path, tmp_path, "fake_pill", "train")
    both_train = PillDefectDataset(manifest_path, tmp_path, ["fake_pill", "fake_capsule"], "train")

    assert len(pill_train) == 1
    assert len(both_train) == 2


def test_build_task_sequence_returns_one_entry_per_domain(tmp_path):
    manifest_path = _make_fake_processed_dataset(tmp_path)

    tasks = build_task_sequence(manifest_path, tmp_path, domain_order=["fake_pill", "fake_capsule"], image_size=64)

    assert [t["domain"] for t in tasks] == ["fake_pill", "fake_capsule"]
    assert len(tasks[0]["train"]) == 1
    assert len(tasks[0]["test"]) == 1
