"""Assemble the per-domain train/val/test datasets into a continual learning task sequence."""

from .dataset import PillDefectDataset
from .transforms import get_transforms

DEFAULT_DOMAIN_ORDER = ["pill", "capsule", "tablet"]


def build_domain_datasets(manifest_path, processed_dir, domain, image_size=224):
    train = PillDefectDataset(manifest_path, processed_dir, domain, "train", get_transforms(True, image_size))
    val = PillDefectDataset(manifest_path, processed_dir, domain, "val", get_transforms(False, image_size))
    test = PillDefectDataset(manifest_path, processed_dir, domain, "test", get_transforms(False, image_size))
    return train, val, test


def build_task_sequence(manifest_path, processed_dir, domain_order=DEFAULT_DOMAIN_ORDER, image_size=224):
    """Returns a list of {"domain", "train", "val", "test"} dicts, one per domain in order."""
    tasks = []
    for domain in domain_order:
        train, val, test = build_domain_datasets(manifest_path, processed_dir, domain, image_size)
        tasks.append({"domain": domain, "train": train, "val": val, "test": test})
    return tasks
