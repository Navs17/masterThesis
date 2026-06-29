"""Dataset class that reads data/processed/manifest.csv."""

import csv
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

LABEL_TO_IDX = {"non_defective": 0, "defective": 1}
IDX_TO_LABEL = {v: k for k, v in LABEL_TO_IDX.items()}


class PillDefectDataset(Dataset):
    """Binary defective/non_defective dataset for one or more domains and a single split."""

    def __init__(self, manifest_path, processed_dir, domains, split, transform=None):
        self.processed_dir = Path(processed_dir)
        self.transform = transform

        domains = {domains} if isinstance(domains, str) else set(domains)
        with open(manifest_path, newline="") as f:
            rows = list(csv.DictReader(f))

        self.rows = [r for r in rows if r["domain"] in domains and r["split"] == split]
        if not self.rows:
            raise ValueError(f"No rows found for domains={domains!r} split={split!r} in {manifest_path}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        image = Image.open(self.processed_dir / row["image_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, LABEL_TO_IDX[row["label"]]
