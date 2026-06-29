"""Build binary-split ImageFolders + manifest.csv from the raw zip archives in data/raw/.

Domains:
  pill, capsule  -- MVTec AD convention: train/ has only non-defective images,
                    defects only appear in test/<defect_type>/.
  tablet         -- Roboflow COCO export with bounding-box labels
                    (defected / no-defect). Only annotated images are used;
                    each image is cropped to its single annotated box.
"""

import argparse
import csv
import io
import json
import random
import zipfile
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

NON_DEFECTIVE = "non_defective"
DEFECTIVE = "defective"


def write_bytes(dst: Path, data: bytes) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(data)


def process_mvtec(zip_path: Path, domain: str, val_fraction: float, seed: int, manifest: list) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()

        train_good = sorted(n for n in names if f"{domain}/train/good/" in n and n.endswith(".png"))
        rng = random.Random(seed)
        rng.shuffle(train_good)
        n_val = int(len(train_good) * val_fraction)
        val_files, train_files = train_good[:n_val], train_good[n_val:]

        for split, files in [("train", train_files), ("val", val_files)]:
            for n in files:
                fname = Path(n).name
                dst = PROCESSED_DIR / domain / split / NON_DEFECTIVE / fname
                write_bytes(dst, z.read(n))
                manifest.append(
                    {
                        "domain": domain,
                        "split": split,
                        "label": NON_DEFECTIVE,
                        "defect_type": "good",
                        "image_path": str(dst.relative_to(PROCESSED_DIR)),
                    }
                )

        test_files = sorted(n for n in names if f"{domain}/test/" in n and n.endswith(".png"))
        for n in test_files:
            defect_type = Path(n).parent.name
            label = NON_DEFECTIVE if defect_type == "good" else DEFECTIVE
            fname = f"{defect_type}_{Path(n).name}"
            dst = PROCESSED_DIR / domain / "test" / label / fname
            write_bytes(dst, z.read(n))
            manifest.append(
                {
                    "domain": domain,
                    "split": "test",
                    "label": label,
                    "defect_type": defect_type,
                    "image_path": str(dst.relative_to(PROCESSED_DIR)),
                }
            )

    print(f"[{domain}] train={len(train_files)} val={len(val_files)} test={len(test_files)}")


def process_tablet_coco(zip_path: Path, domain: str, manifest: list) -> None:
    category_to_label = {"defected": DEFECTIVE, "no-defect": NON_DEFECTIVE}

    with zipfile.ZipFile(zip_path) as z:
        for raw_split in ["train", "valid", "test"]:
            split = "val" if raw_split == "valid" else raw_split
            ann_path = f"{raw_split}/_annotations.coco.json"
            if ann_path not in z.namelist():
                continue
            ann = json.loads(z.read(ann_path))
            cats = {c["id"]: c["name"] for c in ann["categories"]}
            images_by_id = {im["id"]: im for im in ann["images"]}

            count = 0
            for a in ann["annotations"]:
                category_name = cats[a["category_id"]]
                if category_name not in category_to_label:
                    continue
                label = category_to_label[category_name]

                image_info = images_by_id[a["image_id"]]
                img_bytes = z.read(f"{raw_split}/{image_info['file_name']}")
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

                x, y, w, h = a["bbox"]
                left, top = max(0, int(x)), max(0, int(y))
                right, bottom = min(img.width, int(x + w)), min(img.height, int(y + h))
                crop = img.crop((left, top, right, bottom))

                fname = f"{Path(image_info['file_name']).stem}.png"
                dst = PROCESSED_DIR / domain / split / label / fname
                dst.parent.mkdir(parents=True, exist_ok=True)
                crop.save(dst)
                manifest.append(
                    {
                        "domain": domain,
                        "split": split,
                        "label": label,
                        "defect_type": category_name,
                        "image_path": str(dst.relative_to(PROCESSED_DIR)),
                    }
                )
                count += 1

            print(f"[{domain}] {split}: {count} annotated images cropped and saved")


def write_manifest(manifest: list) -> None:
    manifest_path = PROCESSED_DIR / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "split", "label", "defect_type", "image_path"])
        writer.writeheader()
        writer.writerows(manifest)
    print(f"\nWrote manifest with {len(manifest)} rows to {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest: list = []

    process_mvtec(RAW_DIR / "pill.zip", "pill", args.val_fraction, args.seed, manifest)
    process_mvtec(RAW_DIR / "capsule.zip", "capsule", args.val_fraction, args.seed, manifest)
    process_tablet_coco(RAW_DIR / "tablet defect detection_annotated.coco.zip", "tablet", manifest)

    write_manifest(manifest)


if __name__ == "__main__":
    main()
