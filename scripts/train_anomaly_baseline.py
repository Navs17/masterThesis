"""Train a feature-based one-class anomaly detector per domain that follows
the MVTec convention (pill, capsule): fit a Gaussian over frozen backbone
features of non-defective ("good") training images only, then score test
images by Mahalanobis distance from that Gaussian.

This is the anomaly-detection comparison point against
scripts/train_baseline.py's supervised classifier. Since pill and capsule
never have defective labels in train, this is the dataset's intended usage
pattern, and may generalize better across pill domains than a defect-
specific supervised classifier trained on a different domain (tablet).
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import build_domain_datasets
from src.eval import compute_classification_metrics, compute_roc_auc
from src.models import FeatureGaussianAnomalyDetector
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"


def run_domain(domain, backbone_name, image_size, batch_size, threshold_percentile, device):
    train_ds, val_ds, test_ds = build_domain_datasets(MANIFEST_PATH, PROCESSED_DIR, domain, image_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    detector = FeatureGaussianAnomalyDetector(backbone_name=backbone_name, device=device)
    detector.fit(train_loader)

    val_scores, _ = detector.score(val_loader)
    threshold = float(np.percentile(val_scores, threshold_percentile))

    test_scores, test_labels = detector.score(test_loader)
    test_preds = (test_scores > threshold).astype(int)

    metrics = compute_classification_metrics(test_labels, test_preds.tolist())
    metrics["roc_auc"] = compute_roc_auc(test_labels, test_scores.tolist())
    metrics["threshold"] = threshold
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domains", nargs="*", default=["pill", "capsule"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    for domain in args.domains:
        metrics = run_domain(
            domain, args.backbone, args.image_size, args.batch_size, args.threshold_percentile, device
        )
        results[domain] = metrics
        print(
            f"[{domain}] roc_auc={metrics['roc_auc']:.4f} acc={metrics['accuracy']:.4f} "
            f"precision={metrics['precision']:.4f} recall={metrics['recall']:.4f} f1={metrics['f1']:.4f}"
        )

    run_dir = RUNS_DIR / f"anomaly_baseline_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"\nSaved metrics to {run_dir}")


if __name__ == "__main__":
    main()
