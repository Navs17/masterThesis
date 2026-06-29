"""Train a baseline supervised binary classifier on one domain, then zero-shot
evaluate it on the others.

Only the tablet domain has genuine defective/non_defective labels in its
train split -- pill and capsule follow the MVTec AD convention (train/ has
only non-defective images). So the baseline trains on tablet and zero-shot
evaluates on pill and capsule test sets: this is the motivating result for
why naive supervised transfer fails to generalize across pill domains,
which continual learning / anomaly detection approaches aim to address.
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from src.data import PillDefectDataset, build_domain_datasets, get_transforms
from src.eval import compute_classification_metrics
from src.models import PillDefectClassifier
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"


def run_epoch(model, loader, device, optimizer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.CrossEntropyLoss()

    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            loss = criterion(logits, labels)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * images.size(0)
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(labels.tolist())

    metrics = compute_classification_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def evaluate_test_split(model, domain, device, image_size, batch_size):
    test_ds = PillDefectDataset(MANIFEST_PATH, PROCESSED_DIR, domain, "test", get_transforms(False, image_size))
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return run_epoch(model, loader, device, optimizer=None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="tablet", help="Domain to train on (must have train labels)")
    parser.add_argument("--eval-domains", nargs="*", default=["pill", "capsule"])
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds, val_ds, _ = build_domain_datasets(MANIFEST_PATH, PROCESSED_DIR, args.domain, args.image_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = PillDefectClassifier(backbone_name=args.backbone, pretrained=True, freeze_backbone=args.freeze_backbone)
    model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, device, optimizer)
        val_metrics = run_epoch(model, val_loader, device, optimizer=None)
        print(
            f"epoch {epoch}/{args.epochs} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f}"
        )
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})

    test_metrics = evaluate_test_split(model, args.domain, device, args.image_size, args.batch_size)
    print(f"\n[{args.domain}] held-out test: acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f}")

    zero_shot_results = {}
    for eval_domain in args.eval_domains:
        zs_metrics = evaluate_test_split(model, eval_domain, device, args.image_size, args.batch_size)
        zero_shot_results[eval_domain] = zs_metrics
        print(f"[{eval_domain}] zero-shot test: acc={zs_metrics['accuracy']:.4f} f1={zs_metrics['f1']:.4f}")

    run_name = f"baseline_{args.domain}_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(
            {"args": vars(args), "history": history, "test": test_metrics, "zero_shot": zero_shot_results},
            f,
            indent=2,
        )
    print(f"\nSaved checkpoint and metrics to {run_dir}")


if __name__ == "__main__":
    main()
