"""Continual learning training with Learning without Forgetting (LwF).

Domain sequence: tablet -> pill -> capsule.

After finishing each task the model is snapshotted as a frozen teacher.
On the next task every training step adds a knowledge-distillation loss
that penalises the student for drifting away from the teacher's output
distribution, without storing any past-task images.

Tablet is first because it provides genuine defective/non_defective labels
in its train split. Pill and capsule follow the MVTec AD convention (train/
contains only non-defective images), so we use 50% of their labeled test
split for CL fine-tuning -- the same methodological choice as the EWC and
Replay scripts.
"""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Subset

from src.data import PillDefectDataset, build_domain_datasets, get_transforms
from src.eval import compute_bwt, compute_classification_metrics, compute_forgetting
from src.models import PillDefectClassifier
from src.strategies import LwF
from src.utils import set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RUNS_DIR = PROJECT_ROOT / "runs"


def stratified_split(dataset, fraction, seed):
    by_label = {}
    for idx in range(len(dataset)):
        _, label = dataset[idx]
        by_label.setdefault(label, []).append(idx)

    rng = random.Random(seed)
    a_indices, b_indices = [], []
    for indices in by_label.values():
        shuffled = indices[:]
        rng.shuffle(shuffled)
        n_a = int(len(shuffled) * fraction)
        a_indices.extend(shuffled[:n_a])
        b_indices.extend(shuffled[n_a:])

    return Subset(dataset, a_indices), Subset(dataset, b_indices)


def run_epoch(model, loader, device, optimizer=None, lwf=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.CrossEntropyLoss()

    total_loss, total_ce, total_kd = 0.0, 0.0, 0.0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.set_grad_enabled(train_mode):
            logits = model(images)
            ce_loss = criterion(logits, labels)
            kd_loss = lwf.distillation_loss(logits, images) if lwf is not None else torch.tensor(0.0)
            loss = ce_loss + kd_loss

            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        n = images.size(0)
        total_loss += loss.item() * n
        total_ce += ce_loss.item() * n
        total_kd += kd_loss.item() * n
        all_preds.extend(logits.argmax(dim=1).tolist())
        all_labels.extend(labels.tolist())

    metrics = compute_classification_metrics(all_labels, all_preds)
    n_total = len(all_labels)
    metrics["loss"] = total_loss / n_total
    metrics["ce_loss"] = total_ce / n_total
    metrics["kd_loss"] = total_kd / n_total
    return metrics


def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return run_epoch(model, loader, device)


def train_task(model, train_loader, val_loader, device, epochs, lr, lwf):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, lwf)
        val_m = run_epoch(model, val_loader, device)
        print(
            f"  epoch {epoch}/{epochs} "
            f"ce={train_m['ce_loss']:.4f} kd={train_m['kd_loss']:.4f} "
            f"train_acc={train_m['accuracy']:.4f} "
            f"val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Distillation temperature T (higher = softer teacher).")
    parser.add_argument("--lambda-lwf", type=float, default=1.0,
                        help="Weight for the distillation loss. 0.0 = naive sequential.")
    parser.add_argument("--cl-split-fraction", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Mode: LwF | T={args.temperature} lambda={args.lambda_lwf} | device={device}\n"
    )

    eval_transform = get_transforms(train=False, image_size=args.image_size)

    tablet_train, tablet_val, tablet_test = build_domain_datasets(
        MANIFEST_PATH, PROCESSED_DIR, "tablet", args.image_size
    )

    task_specs = {}
    for domain in ["pill", "capsule"]:
        test_ds = PillDefectDataset(MANIFEST_PATH, PROCESSED_DIR, domain, "test", eval_transform)
        cl_train_ds, cl_eval_ds = stratified_split(test_ds, args.cl_split_fraction, args.seed)
        val_ds = PillDefectDataset(MANIFEST_PATH, PROCESSED_DIR, domain, "val", eval_transform)
        task_specs[domain] = {"cl_train": cl_train_ds, "cl_eval": cl_eval_ds, "val": val_ds}

    model = PillDefectClassifier(backbone_name=args.backbone, pretrained=True)
    model.to(device)
    lwf = LwF(temperature=args.temperature, lambda_lwf=args.lambda_lwf)

    task_names = ["tablet", "pill", "capsule"]
    T = len(task_names)
    result_matrix = [[None] * T for _ in range(T)]
    history = []

    # ── Task 0: tablet ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Task 0: tablet")
    train_loader = DataLoader(tablet_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tablet_val, batch_size=args.batch_size, shuffle=False)
    train_task(model, train_loader, val_loader, device, args.epochs_per_task, args.lr, lwf)

    m = evaluate(model, tablet_test, device, args.batch_size)
    result_matrix[0][0] = m["f1"]
    print(f"  [After Task 0] tablet test: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    history.append({"after_task": "tablet", "tablet": m})

    lwf.snapshot(model, device)
    print("  Teacher snapshot saved.")

    # ── Task 1: pill ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 1: pill (CL fine-tune on 50% of labeled test split)")
    cl_train_loader = DataLoader(task_specs["pill"]["cl_train"], batch_size=args.batch_size, shuffle=True)
    cl_val_loader = DataLoader(task_specs["pill"]["val"], batch_size=args.batch_size, shuffle=False)
    train_task(model, cl_train_loader, cl_val_loader, device, args.epochs_per_task, args.lr, lwf)

    m_tablet = evaluate(model, tablet_test, device, args.batch_size)
    m_pill = evaluate(model, task_specs["pill"]["cl_eval"], device, args.batch_size)
    result_matrix[1][0] = m_tablet["f1"]
    result_matrix[1][1] = m_pill["f1"]
    print(f"  [After Task 1] tablet test: acc={m_tablet['accuracy']:.4f} f1={m_tablet['f1']:.4f}")
    print(f"  [After Task 1] pill eval:   acc={m_pill['accuracy']:.4f} f1={m_pill['f1']:.4f}")
    history.append({"after_task": "pill", "tablet": m_tablet, "pill": m_pill})

    lwf.snapshot(model, device)
    print("  Teacher snapshot updated.")

    # ── Task 2: capsule ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 2: capsule (CL fine-tune on 50% of labeled test split)")
    cl_train_loader = DataLoader(task_specs["capsule"]["cl_train"], batch_size=args.batch_size, shuffle=True)
    cl_val_loader = DataLoader(task_specs["capsule"]["val"], batch_size=args.batch_size, shuffle=False)
    train_task(model, cl_train_loader, cl_val_loader, device, args.epochs_per_task, args.lr, lwf)

    m_tablet = evaluate(model, tablet_test, device, args.batch_size)
    m_pill = evaluate(model, task_specs["pill"]["cl_eval"], device, args.batch_size)
    m_capsule = evaluate(model, task_specs["capsule"]["cl_eval"], device, args.batch_size)
    result_matrix[2][0] = m_tablet["f1"]
    result_matrix[2][1] = m_pill["f1"]
    result_matrix[2][2] = m_capsule["f1"]
    print(f"  [After Task 2] tablet test:  acc={m_tablet['accuracy']:.4f} f1={m_tablet['f1']:.4f}")
    print(f"  [After Task 2] pill eval:    acc={m_pill['accuracy']:.4f} f1={m_pill['f1']:.4f}")
    print(f"  [After Task 2] capsule eval: acc={m_capsule['accuracy']:.4f} f1={m_capsule['f1']:.4f}")
    history.append({"after_task": "capsule", "tablet": m_tablet, "pill": m_pill, "capsule": m_capsule})

    # ── CL summary ────────────────────────────────────────────────────────────
    flat = [[result_matrix[t][i] if result_matrix[t][i] is not None else 0.0 for i in range(T)] for t in range(T)]
    bwt = compute_bwt(flat)
    forgetting = compute_forgetting(flat)

    print("\n" + "=" * 60)
    print("CL SUMMARY")
    print(f"  BWT:        {bwt:.4f}  (negative = forgetting)")
    print(f"  Forgetting: {forgetting:.4f}  (mean peak-to-final F1 drop)")

    run_dir = RUNS_DIR / f"cl_lwf_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "mode": "LwF",
                "result_matrix": result_matrix,
                "bwt": bwt,
                "forgetting": forgetting,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"\nSaved checkpoint and metrics to {run_dir}")


if __name__ == "__main__":
    main()
