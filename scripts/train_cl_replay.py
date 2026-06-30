"""Continual learning training with Experience Replay across three pill domains.

Domain sequence: tablet -> pill -> capsule.

After each task, a stratified sample of that task's training images is saved
to a replay buffer. On subsequent tasks, every training batch is augmented
with an equal-sized batch drawn from the buffer, so the model keeps seeing
past-task examples throughout later training.

Tablet is first because it has genuine defective/non_defective labels in
its train split. Pill and capsule follow the MVTec AD convention (train/
has only non-defective images), so we use 50% of their labeled test split
for CL fine-tuning and hold out the remainder for evaluation -- the same
methodological choice as scripts/train_cl_ewc.py.
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
from src.strategies import ReplayBuffer
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


def run_epoch(model, loader, device, optimizer=None, replay_buffer=None):
    train_mode = optimizer is not None
    model.train(train_mode)
    criterion = nn.CrossEntropyLoss()

    total_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if train_mode and replay_buffer is not None and len(replay_buffer) > 0:
            r_images, r_labels = replay_buffer.sample_batch(images.size(0))
            images = torch.cat([images, r_images.to(device)], dim=0)
            labels = torch.cat([labels, r_labels.to(device)], dim=0)

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


def evaluate(model, dataset, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return run_epoch(model, loader, device)


def train_task(model, train_loader, val_loader, device, epochs, lr, replay_buffer):
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    for epoch in range(1, epochs + 1):
        train_m = run_epoch(model, train_loader, device, optimizer, replay_buffer)
        val_m = run_epoch(model, val_loader, device)
        print(
            f"  epoch {epoch}/{epochs} "
            f"train_loss={train_m['loss']:.4f} train_acc={train_m['accuracy']:.4f} "
            f"val_loss={val_m['loss']:.4f} val_acc={val_m['accuracy']:.4f} val_f1={val_m['f1']:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--epochs-per-task", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=100,
                        help="Number of images to store per task in the replay buffer.")
    parser.add_argument("--cl-split-fraction", type=float, default=0.5)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: Experience Replay | buffer={args.buffer_size}/task | device={device}\n")

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
    buffer = ReplayBuffer(n_per_task=args.buffer_size)

    task_names = ["tablet", "pill", "capsule"]
    T = len(task_names)
    result_matrix = [[None] * T for _ in range(T)]
    history = []

    # ── Task 0: tablet ────────────────────────────────────────────────────────
    print("=" * 60)
    print("Task 0: tablet")
    train_loader = DataLoader(tablet_train, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(tablet_val, batch_size=args.batch_size, shuffle=False)
    train_task(model, train_loader, val_loader, device, args.epochs_per_task, args.lr, buffer)

    m = evaluate(model, tablet_test, device, args.batch_size)
    result_matrix[0][0] = m["f1"]
    print(f"  [After Task 0] tablet test: acc={m['accuracy']:.4f} f1={m['f1']:.4f}")
    history.append({"after_task": "tablet", "tablet": m})

    buffer.populate(tablet_train, seed=args.seed)
    print(f"  Buffer: {len(buffer)} images stored from tablet")

    # ── Task 1: pill ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 1: pill (CL fine-tune on 50% of labeled test split)")
    cl_train_loader = DataLoader(task_specs["pill"]["cl_train"], batch_size=args.batch_size, shuffle=True)
    cl_val_loader = DataLoader(task_specs["pill"]["val"], batch_size=args.batch_size, shuffle=False)
    train_task(model, cl_train_loader, cl_val_loader, device, args.epochs_per_task, args.lr, buffer)

    m_tablet = evaluate(model, tablet_test, device, args.batch_size)
    m_pill = evaluate(model, task_specs["pill"]["cl_eval"], device, args.batch_size)
    result_matrix[1][0] = m_tablet["f1"]
    result_matrix[1][1] = m_pill["f1"]
    print(f"  [After Task 1] tablet test: acc={m_tablet['accuracy']:.4f} f1={m_tablet['f1']:.4f}")
    print(f"  [After Task 1] pill eval:   acc={m_pill['accuracy']:.4f} f1={m_pill['f1']:.4f}")
    history.append({"after_task": "pill", "tablet": m_tablet, "pill": m_pill})

    buffer.populate(task_specs["pill"]["cl_train"], seed=args.seed)
    print(f"  Buffer: {len(buffer)} images stored total")

    # ── Task 2: capsule ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Task 2: capsule (CL fine-tune on 50% of labeled test split)")
    cl_train_loader = DataLoader(task_specs["capsule"]["cl_train"], batch_size=args.batch_size, shuffle=True)
    cl_val_loader = DataLoader(task_specs["capsule"]["val"], batch_size=args.batch_size, shuffle=False)
    train_task(model, cl_train_loader, cl_val_loader, device, args.epochs_per_task, args.lr, buffer)

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

    run_dir = RUNS_DIR / f"cl_replay_{args.backbone}_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "mode": "Experience Replay",
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
