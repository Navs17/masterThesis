"""Compare continual learning strategy results across runs.

Scans runs/ for metrics.json files produced by train_cl_ewc.py,
train_cl_replay.py, and train_cl_lwf.py.  For each strategy, the most
recent run is selected.  Outputs:

  1. Console table: Strategy | BWT | Forgetting | F1 per domain | Avg F1
  2. result_matrix heatmaps  (one panel per strategy, saved as PNG)
  3. BWT / Forgetting summary bar chart  (saved as PNG)
  4. Final-F1 per domain grouped bar chart  (saved as PNG)

All figures and a summary CSV are written to a timestamped folder under
runs/comparison_<timestamp>/.

Usage
-----
  uv run python scripts/compare_cl_results.py
  uv run python scripts/compare_cl_results.py --runs-dir /path/to/runs
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"

DOMAIN_LABELS = ["Tablet", "Pill", "Capsule"]
STRATEGY_ORDER = ["Naive Sequential", "EWC", "Experience Replay", "LwF"]
STRATEGY_COLORS = {
    "Naive Sequential": "#9e9e9e",
    "EWC":              "#1976d2",
    "Experience Replay":"#388e3c",
    "LwF":              "#f57c00",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _strategy_label(record: dict) -> str | None:
    """Derive a canonical strategy name from a metrics.json record.

    Returns None for runs that should be excluded (e.g. anomaly baselines).
    """
    mode = record.get("mode", "")
    args = record.get("args", {})
    # Normalise the EWC script's "Naive sequential" (lowercase s)
    if mode.lower() in ("naive sequential", "naive"):
        return "Naive Sequential"
    if mode == "EWC":
        return "Naive Sequential" if args.get("lambda_ewc", 1.0) == 0.0 else "EWC"
    if mode in ("Experience Replay", "LwF"):
        return mode
    return None  # anomaly baselines or other scripts → skip


def _final_f1(result_matrix: list) -> list:
    """Return the last row of the result_matrix (F1 after all tasks)."""
    return result_matrix[-1]


def _avg_final_f1(result_matrix: list) -> float:
    vals = [v for v in _final_f1(result_matrix) if v is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def _load_runs(runs_dir: Path) -> dict[str, dict]:
    """Load the most recent metrics.json per canonical strategy name."""
    candidates: dict[str, list] = {}
    for p in sorted(runs_dir.glob("*/metrics.json")):
        try:
            data = json.loads(p.read_text())
        except Exception:
            continue
        label = _strategy_label(data)
        candidates.setdefault(label, []).append((p.stat().st_mtime, data, p))

    best: dict[str, dict] = {}
    for label, runs in candidates.items():
        if label is None:
            continue
        _, data, path = max(runs, key=lambda x: x[0])
        best[label] = data
        print(f"  [{label}] loaded from {path.parent.name}")
    return best


# ── console table ─────────────────────────────────────────────────────────────

def print_table(runs: dict[str, dict]) -> None:
    header = f"{'Strategy':<22} {'BWT':>7} {'Forget':>7}  {'F1_tab':>7} {'F1_pill':>7} {'F1_cap':>7}  {'F1_avg':>7}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for label in STRATEGY_ORDER:
        if label not in runs:
            continue
        d = runs[label]
        rm = d["result_matrix"]
        finals = _final_f1(rm)
        avg = _avg_final_f1(rm)
        bwt = d.get("bwt", float("nan"))
        fgt = d.get("forgetting", float("nan"))
        f1s = [f"{v:.4f}" if v is not None else "  n/a " for v in finals]
        print(
            f"{label:<22} {bwt:>7.4f} {fgt:>7.4f}  "
            f"{f1s[0]:>7} {f1s[1]:>7} {f1s[2]:>7}  {avg:>7.4f}"
        )
    print("=" * len(header))


# ── figure 1: result-matrix heatmaps ─────────────────────────────────────────

def plot_heatmaps(runs: dict[str, dict], out_dir: Path) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in runs]
    n = len(strategies)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, label in zip(axes, strategies):
        rm = runs[label]["result_matrix"]
        T = len(DOMAIN_LABELS)
        mat = np.full((T, T), np.nan)
        for t in range(T):
            for i in range(T):
                v = rm[t][i] if t < len(rm) and i < len(rm[t]) else None
                if v is not None:
                    mat[t, i] = v

        # Gray background for the whole axes, then overlay valid cells
        ax.set_facecolor("#e0e0e0")
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlGn", aspect="equal")

        for t in range(T):
            for i in range(T):
                v = mat[t, i]
                if not np.isnan(v):
                    ax.text(i, t, f"{v:.3f}", ha="center", va="center",
                            fontsize=9, color="black" if v < 0.75 else "white")
                else:
                    ax.text(i, t, "—", ha="center", va="center",
                            fontsize=11, color="#aaaaaa")

        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(DOMAIN_LABELS, fontsize=9)
        ax.set_yticklabels([f"After {d}" for d in DOMAIN_LABELS], fontsize=9)
        ax.set_title(label, fontsize=10, fontweight="bold",
                     color=STRATEGY_COLORS.get(label, "black"))
        ax.set_xlabel("Evaluated on →", fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("Trained through →", fontsize=8)

    fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="F1 score")
    fig.suptitle("Result matrices  (F1 score: higher = better)", fontsize=12)
    out = out_dir / "heatmaps.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 2: BWT / Forgetting bar chart ─────────────────────────────────────

def plot_bwt_forgetting(runs: dict[str, dict], out_dir: Path) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in runs]
    if not strategies:
        return

    bwts = [runs[s].get("bwt", 0.0) for s in strategies]
    fgts = [runs[s].get("forgetting", 0.0) for s in strategies]
    colors = [STRATEGY_COLORS.get(s, "#888888") for s in strategies]

    x = np.arange(len(strategies))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)

    bars1 = ax1.bar(x, bwts, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    ax1.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax1.set_title("Backward Transfer (BWT)", fontsize=11)
    ax1.set_ylabel("BWT  (↑ better,  negative = forgetting)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
    for bar, v in zip(bars1, bwts):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 0.002 * np.sign(v + 1e-9),
                 f"{v:.3f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8)

    bars2 = ax2.bar(x, fgts, width=0.6, color=colors, edgecolor="white", linewidth=0.8)
    ax2.set_title("Average Forgetting", fontsize=11)
    ax2.set_ylabel("Forgetting  (↓ better,  0 = no forgetting)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(strategies, rotation=15, ha="right", fontsize=9)
    for bar, v in zip(bars2, fgts):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                 f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Continual Learning: catastrophic forgetting metrics", fontsize=12)
    out = out_dir / "bwt_forgetting.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 3: final-F1 per domain ────────────────────────────────────────────

def plot_final_f1(runs: dict[str, dict], out_dir: Path) -> None:
    strategies = [s for s in STRATEGY_ORDER if s in runs]
    if not strategies:
        return

    n_strategies = len(strategies)
    n_domains = len(DOMAIN_LABELS)
    x = np.arange(n_domains)
    width = 0.8 / n_strategies

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    for i, label in enumerate(strategies):
        rm = runs[label]["result_matrix"]
        finals = _final_f1(rm)
        vals = [v if v is not None else 0.0 for v in finals]
        offset = (i - (n_strategies - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92,
                      color=STRATEGY_COLORS.get(label, "#888888"),
                      label=label, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(DOMAIN_LABELS, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("F1 score (after all tasks)", fontsize=10)
    ax.set_title("Final F1 per domain — strategy comparison", fontsize=12)
    ax.legend(fontsize=9, loc="upper right")
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle=":")

    out = out_dir / "final_f1_by_domain.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── figure 4: F1 trajectory per domain ───────────────────────────────────────

def plot_f1_trajectory(runs: dict[str, dict], out_dir: Path) -> None:
    """Line plot showing how F1 on each domain evolves task by task."""
    strategies = [s for s in STRATEGY_ORDER if s in runs]
    if not strategies:
        return

    n_domains = len(DOMAIN_LABELS)
    fig, axes = plt.subplots(1, n_domains, figsize=(4 * n_domains, 4), constrained_layout=True)

    for d_idx, (ax, domain) in enumerate(zip(axes, DOMAIN_LABELS)):
        for label in strategies:
            rm = runs[label]["result_matrix"]
            T = len(rm)
            # Only plot from the task where this domain was first seen (d_idx)
            x_vals, y_vals = [], []
            for t in range(d_idx, T):
                v = rm[t][d_idx] if d_idx < len(rm[t]) else None
                if v is not None:
                    x_vals.append(t)
                    y_vals.append(v)
            if x_vals:
                ax.plot(x_vals, y_vals,
                        marker="o", label=label,
                        color=STRATEGY_COLORS.get(label, "#888888"),
                        linewidth=1.8, markersize=5)

        ax.set_title(f"{domain} F1", fontsize=10, fontweight="bold")
        ax.set_xlabel("After task #")
        ax.set_ylabel("F1 score")
        ax.set_xticks(range(len(DOMAIN_LABELS)))
        ax.set_xticklabels([f"T{i}\n({DOMAIN_LABELS[i]})" for i in range(len(DOMAIN_LABELS))],
                           fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.axvline(d_idx, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("F1 trajectory per domain across tasks", fontsize=12)
    out = out_dir / "f1_trajectory.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {out.name}")


# ── CSV summary ───────────────────────────────────────────────────────────────

def save_csv(runs: dict[str, dict], out_dir: Path) -> None:
    import csv
    rows = []
    for label in STRATEGY_ORDER:
        if label not in runs:
            continue
        d = runs[label]
        rm = d["result_matrix"]
        finals = _final_f1(rm)
        rows.append({
            "strategy": label,
            "bwt": d.get("bwt", ""),
            "forgetting": d.get("forgetting", ""),
            "f1_tablet": finals[0] if finals[0] is not None else "",
            "f1_pill":   finals[1] if finals[1] is not None else "",
            "f1_capsule":finals[2] if finals[2] is not None else "",
            "f1_avg": _avg_final_f1(rm),
        })
    out = out_dir / "summary.csv"
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {out.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default=str(RUNS_DIR),
                        help="Directory containing run sub-folders with metrics.json.")
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    if not runs_dir.exists():
        print(f"runs/ directory not found at {runs_dir}. Train at least one strategy first.")
        return

    print(f"\nScanning {runs_dir} for metrics.json …")
    runs = _load_runs(runs_dir)

    if not runs:
        print("\nNo metrics.json files found. Run one or more of:")
        print("  uv run python scripts/train_cl_ewc.py")
        print("  uv run python scripts/train_cl_replay.py")
        print("  uv run python scripts/train_cl_lwf.py")
        return

    print_table(runs)

    out_dir = runs_dir / f"comparison_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving figures to {out_dir} …")

    plot_heatmaps(runs, out_dir)
    plot_bwt_forgetting(runs, out_dir)
    plot_final_f1(runs, out_dir)
    plot_f1_trajectory(runs, out_dir)
    if runs:
        save_csv(runs, out_dir)

    print(f"\nDone. Open {out_dir} to see the plots.")


if __name__ == "__main__":
    main()
