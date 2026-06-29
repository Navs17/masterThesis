# pharma-cl-thesis

Continual learning for pharmaceutical pill image classification.

## Setup

```
uv sync
```

## Layout

- `src/data/` — data prep, splits, augmentation
- `src/models/` — backbones, heads
- `src/strategies/` — continual learning strategies
- `src/eval/` — metrics: accuracy, BWT, FWT, forgetting
- `src/utils/` — seeds, logging, config
- `scripts/` — entry points (`prepare_data.py`, `train_baseline.py`)
- `configs/` — experiment configs
- `data/` — raw and processed data (gitignored)
- `runs/` — outputs, checkpoints, logs (gitignored)
