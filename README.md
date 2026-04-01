# YOLO Iterative Training Pipeline

Active learning pipeline for iterative YOLO training. Annotate in X-AnyLabeling, pipeline auto-trains as you go, and each new model feeds better predictions back into your annotation tool.

## Installation

```bash
git clone <repository-url>
cd yolo-iterative-pipeline
pip install -e .
```

**Requirements:** Python 3.8+, `ultralytics >= 8.3.0`, `torch >= 2.0.0`

**X-AnyLabeling** ([get started guide](https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md)):
```bash
pip install "x-anylabeling-cvhub[cpu]"   # CPU / Mac
pip install "x-anylabeling-cvhub[gpu]"   # NVIDIA GPU
```

> If `xanylabeling` command is not found after install, reinstall with `pip install "x-anylabeling-cvhub[cpu]" --force-reinstall --no-deps`

## Quick Start

```bash
# 1. Start the pipeline (runs health check + all background services)
source venv/bin/activate
yolo-pipeline-run

# 2. Open annotation tool in a separate terminal
xanylabeling
```

See `QUICKSTART.md` for X-AnyLabeling setup and annotation workflow.

## How It Works

```
Annotate in X-AnyLabeling → press Space (verified) → auto-copied to data/verified/
→ pipeline triggers training at 50 images → new model promoted if mAP improves
→ reload model in X-AnyLabeling → better predictions → repeat
```

- **Trigger:** Training starts automatically after every 50 verified images (25 for first 3 iterations)
- **Promotion:** New model only replaces active model if eval mAP50 improves
- **Active learning:** Pipeline scores remaining images by uncertainty + disagreement + diversity so you annotate the most valuable images first

## CLI Reference

| Command | Description |
|---------|-------------|
| `yolo-pipeline-run` | Launch all services (recommended) |
| `yolo-pipeline-doctor` | Health check — validates structure, configs, model |
| `yolo-pipeline-watch` | File watcher only (monitors `data/verified/`) |
| `yolo-pipeline-monitor` | Status dashboard |
| `yolo-pipeline-train` | Manual training trigger |
| `yolo-pipeline-export` | Export model (ONNX, TensorRT) |

```bash
yolo-pipeline-run --no-doctor    # Skip health check
yolo-pipeline-run --debug        # Verbose logging
yolo-pipeline-export --version v005 --formats onnx
```

## Directory Structure

```
data/
├── working/        # X-AnyLabeling workspace (images/ + labels/)
├── verified/       # Human-verified annotations (triggers training)
└── test/           # Optional fixed test set for generalization metrics

models/
├── active/         # Current best model (best.pt, best.onnx)
└── checkpoints/    # All training runs

configs/
├── pipeline_config.yaml   # Trigger threshold, eval split, active learning weights
└── yolo_config.yaml       # YOLO hyperparameters (epochs, batch, imgsz, device)

logs/
├── training_history.json  # Metrics across all iterations
└── verification_status.json
```

## Configuration

**`configs/pipeline_config.yaml`** — key settings:
```yaml
trigger_threshold: 50      # Train after N new verified images
eval_split_ratio: 0.15     # 15% of verified → eval set
```

**`configs/yolo_config.yaml`** — key settings:
```yaml
model: "yolo11n.pt"
imgsz: 1280       # High-res for small objects
device: [0, 1]    # GPU ids; auto-falls back to MPS or CPU if unavailable
batch_size: 16
```

## Troubleshooting

**Training not triggering:**
```bash
ls data/verified/images/ | wc -l   # check verified count vs trigger_threshold
tail -f logs/watcher.log
rm logs/.training.lock              # remove stale lock if training crashed
```

**OOM during training:** Reduce `batch_size` (or `imgsz`) in `configs/yolo_config.yaml`. Pipeline auto-retries at `batch_size/2`.

**ONNX version mismatch in X-AnyLabeling:** Re-export with `yolo-pipeline-export --formats onnx` (pipeline uses opset 17 for compatibility).

**Stale YOLO cache after changing classes:**
```bash
find data -name "*.cache" -delete
```

## New Project Reset

```bash
./scripts/reset_for_new_project.sh   # backs up current data, clears pipeline state
```

Then update `data/verified/classes.txt` and `data/working/classes.txt` with new class names, add images, and run `yolo-pipeline-run`.
