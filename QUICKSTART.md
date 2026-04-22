# Quick Start

## 1. Enable the Verified Flag (First-Time Setup)

The verified checkbox in X-AnyLabeling requires a one-time config change. Edit `~/.xanylabelingrc` and add:

```yaml
flags:
- verified
```

Or run this one-liner:

```bash
python3 -c "
import os, yaml
rc = os.path.expanduser('~/.xanylabelingrc')
cfg = yaml.safe_load(open(rc)) if os.path.exists(rc) else {}
cfg['flags'] = ['verified']
yaml.dump(cfg, open(rc, 'w'), default_flow_style=False)
print('Done —', rc)
"
```

Restart X-AnyLabeling. A **flag panel** with a "verified" checkbox will appear on the side.

### Change Bounding Box Line Width

If the default bounding boxes are too thin to see clearly, edit `~/.xanylabelingrc` and change the `line_width` value under the `shape` section:

```yaml
shape:
  line_width: 3
```

Increase the value for thicker borders (default is 1). Restart X-AnyLabeling to apply.

---

## 2. Start the Pipeline

```bash
source venv/bin/activate
yolo-pipeline-run
```

This runs a health check then launches the auto-move watcher and training watcher in the background.

---

## 3. Open X-AnyLabeling

```bash
xanylabeling
```

### Open your working directory

`File` → `Open Dir` → select `data/working/`

### Load annotations

`Upload` → `Upload YOLO HBB Annotations`

1. Select the classes file: `data/working/classes.txt`
2. Select the labels directory: `data/working/labels/`

### Load the model (optional — skip on first run)

`AI` → `Load Model` → select `models/active/best.onnx`

---

## 4. Annotate

For each image:

1. Review the pre-labeled boxes — correct any mistakes
   - Drag corners to adjust tight/loose boxes
   - Press `Delete` to remove false positives
   - Press `R` and draw to add missing objects
2. Press **`Space`** to mark as verified (checks the flag)
3. Press **`Ctrl+S`** to save
4. Press **`D`** to go to the next image

After 60 seconds, the auto-move script detects the verified flag and copies the image + label to `data/verified/`. Training triggers automatically at 50 verified images.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Space` | Mark as verified |
| `Ctrl+S` | Save |
| `D` / `A` | Next / previous image |
| `R` | Draw new bounding box |
| `Delete` | Remove selected box |
| `Ctrl+Z` | Undo |

---

## 5. Reload Model After Training

When training completes, a new model is promoted to `models/active/best.onnx` (only if mAP improved).

In X-AnyLabeling: `AI` → `Load Model` → select `models/active/best.onnx`

---

## Monitoring

```bash
# Check verified count and training status
yolo-pipeline-monitor

# Check auto-move is working
tail -f logs/auto_move.log

# Verify the flag is being set in JSON files
grep -l '"verified": true' data/working/images/*.json | wc -l
```

---

## Troubleshooting

**Auto-move not triggering:**
- Did you press `Space` and then `Ctrl+S`?
- Wait 60 seconds after saving
- Check `tail -f logs/auto_move.log`

**Training not triggering:**
- Check `ls data/verified/images/ | wc -l` — need 50 images (25 for first 3 iterations)
- Check `tail -f logs/watcher.log`
- Remove stale lock: `rm logs/.training.lock`

**ONNX fails to load in X-AnyLabeling (version mismatch):**
```bash
yolo-pipeline-export --formats onnx
```
Then reload `models/active/best.onnx`.

**Classes changed / cache errors:**
```bash
find data -name "*.cache" -delete
```

---

## Reset Pipeline

**Retrain from scratch** (keeps all data, clears models and training state):

```bash
./scripts/reset_for_new_project.sh --models-only
```

**Full reset for a new project** (clears data, models, and training state):

```bash
./scripts/reset_for_new_project.sh
```

Both options back up checkpoints and training history to `backups/` before clearing, and rebuild `models/active/config.yaml` with classes from `data/verified/classes.txt`.

---

## Manual Model Promotion

If you want to force-promote a specific checkpoint:

```bash
# List available versions and metrics
yolo-pipeline-monitor --history

# Update active model symlink
ln -sf $(pwd)/models/checkpoints/<run_name>/weights/best.pt models/active/best.pt

# Re-export ONNX
yolo-pipeline-export --formats onnx
```
