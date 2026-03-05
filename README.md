# YOLO Iterative Training Pipeline

**Version:** 0.2.0 (March 2026)

Active learning pipeline for iterative YOLO training that turns annotation into continuous model improvement. Dual-purpose system: assists annotation while producing production-ready detection models.

## Features

- **Active Learning**: Automatically selects most valuable images to annotate next
- **Iterative Training**: Trains YOLO models as you annotate (auto-triggers after N images)
- **Verification Tracking**: Tracks manually verified vs. pre-labeled images separately
- **Dual Metrics**: Eval set (in-distribution) + Test set (generalization)
- **X-AnyLabeling Integration**: Load improved models directly into annotation tool
- **Small Object Optimizations**: High-resolution training (1280px) with copy-paste augmentation
- **Multi-GPU Support**: Leverages dual GPUs for faster training
- **Production Ready**: Export models in ONNX, TensorRT formats

## Quick Start

### 1. Installation

```bash
git clone <repository-url>
cd yolo-iterative-pipeline
pip install -e .
```

**Requirements:**
- Python 3.8+
- CUDA-capable GPU(s)
- ultralytics >= 8.3.0
- torch >= 2.0.0

### 2. Initialize Project

```bash
yolo-pipeline-init --project-name my_project --classes person car bicycle
```

This creates:
- `configs/pipeline_config.yaml` - Pipeline settings
- `configs/yolo_config.yaml` - YOLO training hyperparameters
- `data/` directories (raw, verified, eval, test, working, sam3_annotations)
- `models/` directories (checkpoints, deployed)
- `logs/` directory

### 3. Prepare Data

Place your data in the following structure:

```
data/
├── raw/                    # Original images
│   └── image001.jpg
├── sam3_annotations/       # SAM3 bounding boxes (YOLO format, noisy)
│   ├── image001.txt
│   └── classes.txt
├── working/                # X-AnyLabeling workspace (YOLO format)
│   ├── images/             # Images for annotation
│   ├── labels/             # Label files (YOLO format)
│   └── classes.txt
├── verified/               # Human-verified annotations (YOLO format)
│   ├── images/
│   └── labels/
└── test/                   # Pre-existing labeled data (fixed test set)
    ├── images/
    └── labels/
```

**Note:** `sam3_annotations/` contains initial noisy annotations from SAM3. The pipeline will help you clean these.

**Directory Structure:** Uses YOLO dataset format with separate `images/` and `labels/` subdirectories:
- `working/images/` + `working/labels/` → annotation workspace
- `verified/images/` + `verified/labels/` → manually verified (ready for training)
- `classes.txt` in working/ defines class names (one per line)

**Important:** Pre-labeled annotations require manual verification before training:
- Pre-labeled boxes may be too tight/loose, have false positives, or miss objects
- You manually verify/correct each image in X-AnyLabeling
- Auto-move script monitors `working/labels/`, moves to `verified/images/` + `verified/labels/`
- Verification tracker logs which images have been manually reviewed

### 4. Choose Your Workflow

**See `QUICKSTART.md` and `docs/IMPLEMENTATION_SUMMARY.md` for detailed setup instructions.**

#### Option A: Single Command (Recommended - NEW in v0.2.0)

```bash
# Launch all services with one command
yolo-pipeline-run

# Then open X-AnyLabeling
xanylabeling  # Open Dir: data/working/
```

**Features:**
- Runs health check automatically
- Manages 3 background services (auto-move, watcher, monitor)
- Graceful shutdown on Ctrl+C
- Real-time log streaming

#### Option B: Automatic Workflow (4 terminals)

**How it works:**
1. Start the auto-move script (records current timestamp)
2. Pre-labeled files (modified before script start) are **skipped**
3. You manually verify images in X-AnyLabeling and save (Ctrl+S)
4. Saving updates file modification time to NOW
5. 60 seconds after save, script automatically moves file to verified/

**Key insight:** Script only moves files modified AFTER it started = only manually reviewed files

```bash
# Terminal 1: Auto-move watcher (working/labels/ → verified/labels/)
python scripts/auto_move_verified.py

# Terminal 2: Training watcher (verified/ → training)
yolo-pipeline-watch

# Terminal 3: Status monitor
watch -n 5 yolo-pipeline-monitor

# Terminal 4: Annotation (manual verification)
xanylabeling  # Open Dir: data/working/
```

#### Option C: Manual Workflow (3 terminals)

```bash
# Terminal 1: Training watcher
yolo-pipeline-watch

# Terminal 2: Annotation
x-anylabeling  # Open Dir: data/working/

# Terminal 3: Manual move when ready
./scripts/move_verified.sh
```

#### Option D: Direct Annotation (2 terminals)

```bash
# Terminal 1: Training watcher
yolo-pipeline-watch

# Terminal 2: Annotation
x-anylabeling  # Open Dir: data/verified/ (save directly)
```

### 5. Annotate with X-AnyLabeling

1. Open X-AnyLabeling: `xanylabeling`
2. File → Open Dir → `data/working/` (X-AnyLabeling auto-detects images/ and labels/)
3. Edit → Label Settings → Verify classes (auto-loaded from classes.txt)
4. (Optional) AI → Load Model → `models/active/best.pt`
5. Review and correct annotations
6. Press `Ctrl+S` to save (updates label file in working/labels/)

**Workflow:**
- Option A (Auto): **Manual verification** with Space key + **automatic movement** based on verified flag
  - Review images in X-AnyLabeling
  - Correct boxes, delete false positives, add missing objects
  - Press **Space** to mark as verified (sets `flags.verified: true` in JSON)
  - Save (Ctrl+S) to update JSON file
  - 60s after save, script checks JSON for verified flag
  - Auto-moves image and creates YOLO label in verified/
  - **✅ Guarantees: Only images marked as verified are moved**
- Option B (Manual): Manual verification + manual batch movement with `./scripts/move_verified.sh`
- Option C (Direct): Save directly to verified/ (no intermediate step, no pre-labels)

## Workflow

### Iteration Cycle (Option A: Automatic)

```
1. Manual verification in X-AnyLabeling (data/working/)
   - Review pre-labeled boxes from SAM3
   - Correct tight/loose boxes (drag corners)
   - Delete false positives (Delete key)
   - Add missing objects (W key + draw)
   - Fix wrong class labels
   - Press Space to mark as verified
   - Save (Ctrl+S), move to next image (D key)
   ↓
2. Auto-copy script monitors (Terminal 1)
   - Monitors data/working/images/*.json files
   - Checks flags.verified == true in JSON
   - Waits 60s after JSON modification for stability
   - Converts JSON annotations → YOLO format
   - COPIES image → data/verified/images/ (original stays in working/)
   - Creates YOLO label → data/verified/labels/
   - Keeps JSON + image in working/images/ (continue annotating!)
   - Logs in verification tracker as "verified"
   ↓
3. Training watcher counts files → triggers at 50 images
   (first 3 iterations: 25 images for faster feedback)
   ↓
4. Training runs automatically
   - Samples 15% eval set (stratified by class)
   - Trains YOLO11n model (~15-20 min on dual A5500)
   - Evaluates on eval + test sets
   ↓
5. Model promotion (only if eval mAP50 improves)
   - New model → models/active/best.pt
   - Priority queue re-scored
   ↓
6. Reload model in X-AnyLabeling (AI → Load Model)
   - Better predictions for next batch!
   ↓
7. Repeat
```

**Key Points:**
- **Manual verification** = You review/correct in X-AnyLabeling
- **Automatic movement** = Script moves after 60s stability (you've moved to next image)
- `yolo-pipeline-watch` monitors `data/verified/` (NOT `data/working/`)
- `auto_move_verified.py` handles working/ → verified/ movement with verification tracking
- Training only triggers from verified/ directory
- Model promoted only if evaluation metrics improve

### Active Learning Priority

The pipeline automatically scores remaining raw images by:

```
Priority = 0.40×uncertainty + 0.35×disagreement + 0.25×diversity
```

- **Uncertainty**: Low model confidence → needs more examples
- **Disagreement**: Model vs SAM3 mismatch → model learned something new
- **Diversity**: Under-represented detection counts → ensure coverage

View priorities with:
```bash
yolo-pipeline-score --top 20
```

## CLI Commands

### Process Management (NEW in v0.2.0)

```bash
# Launch all services with one command
yolo-pipeline-run [--no-doctor] [--no-auto-move] [--debug]

# Health check before starting
yolo-pipeline-doctor
```

**What `yolo-pipeline-run` does:**
1. Runs `yolo-pipeline-doctor` (validates structure, configs, annotations, model)
2. Launches 3 services: auto-move watcher, training watcher, status monitor
3. Shows real-time logs from all services
4. Handles graceful shutdown on Ctrl+C (SIGINT/SIGTERM)
5. Cleans up processes on exit

### Helper Scripts

```bash
# View setup instructions for 4-terminal workflow
./scripts/start_pipeline.sh

# Automatic file copying (JSON-based verification)
# Monitors working/images/*.json for verified flag
# Checks flags.verified == true, converts JSON → YOLO format
# COPIES image to verified/images/ (keeps original in working/)
# Creates label in verified/labels/
# Requires: data/verified/classes.txt for class mapping
python scripts/auto_move_verified.py [--interval 60] [--stability 60]

# Check verification status (verified vs unverified images)
python scripts/track_verification.py
python scripts/track_verification.py --list-unverified
python scripts/track_verification.py --scan  # Scan working directory

# Manual batch movement (Option B)
./scripts/move_verified.sh              # Interactive mode
./scripts/move_verified.sh --all        # Move all files
./scripts/move_verified.sh pattern      # Move files matching pattern
```

### Initialize Project
```bash
yolo-pipeline-init --project-name <name> --classes <class1> <class2> ...
```

### Start File Watcher
```bash
yolo-pipeline-watch [--config configs/pipeline_config.yaml]
```

Monitors `data/verified/` and auto-triggers training when threshold reached.

### Manual Training
```bash
yolo-pipeline-train [--config configs/yolo_config.yaml]
```

Trains YOLO model on current `data/verified/` set.

### Monitor Status
```bash
yolo-pipeline-monitor [--watch]
```

Shows:
- Current iteration
- Image counts (verified, eval, test)
- Latest model metrics (mAP50, F1, precision, recall)
- Training history

Add `--watch` for real-time updates.

### Score & Prioritize
```bash
yolo-pipeline-score [--top N] [--rescore]
```

- `--top N`: Show top N priority images
- `--rescore`: Re-calculate priorities with latest model

### Export Models
```bash
yolo-pipeline-export --version v007 --formats onnx tensorrt
```

Exports trained model to production formats.

## Directory Structure

```
yolo-iterative-pipeline/
├── configs/
│   ├── pipeline_config.yaml    # Pipeline settings
│   └── yolo_config.yaml         # YOLO hyperparameters
├── data/
│   ├── raw/                     # Original images
│   ├── sam3_annotations/        # SAM3 boxes (noisy)
│   ├── working/                 # X-AnyLabeling workspace
│   ├── verified/                # Human-verified annotations
│   ├── eval/                    # Auto-sampled 10-15% from verified
│   └── test/                    # Pre-existing labeled data (fixed)
├── models/
│   ├── checkpoints/             # Training checkpoints
│   └── deployed/                # Production-ready models
├── logs/
│   ├── training.log             # Training logs
│   ├── watcher.log              # File watcher logs
│   └── training_history.json    # Metrics over time
├── notebooks/
│   └── analysis.ipynb           # Metrics visualization
└── pipeline/                    # Source code
```

## Configuration

### Pipeline Config (`configs/pipeline_config.yaml`)

```yaml
project_name: "my_project"
classes:
  - "person"
  - "car"
  - "bicycle"

# Trigger settings
trigger_threshold: 50        # Train after N new verified images
early_trigger: 25            # Use lower threshold early on
min_train_images: 50         # Minimum images to start training

# Data splits
eval_split_ratio: 0.15       # 15% of verified → eval set
stratify: true               # Balance classes in eval split

# Active learning weights (must sum to 1.0)
uncertainty_weight: 0.40
disagreement_weight: 0.35
diversity_weight: 0.25

# Notifications
desktop_notify: true         # Desktop notifications (Linux only)
slack_webhook: null          # Optional Slack webhook URL

# Cleanup
keep_last_n_checkpoints: 10  # Keep only last N checkpoints
```

### YOLO Config (`configs/yolo_config.yaml`)

```yaml
model: "yolo11n.pt"          # Base model
epochs: 50
batch_size: 16
imgsz: 1280                  # High-res for small objects
device: [0, 1]               # Multi-GPU training
patience: 10                 # Early stopping patience

# Small object optimizations
close_mosaic: 10             # Disable mosaic in final 10 epochs
copy_paste: 0.5              # Duplicate small objects (50% chance)
mixup: 0.1

# Standard augmentation
scale: 0.9
fliplr: 0.5
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
mosaic: 1.0
```

## Expected Performance

| Iteration | Verified Images | Eval mAP50 | Test mAP50 | Status |
|-----------|----------------|------------|------------|--------|
| 1-2       | 100-200        | 0.75-0.82  | 0.70-0.78  | Model becomes useful |
| 3-5       | 300-500        | 0.82-0.88  | 0.78-0.85  | Production-ready with oversight |
| 6+        | 700+           | 0.88-0.93  | 0.85-0.90  | Autonomous deployment |

**Training Time:** ~15-20 minutes per iteration (dual A5500 GPUs, 1280px images)

## Advanced Usage

### Bootstrap Training

**Option 1: Train on noisy SAM3 annotations first**
```bash
# Copy SAM3 annotations to verified/
cp data/sam3_annotations/*.txt data/verified/labels/
cp data/raw/*.jpg data/verified/images/

# Trigger training
yolo-pipeline-train
```

**Option 2: Manually clean 50-100 images first** (recommended)
```bash
# Clean annotations in X-AnyLabeling, save to working/
# Watcher auto-moves to verified/
# Training auto-triggers at 50 images
```

### Adjust Trigger Threshold

Early iterations benefit from faster feedback:

```yaml
# configs/pipeline_config.yaml
early_trigger: 25      # First 2-3 iterations
trigger_threshold: 50  # Mid iterations
# Late iterations: increase to 100 (diminishing returns)
```

### Custom Active Learning Weights

Adjust if specific signal more important:

```yaml
# Increase uncertainty if model very uncertain
uncertainty_weight: 0.50
disagreement_weight: 0.30
diversity_weight: 0.20

# Increase disagreement if SAM3 systematically wrong
uncertainty_weight: 0.30
disagreement_weight: 0.50
diversity_weight: 0.20

# Increase diversity if dataset very imbalanced
uncertainty_weight: 0.30
disagreement_weight: 0.30
diversity_weight: 0.40
```

### Production Deployment

Models can be deployed during annotation (not just at end):

- **After iteration 2-3**: Good enough for screening, low-stakes detection
- **After iteration 4-5**: Production with human oversight
- **After iteration 7+**: Autonomous deployment

```bash
# Export to ONNX
yolo-pipeline-export --version v005 --formats onnx

# Export to TensorRT (optimized for A5500)
yolo-pipeline-export --version v005 --formats tensorrt
```

### Starting a New Project

To reset the pipeline for a completely different project (new classes/dataset):

```bash
# Quick reset (backs up everything automatically)
./scripts/reset_for_new_project.sh

# Then update classes
cat > data/verified/classes.txt << 'EOC'
person
car
dog
EOC
cp data/verified/classes.txt data/working/classes.txt

# Add new images and train from scratch
source venv/bin/activate
python -m pipeline.train --from-scratch
```

**What the reset script does:**
- Backs up existing data/models/logs to `backups/TIMESTAMP/`
- Clears verified data, active models, and checkpoints
- Resets training history and verification tracking
- Removes YOLO cache files

**Training flags:**
- No flag: Resume from active model (smart resume)
- `--from-scratch`: Start fresh from pretrained YOLO11n

See `QUICKSTART.md` for detailed reset instructions.

### Health Check

```bash
yolo-pipeline-monitor --health-check
```

Checks:
- Configuration validity
- Directory structure
- Training lock status
- Latest model metrics
- Data distribution

## Troubleshooting

### Training Fails with OOM Error

**Problem:** GPU out of memory

**Solution 1:** Reduce batch size
```yaml
# configs/yolo_config.yaml
batch_size: 8  # or 4
```

**Solution 2:** Reduce image size
```yaml
# configs/yolo_config.yaml
imgsz: 640  # from 1280
```

The pipeline auto-retries with `batch_size/2` on OOM.

### Watcher Not Triggering Training

**Problem:** Verified images not triggering training

**Check 1:** Verify threshold
```bash
# How many verified images?
ls data/verified/images/*.jpg | wc -l

# Check threshold
grep trigger_threshold configs/pipeline_config.yaml
```

**Check 2:** Training lock
```bash
# Remove stale lock if training crashed
rm logs/.training.lock
```

**Check 3:** Watcher logs
```bash
tail -f logs/watcher.log
```

### Model Not Improving

**Problem:** Eval mAP plateauing or decreasing

**Cause 1:** Not enough diversity
```bash
# Check active learning priorities
yolo-pipeline-score --top 50

# Focus on high-priority images
```

**Cause 2:** Annotation quality issues
```bash
# Review recent annotations
ls -lt data/verified/labels/*.txt | head -20
```

**Cause 3:** Overfitting
```yaml
# Increase eval split
eval_split_ratio: 0.20  # from 0.15
```

### Corrupted Annotations

**Problem:** Training crashes with invalid annotations

**Solution:** The pipeline automatically skips corrupted files and logs them:
```bash
grep "Skipping invalid" logs/training.log
```

Remove or fix corrupted annotations:
```bash
# Find empty label files
find data/verified/labels -name "*.txt" -empty

# Find labels with invalid class IDs
grep -r "^[3-9]" data/verified/labels/  # if only 3 classes (0,1,2)
```

### X-AnyLabeling Not Loading Model

**Problem:** Cannot load trained model in X-AnyLabeling

**Solution:** X-AnyLabeling requires `.pt` format:
```bash
# Models are already in .pt format in models/checkpoints/
ls models/checkpoints/*.pt

# Load best.pt or latest.pt
```

If model still fails:
```bash
# Export to ONNX and load that
yolo-pipeline-export --version v003 --formats onnx
# Load models/deployed/v003.onnx in X-AnyLabeling
```

## Metrics Tracked

**Per Model:**
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, F1 Score
- Both eval and test sets
- Per-class breakdown

**Training History:**
- Version, timestamp, image counts
- Metrics progression over iterations
- Improvement over previous model
- Training time

View in Jupyter notebook:
```bash
jupyter notebook notebooks/analysis.ipynb
```

## Success Criteria

**Technical:**
- Eval mAP50 > 0.85, F1 > 0.80
- Test mAP50 > 0.82, F1 > 0.77
- Per-class recall > 0.75

**Operational:**
- 30-50% reduction in annotation time vs manual
- Production model ready at 300-500 images (vs 1500+ manual)
- Pipeline uptime > 95%

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{yolo_iterative_pipeline,
  title = {YOLO Iterative Training Pipeline},
  author = {Your Name},
  year = {2026},
  url = {https://github.com/yourusername/yolo-iterative-pipeline}
}
```

## License

MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Support

For issues and questions:
- GitHub Issues: [repository-url]/issues
- Documentation: `docs/plans/`
- Email: your.email@example.com

---

**Hardware Used:** 2x NVIDIA A5500 GPUs (excellent for fast iteration)

**Related Tools:**
- [X-AnyLabeling](https://github.com/CVHub520/X-AnyLabeling) - Annotation tool with YOLO model loading
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - YOLO11 detection framework
- [SAM](https://github.com/facebookresearch/segment-anything) - Segment Anything Model
