# YOLO Iterative Training Pipeline for Assisted Annotation

**Date:** 2026-02-26
**Author:** Design Session with User
**Status:** Approved

## Executive Summary

An active learning pipeline that transforms the annotation process into a continuous model improvement cycle. Starting with noisy SAM3 bounding box annotations, the system trains YOLO26 models that progressively improve as the user cleans annotations. The trained models serve dual purposes: assisting with annotation through X-AnyLabeling integration and serving as production-ready detection models.

**Key Innovation:** Annotation effort directly produces production models - not just training data.

---

## 1. Architecture Overview

### System Components

The pipeline consists of four main components operating in a continuous loop:

1. **Annotation Frontend (X-AnyLabeling)** - User cleans annotations with YOLO model assistance
2. **File Watcher & Trigger Service** - Monitors verified annotations, triggers training automatically
3. **Training Pipeline** - Trains YOLO26, evaluates on eval/test sets, versions models
4. **Model Management & Registry** - Tracks model versions, metrics, deployment artifacts

### Directory Structure

```
yolo-iterative-pipeline/
├── data/
│   ├── raw/                    # Original images
│   ├── sam3_annotations/       # Initial SAM3 bounding boxes (YOLO format)
│   ├── working/                # Current annotations being cleaned (X-AnyLabeling workspace)
│   ├── verified/               # Human-verified annotations (triggers training)
│   ├── eval/                   # Auto-sampled from verified (10-15%, held out from training)
│   └── test/                   # Pre-existing labeled data (fixed benchmark)
├── models/
│   ├── checkpoints/            # Versioned model weights (model_v001/, model_v002/, ...)
│   ├── active/                 # Symlink to current best model (X-AnyLabeling loads this)
│   └── deployed/               # Production exports (ONNX, TensorRT, TorchScript)
├── pipeline/
│   ├── watcher.py              # Monitors verified/ folder, triggers training
│   ├── train.py                # Training orchestration + eval/test evaluation
│   ├── active_learning.py      # Scores images for prioritization
│   ├── export.py               # Model export to deployment formats
│   └── utils.py                # Data splitting, metrics calculation
├── configs/
│   ├── pipeline_config.yaml    # Trigger thresholds, paths, sampling ratios
│   └── yolo_config.yaml        # YOLO26 hyperparameters
├── logs/
│   ├── training_history.json   # Tracks: version, timestamp, metrics, image counts
│   ├── priority_queue.txt      # Ordered list of images to clean next
│   ├── pipeline_errors.log     # Error logging
│   └── watcher.log             # File watcher activity
└── notebooks/
    └── analysis.ipynb          # Visualize metrics, compare versions
```

### Key Architectural Decisions

- **Separate eval and test sets**: Eval sampled from verified data (in-distribution), test is fixed pre-existing labels (generalization)
- **Symlink-based active model**: X-AnyLabeling always points to `models/active/`, pipeline updates symlink when model improves
- **Event-driven training**: File watcher detects new verified annotations, triggers training automatically
- **Versioned checkpoints**: Every training run preserved with complete metadata
- **YOLO format throughout**: Consistent annotation format from SAM3 → working → verified

---

## 2. Data Flow & Workflow

### Initial Setup (One-time)

1. Place pre-existing labeled images in `data/test/` (images + YOLO format labels)
2. Place all raw images in `data/raw/`
3. Place SAM3 annotations in `data/sam3_annotations/` (YOLO format .txt files)
4. Copy SAM3 annotations to `data/working/` (X-AnyLabeling workspace)
5. Train initial YOLO26 model on SAM3 annotations (quick baseline, tolerates noise)

### Continuous Iteration Loop

```
┌─────────────────────────────────────────────────────────┐
│ 1. Active Learning Prioritization                       │
│    - pipeline/active_learning.py scores all images in   │
│      working/ based on:                                  │
│        • Model confidence (low = uncertain)              │
│        • Disagreement with SAM3 (learned something new)  │
│        • Detection density (outliers = interesting)      │
│    - Writes priority_queue.txt (ordered list)           │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 2. Human Annotation (X-AnyLabeling)                     │
│    - Load current model from models/active/             │
│    - Open images in priority order from working/        │
│    - Model auto-annotates, user fixes errors:           │
│        • Delete false positives                          │
│        • Resize incorrect boxes                          │
│        • Draw new boxes for missed objects               │
│    - Save to working/ (auto-moved to verified/)         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 3. Auto-Trigger (File Watcher)                          │
│    - pipeline/watcher.py monitors verified/ folder      │
│    - When count increases by N (e.g., 50 images):       │
│        • Sample 10-15% to eval/ (stratified)            │
│        • Trigger training pipeline                       │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 4. Training Pipeline                                     │
│    - Train YOLO26 on verified/ minus eval/ samples      │
│    - Evaluate on eval/ set (in-distribution)            │
│    - Evaluate on test/ set (generalization)             │
│    - Save checkpoint: models/checkpoints/model_vXXX/    │
│    - Log metrics to training_history.json               │
│    - If eval mAP improved: update models/active/ symlink│
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ 5. Model Update & Re-prioritization                     │
│    - X-AnyLabeling detects new model in active/         │
│    - User reloads model (better predictions)            │
│    - Pipeline re-scores priority queue with new model   │
│    - Next images benefit from improved predictions      │
└─────────────────┬───────────────────────────────────────┘
                  │
                  └──────────► Back to Step 1
```

### Stopping Conditions

User decides when to stop based on:
- **Eval mAP plateaus**: No improvement in last 2-3 iterations
- **Test mAP meets requirement**: Production threshold reached (e.g., ≥ 0.85)
- **All priority images cleaned**: High-value annotations complete
- **Time/budget constraints**: Project timeline reached

---

## 3. X-AnyLabeling Integration

### Why X-AnyLabeling

- Native YOLO model loading (YOLOv5, YOLOv8, YOLO11, YOLO26)
- Auto-annotation with custom models
- Fast keyboard shortcuts for correction
- Saves directly in YOLO format
- Displays confidence scores
- Lightweight, runs locally

### Workspace Configuration

**Model Loading:**
- X-AnyLabeling loads `models/active/best.pt`
- Pipeline updates symlink after successful training
- User reloads model: Model → Load Custom Model

**Project Setup:**
- Open `data/working/` as project folder
- Reference `logs/priority_queue.txt` for image order
- Save format: YOLO (auto-configured)

### Save Workflow: Confidence-Based Auto-Move

**How It Works:**
1. X-AnyLabeling saves annotations to `data/working/`
2. Background script watches `working/` folder
3. Auto-moves to `data/verified/` if:
   - File has been modified (timestamp change)
   - At least one annotation exists
   - Image has corresponding .txt label file

**Benefits:**
- ✅ Automated - no manual file copying
- ✅ Safe - only moves files you've actually touched
- ✅ Clear signal - saving = verification
- ✅ Reversible - can move files back from verified/ if needed

### Priority Queue Integration

**priority_queue.txt Format:**
```
# Generated: 2026-02-26 14:32:01, Model: v003
# Format: filename | priority_score | uncertainty | disagreement | diversity
IMG_0453.jpg | 0.847 | 0.92 | 0.81 | 0.73
IMG_1203.jpg | 0.801 | 0.88 | 0.75 | 0.79
IMG_0089.jpg | 0.776 | 0.71 | 0.93 | 0.62
...
```

**Usage:**
- Open images in order from top to bottom
- Optional: Simple script to auto-load next image in X-AnyLabeling
- Optional: Visual progress indicator (e.g., "Image 127/1500")

---

## 4. Active Learning Prioritization

### Goal

Show the most valuable images first - images that will teach the model the most.

### Scoring Strategy

Each image in `data/working/` receives a combined priority score:

**1. Uncertainty Score (40% weight):**
- Run current model on image
- Calculate average confidence across all predictions
- Low confidence = high uncertainty = high priority
- **Formula:** `uncertainty = 1 - mean(confidence_scores)`

**2. Disagreement Score (35% weight):**
- Compare model predictions vs SAM3 annotations
- Count: missed detections, extra detections, IoU mismatches
- High disagreement = model learned something different = high learning value
- **Formula:** `disagreement = (num_missed + num_extra + num_low_iou) / max(model_boxes, sam3_boxes)`

**3. Diversity Score (25% weight):**
- Track distribution of detection counts (images with 0, 1, 5, 20 objects)
- Prioritize under-represented bins
- Ensures model learns full range of scenarios
- **Formula:** `diversity = 1 / frequency_of_detection_bin`

**Combined Priority:**
```python
priority = 0.40 × uncertainty + 0.35 × disagreement + 0.25 × diversity
```

### Re-scoring Frequency

- After each training iteration (new model = new uncertainties)
- Verified images removed from queue
- Priority queue rewritten to `logs/priority_queue.txt`

### Edge Cases

- **Iteration 0** (no trained model yet): Use random order or SAM3 disagreement only
- **Working folder nearly empty**: Optionally pull more images from raw/ with SAM3 annotations
- **Model predicts nothing**: Fall back to SAM3 disagreement only, log warning

---

## 5. Training Pipeline

### YOLO26 Configuration

**Model Selection:**
- **YOLO26n** (nano) or **YOLO26s** (small)
- Training time: 15-20 minutes per iteration on 2x A5500 GPUs
- Built-in features:
  - **STAL**: Small-Target-Aware Label Assignment
  - **ProgLoss**: Progressive Loss Balancing
  - **MuSGD**: Hybrid optimizer (SGD + Muon)

### Training Hyperparameters

```yaml
# configs/yolo_config.yaml
model: yolo26n.pt          # or yolo26s.pt
epochs: 50
batch_size: 16             # Smaller batch for high resolution
imgsz: 1280                # High resolution for small objects (vs 640)
device: [0, 1]             # Use both A5500 GPUs
patience: 10               # Early stopping

# Small object detection optimizations
close_mosaic: 10           # Disable mosaic in last 10 epochs
copy_paste: 0.5            # Copy-paste augmentation for small objects
mixup: 0.1                 # Mixup augmentation
scale: 0.9                 # Aggressive scaling (0.5 → 1.5x)
fliplr: 0.5                # Horizontal flip

# Standard augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
mosaic: 1.0
```

**Key Settings for Small Objects:**
- ✅ **imgsz: 1280** - Doubled resolution (4x more pixels than 640)
- ✅ **copy_paste: 0.5** - Duplicates small objects in training
- ✅ **STAL** - Built into YOLO26, automatically improves small object detection
- ✅ **close_mosaic: 10** - Disables mosaic in final epochs (can hurt small object precision)

### Training Process

**1. Data Preparation:**
- Create train/eval split from verified annotations
- Eval set: 10-15% sampled, stratified by class if possible
- Generate `data.yaml` with paths and class names
- Validate all annotations (check for malformed files)

**2. Training Execution:**
- Multi-GPU training with DDP (Distributed Data Parallel)
- Monitor loss curves (logged to `logs/train_vXXX/`)
- Save best model based on eval mAP

**3. Evaluation:**
- **Eval Set:** mAP@0.5, mAP@0.5:0.95, precision, recall, F1
- **Test Set:** Same metrics for generalization check
- Generate confusion matrix and PR curves

**4. Model Promotion:**
- Only promote to `models/active/` if eval mAP improves
- Keep all checkpoints in `models/checkpoints/` with metadata
- Log promotion or no-change event

### Training Trigger Logic

```python
# pipeline/watcher.py (pseudo-code)
while True:
    verified_count = count_files('data/verified/')

    if verified_count >= last_train_count + TRIGGER_THRESHOLD:
        # Sample eval set (10-15% of verified, stratified)
        sample_eval_set()

        # Launch training
        train_model()

        # Evaluate on eval and test
        eval_metrics = evaluate(model, 'data/eval/')
        test_metrics = evaluate(model, 'data/test/')

        # Calculate F1 scores
        eval_metrics['f1'] = 2 * (P * R) / (P + R)
        test_metrics['f1'] = 2 * (P * R) / (P + R)

        # Log results
        log_metrics(version, eval_metrics, test_metrics, verified_count)

        # Update active model if improved
        if eval_metrics['mAP50'] > best_mAP:
            update_active_model(model)
            notify_user("New model v{} active: mAP={:.3f}".format(version, mAP))

        # Re-score priority queue with new model
        update_priority_queue()

        last_train_count = verified_count

    sleep(60)  # Check every minute
```

**Trigger Threshold:**
- Default: 50 new verified images
- Configurable in `pipeline_config.yaml`
- Early iterations: 25 images (faster feedback)
- Later iterations: 100 images (diminishing returns)

### Expected Training Times

- **1280px resolution with batch_size=16**: ~15-20 minutes per run on 2x A5500
- **Slower than 640px** (5-10 min) but necessary for small objects
- **User workflow**: Clean 50 images (~1-2 hours) → take coffee break → model ready

---

## 6. Model Management & Versioning

### Model Versioning Strategy

Each training iteration produces a versioned model with complete metadata.

### Checkpoint Structure

```
models/checkpoints/model_v003_20260227_091022/
├── weights/
│   ├── best.pt              # Best performing on eval set
│   └── last.pt              # Last epoch checkpoint
├── metadata.json            # Complete training info
├── results.png              # Loss curves, mAP plots
├── confusion_matrix.png     # Confusion matrix on eval set
└── PR_curve.png             # Precision-Recall curve
```

### Metadata Format

```json
{
  "version": "v003",
  "timestamp": "2026-02-27T09:10:22",
  "model_type": "yolo26n",
  "training_data": {
    "total_verified": 387,
    "train_images": 329,
    "eval_images": 58,
    "test_images": 200,
    "classes": ["object_class_1", "object_class_2"]
  },
  "hyperparameters": {
    "epochs": 50,
    "batch_size": 16,
    "imgsz": 1280,
    "augmentation": "full"
  },
  "metrics": {
    "eval": {
      "mAP50": 0.847,
      "mAP50-95": 0.612,
      "precision": 0.881,
      "recall": 0.792,
      "f1": 0.834
    },
    "test": {
      "mAP50": 0.791,
      "mAP50-95": 0.557,
      "precision": 0.823,
      "recall": 0.748,
      "f1": 0.784
    }
  },
  "training_time_minutes": 18.3,
  "improvement_over_previous": {
    "eval_mAP50": +0.042,
    "eval_f1": +0.037,
    "test_mAP50": +0.031,
    "test_f1": +0.029
  },
  "deployed": false,
  "notes": "First iteration with YOLO26, significant improvement in small object recall"
}
```

### Active Model Management

```
models/active/
├── best.pt -> ../checkpoints/model_v003_20260227_091022/weights/best.pt
└── metadata.json -> ../checkpoints/model_v003_20260227_091022/metadata.json
```

**Promotion Logic:**
1. Training completes, model evaluated on eval and test
2. If `eval_mAP50` > current active model's `eval_mAP50`:
   - Update symlink in `models/active/`
   - Log promotion event
   - Trigger priority queue re-scoring
   - Notify user (desktop notification or terminal message)
3. If no improvement:
   - Keep checkpoint but don't promote
   - Normal in later stages (diminishing returns)

### Deployment Export

When model is production-ready:

```bash
python pipeline/export.py --version v003 --formats onnx tensorrt torchscript
```

Creates:
```
models/deployed/model_v003/
├── model_v003.onnx           # For ONNX Runtime
├── model_v003.engine         # For TensorRT (optimized for A5500)
├── model_v003.torchscript    # For LibTorch/C++
└── deployment_info.json      # Export settings, benchmark results
```

### Training History

```json
// logs/training_history.json
[
  {
    "version": "v001",
    "timestamp": "2026-02-26T14:32:01",
    "train_images": 150,
    "eval_mAP50": 0.673,
    "eval_f1": 0.691,
    "test_mAP50": 0.612,
    "test_f1": 0.628,
    "notes": "Baseline trained on SAM3 annotations"
  },
  {
    "version": "v002",
    "timestamp": "2026-02-26T17:15:43",
    "train_images": 238,
    "eval_mAP50": 0.805,
    "eval_f1": 0.798,
    "test_mAP50": 0.760,
    "test_f1": 0.752,
    "improvement": "+0.132 eval mAP, +0.107 eval F1, +0.148 test mAP"
  }
]
```

### Rollback Capability

- All checkpoints preserved indefinitely (or auto-cleanup after N versions)
- Manual rollback: `ln -sf checkpoints/model_v002/.../best.pt active/best.pt`
- Useful if newer model performs worse on specific use cases

---

## 7. Error Handling & Edge Cases

### Training Failures

**OOM (Out of Memory):**
- Auto-retry with reduced batch size (16 → 8 → 4)
- Log warning if batch size < 4
- Fallback: suggest reducing imgsz (1280 → 1024 → 640)

**Corrupted Annotations:**
- Pre-training validation: check all YOLO format files
- Skip malformed files, log to `logs/corrupted_files.txt`
- Continue training with valid subset
- Notify user to fix corrupted files

**Training Crash:**
- YOLO auto-saves checkpoints every N epochs
- Resume from last checkpoint if interrupted
- Lock file prevents simultaneous training runs

### File System Issues

**Disk Space:**
- Monitor available space before training
- Fail gracefully if < 10GB free (clear error message)
- Auto-cleanup old checkpoints if configured

**File Conflicts:**
- Race condition: same image in working/ and verified/
- Resolution: verified/ takes precedence, remove from working/
- Atomic file moves to prevent partial writes

### X-AnyLabeling Integration

**Model Load Failure:**
- If active/best.pt corrupted, fallback to previous version
- Log error and notify user
- Keep last 3 active models as backup

**Format Mismatches:**
- Validate YOLO format before accepting into verified/
- Check: bbox coords in [0, 1], class IDs valid
- Reject invalid files with clear error message

### Active Learning Edge Cases

**Empty Priority Queue:**
- All working/ images verified
- Prompt: "All images annotated! Train final model?"
- Option to import more raw images with SAM3

**Model Predicts Nothing:**
- Early iterations, model may fail to detect
- Prioritization falls back to SAM3 disagreement only
- Log warning: "Model not detecting, needs more data"

### Data Split Issues

**Too Few Images for Eval:**
- If verified/ < 100 images, skip eval sampling
- Train on all verified, evaluate only on test
- Start eval sampling once >= 100 verified images

**Class Imbalance:**
- Track class distribution in splits
- Stratified sampling when possible
- Log warning if any class < 10 examples

### Pipeline Health States

```
HEALTHY:   All systems operational
DEGRADED:  Working with fallbacks (e.g., smaller batch size)
WARNING:   Can continue but user attention needed
FAILED:    Cannot proceed, manual intervention required
```

### Logging & Alerts

- All errors: `logs/pipeline_errors.log` (timestamped)
- Critical failures: Desktop notification or email (optional)
- Daily summary: images annotated, models trained, errors

---

## 8. Monitoring & Progress Tracking

### Real-Time Monitoring

**1. Training Dashboard (`notebooks/analysis.ipynb`):**

Jupyter notebook with interactive visualizations:

```python
# Key plots:
├── mAP & F1 Progression Over Iterations
│   ├── Eval mAP50 (blue), Test mAP50 (orange)
│   ├── Eval F1 (blue dashed), Test F1 (orange dashed)
│   └── Gap between eval/test (shaded area)
│
├── Training Data Growth
│   ├── Cumulative verified images
│   ├── Images per iteration
│   └── Annotation velocity (images/day)
│
├── Model Performance Breakdown
│   ├── Per-class mAP and F1
│   ├── Precision-Recall curves
│   └── Confusion matrices (latest model)
│
└── Active Learning Efficiency
    ├── Priority score distribution
    ├── Uncertainty reduction over time
    └── Diminishing returns indicator
```

**2. CLI Progress Monitor:**

```bash
$ python pipeline/monitor.py

┌─────────────────────────────────────────────────────────┐
│ YOLO Iterative Pipeline Status                          │
├─────────────────────────────────────────────────────────┤
│ Active Model: v007 (2026-02-27 15:32)                   │
│ Eval Metrics:                                           │
│   mAP50:      0.872  (+0.018 from v006)                │
│   F1 Score:   0.834  (+0.021 from v006)                │
│   Precision:  0.881  Recall: 0.792                     │
│ Test Metrics:                                           │
│   mAP50:      0.824  (+0.012 from v006)                │
│   F1 Score:   0.784  (+0.015 from v006)                │
├─────────────────────────────────────────────────────────┤
│ Data Progress:                                          │
│   Verified:   687 / 1500 images  [████████░░] 45.8%   │
│   Training:   584 images (85%)                          │
│   Eval:       103 images (15%)                          │
│   Test:       200 images (fixed)                        │
├─────────────────────────────────────────────────────────┤
│ Pipeline Status: HEALTHY                                │
│   File Watcher:   Running ✓                            │
│   Next Training:  23 images until trigger (50 thresh)  │
│   Last Training:  1.2 hours ago (v007)                 │
│   Training Time:  17.8 min average                      │
├─────────────────────────────────────────────────────────┤
│ Annotation Velocity:                                    │
│   Last hour:     12 images                              │
│   Last 24h:      147 images                             │
│   Estimated:     3.2 days to complete                   │
└─────────────────────────────────────────────────────────┘

Priority Queue Preview:
  1. IMG_0847.jpg (score: 0.923) - High uncertainty
  2. IMG_1204.jpg (score: 0.891) - SAM3 disagreement
  3. IMG_0332.jpg (score: 0.876) - Rare detection density
  ...
```

**3. File Watcher Logs:**

```bash
# logs/watcher.log (tail -f)
[2026-02-27 15:45:23] INFO: Monitoring data/verified/ (current: 687 files)
[2026-02-27 15:47:11] INFO: New file detected: IMG_0523.jpg (688 total)
[2026-02-27 15:52:35] TRIGGER: 50 new images since last training, starting...
[2026-02-27 15:52:42] INFO: Training started (model_v008, 586 train images)
[2026-02-27 16:10:18] INFO: Training completed (17.6 min)
[2026-02-27 16:10:19] METRICS: Eval mAP50=0.879 (+0.007), F1=0.841 (+0.007)
[2026-02-27 16:10:19] METRICS: Test mAP50=0.831 (+0.007), F1=0.791 (+0.007)
[2026-02-27 16:10:20] SUCCESS: Model v008 promoted to active (improved)
[2026-02-27 16:10:25] INFO: Priority queue updated (810 images remaining)
```

### Key Metrics to Watch

**Early Stage (Iterations 1-3):**
- Eval mAP should jump significantly (+0.10-0.15 per iteration)
- F1 should track similarly
- Test metrics lag slightly (expected)
- **If no improvement:** Check data quality, class balance

**Mid Stage (Iterations 4-8):**
- Improvements slow down (+0.02-0.05 per iteration)
- Eval/test gap should stabilize
- **If gap widens:** Possible overfitting, review data diversity

**Late Stage (Iterations 9+):**
- Diminishing returns (+0.00-0.02 per iteration)
- Time to consider stopping or collecting edge cases
- Focus on per-class performance (some classes may lag)

### Stopping Criteria

```python
# Automated stopping suggestion
if last_3_iterations_improvement < 0.01:
    print("⚠️  mAP/F1 plateau detected. Consider:")
    print("   - Review per-class performance for weak classes")
    print("   - Collect more diverse data for edge cases")
    print("   - Current model may be production-ready")
    print(f"   - Current eval mAP: {mAP:.3f}, F1: {f1:.3f}")
```

**User-Defined Stopping:**
- Eval F1 > 0.85 and plateau for 2 iterations
- Test mAP meets production requirement
- Time/budget exhausted

---

## 9. Setup & Configuration

### System Requirements

- **OS:** Linux/Ubuntu (primary support)
- **Python:** 3.8+
- **GPU:** 2x NVIDIA A5500 with CUDA 11.8+
- **Disk:** ~50GB free space
- **RAM:** 16GB+ recommended

### Dependencies

```txt
# requirements.txt
ultralytics>=8.3.0        # YOLO26 support
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
pyyaml>=6.0
watchdog>=3.0.0           # File system monitoring
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

### Initial Setup Steps

**1. Environment Setup:**

```bash
cd yolo-iterative-pipeline
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create directory structure
python setup.py init
```

**2. Data Preparation:**

```bash
# Place your data
cp -r /path/to/raw/images data/raw/
cp -r /path/to/sam3/labels data/sam3_annotations/
cp -r /path/to/test/images data/test/images/
cp -r /path/to/test/labels data/test/labels/

# Initialize working directory
python setup.py prepare-working
# This copies SAM3 annotations to data/working/
```

**3. Configuration:**

```yaml
# configs/pipeline_config.yaml
project_name: "my-detection-project"
classes: ["class1", "class2", "class3"]

# Trigger settings
trigger_threshold: 50        # Train after N new verified images
early_trigger: 25           # Lower threshold for first 3 iterations
min_train_images: 50        # Minimum before first training

# Data splits
eval_split_ratio: 0.15      # 15% of verified goes to eval
stratify: true              # Balance classes in splits

# Active learning weights
uncertainty_weight: 0.40
disagreement_weight: 0.35
diversity_weight: 0.25

# Notifications
desktop_notify: true
slack_webhook: null         # Optional

# Cleanup
keep_last_n_checkpoints: 10
```

**4. Start Pipeline:**

```bash
# Terminal 1: File watcher (runs continuously)
python pipeline/watcher.py --config configs/pipeline_config.yaml

# Terminal 2: Monitor dashboard
python pipeline/monitor.py

# Terminal 3: Annotate
x-anylabeling
# In X-AnyLabeling:
#   - Open Project: data/working/
#   - Load Model: models/active/best.pt
#   - Start annotating!
```

**5. Bootstrap Training:**

```bash
# Option A: Train on SAM3 annotations (noisy baseline)
python pipeline/train.py --bootstrap

# Option B: Clean 50-100 images first, then start watcher
# (Recommended if SAM3 is very noisy)
```

### Configuration Files

```
configs/
├── pipeline_config.yaml    # Pipeline behavior, thresholds
├── yolo_config.yaml        # YOLO26 training hyperparameters
└── classes.yaml            # Class names and IDs
```

### X-AnyLabeling Configuration

```yaml
# data/working/.anylabeling/config.yaml
auto_save: true
save_format: "yolo"
model_path: "../../models/active/best.pt"
confidence_threshold: 0.25
show_scores: true
```

### Maintenance Tasks

```bash
# Check pipeline health
python pipeline/monitor.py --health-check

# Re-score priority queue manually
python pipeline/active_learning.py --rescore

# Export production model
python pipeline/export.py --version v007 --formats onnx tensorrt

# Backup checkpoints
python pipeline/backup.py --destination /backup/path/

# Clean old checkpoints (keeps last 10)
python pipeline/cleanup.py --keep 10
```

### Quick Start Checklist

- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Create directory structure (`python setup.py init`)
- [ ] Copy raw images, SAM3 annotations, test set
- [ ] Configure `pipeline_config.yaml` (classes, thresholds)
- [ ] Run bootstrap training OR clean 50+ images manually
- [ ] Start file watcher (`python pipeline/watcher.py`)
- [ ] Open X-AnyLabeling and start cleaning
- [ ] Monitor progress (`python pipeline/monitor.py`)

---

## 10. Expected Outcomes & Success Metrics

### Iteration Performance Expectations

**Iteration 0 (Bootstrap on SAM3):**
- Eval mAP: 0.55-0.70 (noisy baseline)
- Test mAP: 0.50-0.65
- Purpose: Establish baseline, enable active learning

**Iteration 1-2 (100-200 verified images):**
- Eval mAP: 0.75-0.82 (large jumps)
- Test mAP: 0.70-0.78
- Model becomes useful for assisted annotation

**Iteration 3-5 (300-500 verified images):**
- Eval mAP: 0.82-0.88 (steady improvement)
- Test mAP: 0.77-0.84
- Model ready for production with human oversight

**Iteration 6+ (700+ verified images):**
- Eval mAP: 0.88-0.93 (diminishing returns)
- Test mAP: 0.84-0.90
- Production-quality, autonomous deployment possible

### Value Proposition

**Traditional Annotation Workflow:**
1. Annotate 1500 images manually (200+ hours)
2. Train model once at end
3. No assistance during annotation
4. Model quality unknown until finished

**Iterative Pipeline Workflow:**
1. Annotate 1500 images with increasing assistance (100-150 hours)
2. Model improves continuously (8-12 training runs)
3. Early iterations provide annotation help
4. Production model ready at iteration 3-5 (300-500 images)
5. Remaining iterations refine and polish

**Time Savings:**
- 30-50% reduction in annotation time
- Production model available 60-70% earlier
- Higher quality annotations (model helps catch mistakes)

### Deployment Timeline

```
Week 1: Setup + Bootstrap
├── Day 1-2: Environment setup, data preparation
├── Day 3: Bootstrap training, validate pipeline
└── Day 4-7: Annotate first 100 images, iteration 1-2

Week 2-3: Rapid Iteration
├── Annotate 50-100 images/day with improving model
├── Train every 1-2 days (50-100 verified threshold)
├── By end: 500-700 verified images, production-ready model

Week 4+: Refinement (Optional)
├── Continue annotating edge cases
├── Focus on weak classes or scenarios
└── Polish model to target performance
```

### Success Criteria

**Technical:**
- Eval mAP50 > 0.85, F1 > 0.80
- Test mAP50 > 0.82, F1 > 0.77
- Per-class recall > 0.75 for all classes
- Inference speed: < 50ms per image on A5500

**Operational:**
- Annotation velocity: 50+ images/day sustained
- Pipeline uptime: > 95% (minimal manual intervention)
- Model promotion rate: 60-80% of training runs improve
- User satisfaction: Model assistance reduces manual effort

---

## 11. Future Enhancements

### Phase 2 Improvements

**1. Multi-Model Ensemble:**
- Train multiple YOLO variants in parallel
- Ensemble predictions for higher accuracy
- Use disagreement between models for active learning

**2. Semi-Supervised Learning:**
- Use high-confidence predictions on unlabeled data
- Pseudo-labeling for data augmentation
- Self-training loop for continuous improvement

**3. Integration with SAM3 Refinement:**
- Use YOLO detections as prompts for SAM3
- Get refined segmentation masks from boxes
- Support both detection and segmentation workflows

**4. Advanced Active Learning:**
- Uncertainty estimation with dropout or ensembles
- Bayesian optimization for sample selection
- Hard negative mining for challenging examples

**5. Cloud Training Support:**
- Optional offload training to cloud GPUs
- Keep annotation local, sync data/models
- Cost-effective for teams without local GPUs

### Long-Term Vision

- Multi-user annotation support with conflict resolution
- Integration with annotation marketplaces (Label Studio teams, MTurk)
- Automatic hyperparameter tuning (learning rate, augmentation)
- Model compression and quantization for edge deployment
- Real-time feedback: model predicts as you annotate

---

## Appendix A: File Formats

### YOLO Format Annotations

```
# data/verified/IMG_0001.txt
# Format: class_id center_x center_y width height
0 0.512 0.345 0.123 0.089
1 0.723 0.612 0.098 0.145
0 0.234 0.789 0.156 0.112
```

- All coordinates normalized to [0, 1]
- `class_id`: Zero-indexed integer
- `center_x, center_y`: Box center coordinates
- `width, height`: Box dimensions

### Training History JSON

```json
{
  "version": "v003",
  "timestamp": "2026-02-27T09:10:22Z",
  "model_type": "yolo26n",
  "training_data": {
    "total_verified": 387,
    "train_images": 329,
    "eval_images": 58
  },
  "metrics": {
    "eval": {
      "mAP50": 0.847,
      "precision": 0.881,
      "recall": 0.792,
      "f1": 0.834
    }
  }
}
```

### Priority Queue Format

```
# logs/priority_queue.txt
# Generated: 2026-02-27T09:10:25Z
# Model: v003
# Format: filename | priority | uncertainty | disagreement | diversity
IMG_0453.jpg | 0.847 | 0.92 | 0.81 | 0.73
IMG_1203.jpg | 0.801 | 0.88 | 0.75 | 0.79
```

---

## Appendix B: Common Issues & Solutions

### Issue: Training OOM (Out of Memory)

**Symptoms:** CUDA out of memory error during training

**Solutions:**
1. Reduce batch_size: 16 → 8 → 4
2. Reduce imgsz: 1280 → 1024 → 640
3. Use single GPU: device: [0] instead of [0, 1]
4. Disable copy_paste augmentation

### Issue: Model Not Improving

**Symptoms:** Eval mAP plateaus early (< 0.75)

**Solutions:**
1. Check class balance in training data
2. Validate annotations (corrupted files?)
3. Increase training epochs: 50 → 100
4. Review per-class performance (one class dragging down overall?)
5. Collect more diverse data

### Issue: X-AnyLabeling Not Loading Model

**Symptoms:** Error when loading custom model

**Solutions:**
1. Check symlink: `ls -la models/active/best.pt`
2. Verify model format: Should be .pt file, not .engine or .onnx
3. Check X-AnyLabeling version compatibility with YOLO26
4. Fallback to previous checkpoint manually

### Issue: Priority Queue Empty But Images Remain

**Symptoms:** priority_queue.txt is empty but working/ has files

**Solutions:**
1. Re-run scoring: `python pipeline/active_learning.py --rescore`
2. Check if files are duplicates already in verified/
3. Validate YOLO format of remaining files

### Issue: Eval/Test Gap Growing

**Symptoms:** Eval mAP high but test mAP low/stagnant

**Solutions:**
1. Indicates overfitting to current annotation style
2. Review if test set represents production data
3. Add more diverse examples to training set
4. Reduce augmentation intensity
5. Increase eval set size (15% → 20%)

---

## References

- [Ultralytics YOLO26 Documentation](https://docs.ultralytics.com/models/yolo26/)
- [YOLO26 Blog Post](https://www.ultralytics.com/blog/meet-ultralytics-yolo26-a-better-faster-smaller-yolo-model)
- [X-AnyLabeling GitHub](https://github.com/CVHub520/X-AnyLabeling)
- [Active Learning for Object Detection Survey](https://arxiv.org/abs/2004.04699)
- [Small Object Detection Best Practices](https://blog.roboflow.com/small-object-detection/)

---

**Document Version:** 1.0
**Last Updated:** 2026-02-26
**Status:** Ready for Implementation
