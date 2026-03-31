# Quick Start Guide - Option A: Automatic Workflow

## ✅ Setup Complete!

Your YOLO Iterative Pipeline is ready with:
- **1,568 images** in `data/working/` ready for manual verification
- **1,568 pre-labeled annotations** converted from detections.json (YOLO format)
- **7 classes**: boat-rgb, human head, human torso, outboard motor-rgb, boat, human, outboard motor
- **Verification tracking** initialized (check status with: `python scripts/track_verification.py`)

**Important:** These are pre-labeled annotations from SAM3 that require manual verification:
- Boxes may be too tight/loose
- May have false positives or missing objects
- Class labels may be incorrect
- You need to review and correct each image

## 🚀 Launch the Pipeline

### Method 1: Single Command (Recommended - NEW in v0.2.0)

```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
yolo-pipeline-run
```

**What it does:**
- Runs health check (validates structure, configs, annotations)
- Launches 2 background services: auto-move watcher, training watcher
- Logs only when files are moved or training triggers (no spam!)
- Graceful shutdown on Ctrl+C

**Then open X-AnyLabeling in a separate terminal:**
```bash
xanylabeling  # Open Dir: data/working/
```

**Optional: Monitor status in a third terminal:**
```bash
watch -n 5 yolo-pipeline-monitor
```

That's it! Just 2-3 terminals instead of 4.

**Optional flags:**
```bash
yolo-pipeline-run --no-doctor       # Skip health check
yolo-pipeline-run --no-auto-move    # Manual file movement
yolo-pipeline-run --debug           # Verbose logging
```

### Method 2: View 4-Terminal Instructions

```bash
./scripts/start_pipeline.sh
```

This displays the commands for the traditional 4-terminal workflow.

### Method 3: Manual Launch (4 Terminals)

Copy and paste these commands into 4 separate terminals:

#### **Terminal 1: Auto-Move Watcher**
```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
python scripts/auto_move_verified.py
```

**What it does:**
- Monitors `data/working/images/` for X-AnyLabeling JSON files
- **Checks verified flag** in JSON (`flags.verified == true`)
- Waits 60s after JSON modification for stability
- Converts JSON annotations → YOLO format
- **Copies** image to `data/verified/images/` (original stays in working/)
- Creates YOLO label in `data/verified/labels/`
- Leaves JSON and image in `working/images/` (for continued annotation)

**How it ensures manual verification:**
1. You review image in X-AnyLabeling
2. Correct boxes, delete false positives, add missing objects
3. Press **Space** to mark as verified (sets `flags.verified: true` in JSON)
4. Save (Ctrl+S) - updates JSON file
5. 60 seconds later, script auto-moves image + creates YOLO label
6. Only verified images are moved to training pipeline

#### **Terminal 2: Training Watcher**
```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
yolo-pipeline-watch
```

#### **Terminal 3: Status Monitor**
```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
watch -n 5 yolo-pipeline-monitor
```

#### **Terminal 4: X-AnyLabeling**
```bash
xanylabeling
```

## 📝 X-AnyLabeling Setup

**Installation:** See the official get started guide: https://github.com/CVHub520/X-AnyLabeling/blob/main/docs/en/get_started.md

### Step 0: Enable the Verified Flag (First-Time Setup)

The "verified" checkmark is not enabled by default — it must be added to your X-AnyLabeling config.

Edit `~/.xanylabelingrc` (create it if it doesn't exist) and add:

```yaml
flags:
- verified
```

Or run this one-liner:

```bash
# Append if file exists, or create it
python3 -c "
import os, yaml
rc = os.path.expanduser('~/.xanylabelingrc')
cfg = yaml.safe_load(open(rc)) if os.path.exists(rc) else {}
cfg['flags'] = ['verified']
yaml.dump(cfg, open(rc, 'w'), default_flow_style=False)
print('Done — verified flag added to', rc)
"
```

Restart X-AnyLabeling after making this change. You should see a **flag panel** on the side with a "verified" checkbox (toggled with `Space`).

---

### Step 1: Open Directory

- Click `File` → `Open Dir`
- Navigate to: `data/working/`
- Click `Select Folder`

### Step 2: Load YOLO Annotations

After opening the directory, the images will load but annotations won't appear yet. You need to import them:

- Hover over **`Upload`** in the menu bar
- Click **`Upload YOLO HBB Annotations`**
- **First**, select the classes file: `data/working/classes.txt`
- **Then**, select the labels directory: `data/working/labels/`
- Annotations will now appear on the images

### Step 3: Load Model (optional, for AI-assisted annotation)

- Click `AI` → `Load Model`
- Navigate to: `models/active/best.onnx`
- (Skip this if no model exists yet — bootstrap training will create one)

### Step 4: Start Annotating

- Review the existing annotations (from detections.json)
- Correct any mistakes:
  - Too tight/loose boxes → drag corners to adjust
  - Wrong class → click box, select new class
  - Missing objects → press `R` and draw new boxes
  - False positives → select box, press `Delete`
- **Press `Space` to mark as verified** (checks the "verified" flag in the side panel)
- Press `Ctrl+S` to save after each image
- Move to next image with `D`

## 🔄 Workflow

**Key Concept:** Verified flag + automatic movement

- **Manual verification** = You review images and press Space to mark as verified
- **Automatic movement** = Script detects verified flag in JSON and moves files
- **Flag-based protection** = Only images with `verified=true` flag are moved

```
┌──────────────────────────────────────────────────────────┐
│ 1. MANUAL VERIFICATION in X-AnyLabeling (data/working/) │
│    - Review pre-labeled boxes from SAM3                 │
│    - Adjust tight/loose boxes (drag corners)            │
│    - Delete false positives (Delete key)                │
│    - Add missing objects (W key + draw)                 │
│    - Fix wrong class labels                             │
│    - **Press Space to mark as verified** ✓              │
│    - Save (Ctrl+S)                                      │
│    - Move to next image (D key)                         │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ 2. AUTOMATIC COPYING (Terminal 1)                       │
│    - Monitors data/working/images/*.json every 60s      │
│    - Checks for flags.verified == true in JSON          │
│    - Waits 60s after JSON modification for stability    │
│    - Converts JSON annotations → YOLO format            │
│    - COPIES image → data/verified/images/               │
│    - Creates YOLO label → data/verified/labels/         │
│    - Keeps JSON + image in working/images/ (continue!)  │
│    - Logs in verification tracker as "verified"         │
│                                                          │
│    ✅ Verified flag ensures ONLY manually reviewed      │
│       files are copied for training!                    │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ 3. Training watcher counts (Terminal 2)                 │
│    - Counts files in data/verified/                     │
│    - Triggers training after 50 images                  │
│    - First 3 iterations: 25 images (early trigger)      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ 4. Training runs automatically                           │
│    - Samples 15% eval set                               │
│    - Trains YOLO11n model (~15-20 min)                  │
│    - Evaluates on eval + test sets                      │
│    - Promotes if eval mAP improves                      │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ 5. New model available at models/active/best.pt         │
│    - Reload in X-AnyLabeling (AI → Load Model)          │
│    - Model gives better predictions now!                │
│    - Continue verifying with AI assistance              │
└──────────────────────────────────────────────────────────┘
                         ↓
                    REPEAT! 🔁
```

## 📊 Monitoring Progress

**Terminal 3** shows real-time status:
- Current model version and metrics
- Verified image count
- Training status (HEALTHY / TRAINING)
- Priority queue preview

**Check verification status:**
```bash
# View verification progress
python scripts/track_verification.py

# Output:
# ✓ Verified:   20
# ⚠ Unverified: 1,548
#   Total:      1,568
#   Progress:   1.3%
```

**Manual checks:**
```bash
yolo-pipeline-monitor                # Current status
yolo-pipeline-monitor --history      # View all training runs
python scripts/track_verification.py --list-unverified  # Show unverified images
```

## ⌨️ Essential Keyboard Shortcuts (X-AnyLabeling)

| Key | Action |
|-----|--------|
| `Space` | **Mark as verified** (required for auto-move) ✓ |
| `Ctrl+S` | Save current annotations |
| `D` | Next image |
| `A` | Previous image |
| `W` | Draw new box |
| `Delete` | Remove selected box |
| `R` | Run AI prediction (if model loaded) |
| `Ctrl+Z` | Undo |
| `1-3` | Quick-select class (1=boat, 2=human, 3=outboard motor) |

## 🎯 Expected Timeline

With 1,568 images and automatic workflow:

| Milestone | Images Verified | Cumulative Time | Model Performance | Action |
|-----------|----------------|-----------------|-------------------|--------|
| **Bootstrap** | 50-100 | ~1-2 hours | mAP50: 0.60-0.70 | Basic model, useful for screening |
| **Iteration 1** | 100-200 | ~3-4 hours | mAP50: 0.75-0.82 | Good assistance, speeds up verification |
| **Iteration 2-3** | 300-500 | ~8-12 hours | mAP50: 0.82-0.88 | Production-ready with oversight |
| **Iteration 4+** | 700-1000 | ~20-30 hours | mAP50: 0.88-0.93 | High accuracy, minimal corrections |
| **Complete** | 1,568 | ~40-50 hours | mAP50: 0.90+ | Full dataset verified |

**Time Savings:** ~30-50% faster than manual annotation from scratch

- Pre-labeled boxes from SAM3 (correct ~60-70% of time)
- Model improves iteratively (better predictions each round)
- Only need to correct, not draw from scratch

## 🔁 Manual Model Promotion

The pipeline auto-promotes a model only if its eval mAP50 improves over the previous version. If you want to force-promote a specific version (e.g. it trained on more images but scored slightly lower), do this in a separate terminal:

**1. Check available versions and their metrics:**
```bash
yolo-pipeline-monitor --history
```

**2. Find the checkpoint directory for the version you want:**
```bash
ls models/checkpoints/ | grep v025   # replace v025 with your target version
```

**3. Update the active model symlink:**
```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate

ln -sf $(pwd)/models/checkpoints/model_v006_20260316_083417/weights/best.pt models/active/best.pt
```

**4. Re-export ONNX for X-AnyLabeling:**
```bash
python3 -c "
from pipeline.train import export_to_onnx
from pathlib import Path
export_to_onnx(
    Path('models/checkpoints/model_v006_20260316_083417/weights/best.pt'),
    Path('models/active/best.onnx'),
    imgsz=1280
)
"
```

**5. Reload in X-AnyLabeling:** AI → Load Model → select `models/active/best.onnx`

> **Note:** The running watcher will overwrite the symlink if a future model auto-promotes (i.e. improves on eval mAP). Your manually promoted model stays active until then.

## 🛑 Stopping the Pipeline

**Method 1 (Single Command):**
- Press `Ctrl+C` in the `yolo-pipeline-run` terminal
- Process manager gracefully shuts down all services
- Close X-AnyLabeling: `File` → `Exit`

Quick check/remove command:
Check if lock exists, run 
`ls logs/.training.lock 2>/dev/null && echo "Lock exists" || echo "No lock"`

Remove if stale (no training running) run `rm logs/.training.lock`

  When to Delete Cache Files:
  - After changing class count
  - After modifying class names
  - After adding/removing label files
  - If you see class count errors

  Quick command:
  ### Remove all YOLO caches
  
  find data -name "*.cache" -type f -delete

**Method 2 (4-Terminal Workflow):**
1. Terminal 1 (auto-move): Press `Ctrl+C`
2. Terminal 2 (training watcher): Press `Ctrl+C`
3. Terminal 3 (monitor): Press `Ctrl+C`
4. Terminal 4 (X-AnyLabeling): `File` → `Exit`

All progress is automatically saved. You can resume anytime by restarting.

## 🩺 Health Check (NEW in v0.2.0)

Before starting the pipeline, run the doctor command to check everything is configured correctly:

```bash
yolo-pipeline-doctor
```

**Checks:**
- ✅ Directory structure (14 required directories)
- ✅ Configuration files (YAML syntax and values)
- ✅ Annotations (YOLO format compliance, class IDs)
- ✅ Active model (loads and validates)

**Exit codes:**
- `0` = Healthy, ready to run
- `1` = Errors found, must fix before running

**Example output:**
```
🩺 YOLO Pipeline Health Check

✅ PASS: Structure Validation
  All 14 required directories exist with YOLO layout

✅ PASS: Config Validation
  All configuration files valid

⚠️  WARNING: Annotation Validation
  Found 3 orphaned label files in data/working/

✅ PASS: Model Validation
  Active model loads successfully

Overall Status: WARNINGS (can proceed with caution)
```

## 🔧 Troubleshooting

### Auto-move not working
```bash
# Check logs
tail -f logs/auto_move.log

# Common issues:
# 1. Did you press Space to mark as verified?
# 2. Did you save (Ctrl+S) after marking verified?
# 3. Wait 60 seconds after save for stability threshold

# Verify JSON has verified flag:
grep -l '"verified": true' data/working/images/*.json | head -5

# Check classes.txt exists in verified/:
ls -la data/verified/classes.txt
```

### Training not triggering
```bash
# Check verified count
ls data/verified/*.txt | wc -l

# Should see: "Moved X files. Total verified: Y" in Terminal 1
# Training triggers at 50 files (or 25 for first 3 iterations)
```

### X-AnyLabeling can't find images
```bash
# Ensure you opened the correct directory
# Should be: /home/lenovo6/TiongKai/yolo-iterative-pipeline/data/working/
```

### Training failed
```bash
# Check training logs
tail -f logs/watcher.log

# Common issues:
# - Out of memory: Reduce batch_size in configs/yolo_config.yaml
# - No test set: Create data/test/images/ and data/test/labels/
```

## 📚 Additional Resources

- **Full docs**: `README.md`
- **Implementation summary**: `docs/IMPLEMENTATION_SUMMARY.md` (comprehensive technical details)
- **Analysis notebook**: `notebooks/analysis.ipynb`
- **Config files**: `configs/pipeline_config.yaml`, `configs/yolo_config.yaml`
- **Training history**: `logs/training_history.json`
- **Verification status**: `logs/verification_status.json`
- **Priority queue**: `logs/priority_queue.txt`

## 🎉 You're Ready!

Open 4 terminals, run the commands, and start manual verification. The pipeline handles movement and training automatically!

**Remember:**
- **Manual verification** = You review/correct in X-AnyLabeling
- **Automatic movement** = Script moves files after 60s stability
- Track progress: `python scripts/track_verification.py`

Questions? Check these resources:
- `README.md` - Full feature documentation
- `docs/IMPLEMENTATION_SUMMARY.md` - Comprehensive technical details
- Inline help:
  ```bash
  yolo-pipeline-run --help           # Single-command workflow
  yolo-pipeline-doctor --help        # Health check
  yolo-pipeline-train --help         # Manual training
  yolo-pipeline-watch --help         # File watcher
  yolo-pipeline-monitor --help       # Status monitor
  python scripts/track_verification.py --help  # Verification tracking
  ```

## 🔄 Starting a New Project

If you want to reset the pipeline for a completely different project (new classes/dataset):

### Quick Reset with Script

```bash
# Reset everything (backs up automatically with timestamp)
./scripts/reset_for_new_project.sh
```

**What it does:**
- ✅ Backs up current data, models, and logs to `backups/TIMESTAMP/`
- 🗑️ Clears verified data, active models, checkpoints
- 🗑️ Clears training history and cache files
- 🗑️ Resets verification tracking
- 📋 Shows next-steps instructions

**After reset:**

1. **Update classes for new project:**
   ```bash
   cat > data/verified/classes.txt << 'EOF'
   person
   car
   dog
   EOF
   
   cp data/verified/classes.txt data/working/classes.txt
   ```

2. **Add new images:**
   ```bash
   cp /path/to/new/images/* data/working/images/
   ```

3. **Update config (optional):**
   ```bash
   nano configs/pipeline_config.yaml
   # Update classes list (for reference)
   # Adjust trigger_threshold if needed
   ```

4. **Start training from scratch:**
   ```bash
   source venv/bin/activate
   python -m pipeline.train --from-scratch
   ```

5. **Or run the full pipeline:**
   ```bash
   yolo-pipeline-run
   ```

### Manual Reset (More Control)

```bash
# 1. Backup old data
BACKUP="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP
cp -r data/verified $BACKUP/
cp -r models/checkpoints $BACKUP/
cp logs/training_history.json $BACKUP/

# 2. Clear everything
rm -rf data/verified/images/* data/verified/labels/*
rm -f models/active/best.pt models/active/best.onnx
rm -rf models/checkpoints/*
rm -f logs/training_history.json logs/priority_queue.txt
find data -name "*.cache" -delete

# 3. Update classes
nano data/verified/classes.txt
cp data/verified/classes.txt data/working/classes.txt

# 4. Train from scratch
source venv/bin/activate
python -m pipeline.train --from-scratch
```

### Training Flags

| Flag | When to Use | Effect |
|------|-------------|--------|
| **No flag** | Normal iteration | Resumes from active model (smart resume) |
| **`--from-scratch`** | New project or reset | Starts fresh from pretrained YOLO11n |

**Example:**
```bash
# Continue from existing model (default)
python -m pipeline.train

# Start fresh (new project)
python -m pipeline.train --from-scratch
```

