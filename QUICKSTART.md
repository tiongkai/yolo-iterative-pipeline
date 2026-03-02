# Quick Start Guide - Option 2: Automatic Workflow

## ✅ Setup Complete!

Your YOLO Iterative Pipeline is ready with:
- **1,568 images** in `data/working/` ready for annotation
- **1,568 labels** converted from detections.json (YOLO format)
- **3 classes**: boat, human, outboard motor

## 🚀 Launch the 4-Terminal Workflow

### Method 1: View Instructions

```bash
./scripts/start_pipeline.sh
```

This displays the commands for all 4 terminals.

### Method 2: Manual Launch

Copy and paste these commands into 4 separate terminals:

#### **Terminal 1: Auto-Move Watcher**
```bash
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
python scripts/auto_move_verified.py
```

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
x-anylabeling
```

## 📝 X-AnyLabeling Setup

When X-AnyLabeling opens:

1. **Open Directory**
   - Click `File` → `Open Dir`
   - Navigate to: `/home/lenovo6/TiongKai/yolo-iterative-pipeline/data/working/`
   - Click `Select Folder`

2. **Configure Classes** (classes.txt auto-loads, but verify):
   - Click `Edit` → `Label Settings`
   - Ensure these classes are listed:
     - boat
     - human
     - outboard motor
   - Click `OK`

3. **Load Model** (optional, for AI-assisted annotation):
   - Click `AI` → `Load Model`
   - Navigate to: `models/active/best.pt`
   - (Skip this if no model exists yet - bootstrap training will create one)

4. **Start Annotating!**
   - Review the existing annotations (from detections.json)
   - Correct any mistakes:
     - Too tight/loose boxes → drag corners to adjust
     - Wrong class → click box, select new class
     - Missing objects → press `W` and draw new boxes
     - False positives → select box, press `Delete`
   - Press `Ctrl+S` to save after each image

## 🔄 Workflow

```
┌──────────────────────────────────────────────────────────┐
│ 1. Review/correct in X-AnyLabeling (data/working/)      │
│    - Fix boxes, delete false positives, add missing     │
│    - Press Ctrl+S to save                               │
│    - Press D to go to next image                        │
└──────────────────────────────────────────────────────────┘
                         ↓
┌──────────────────────────────────────────────────────────┐
│ 2. Auto-move watches (Terminal 1)                       │
│    - Waits 60 seconds after last edit                   │
│    - Validates YOLO format                              │
│    - Moves to data/verified/                            │
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
│    - Continue annotating with assistance                │
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

**Manual check:**
```bash
yolo-pipeline-monitor
yolo-pipeline-monitor --history  # View all training runs
```

## ⌨️ Essential Keyboard Shortcuts (X-AnyLabeling)

| Key | Action |
|-----|--------|
| `D` | Next image |
| `A` | Previous image |
| `W` | Draw new box |
| `Delete` | Remove selected box |
| `Ctrl+S` | Save current annotations |
| `R` | Run AI prediction (if model loaded) |
| `Ctrl+Z` | Undo |
| `Space` | Flag as verified |
| `1-3` | Quick-select class (1=boat, 2=human, 3=outboard motor) |

## 🎯 Expected Timeline

With 1,568 images and Option 2 workflow:

| Milestone | Images Reviewed | Time | Model Performance | Action |
|-----------|----------------|------|-------------------|---------|
| **Bootstrap** | 50-100 | ~1-2 hours | mAP50: 0.60-0.70 | Basic model, useful for screening |
| **Iteration 1** | 100-200 | ~3-4 hours | mAP50: 0.75-0.82 | Good assistance, speeds up annotation |
| **Iteration 2-3** | 300-500 | ~8-12 hours | mAP50: 0.82-0.88 | Production-ready with oversight |
| **Iteration 4+** | 700-1000 | ~20-30 hours | mAP50: 0.88-0.93 | High accuracy, minimal corrections |

**Note:** With automatic workflow, actual annotation time is ~50-60% of manual annotation!

## 🛑 Stopping the Pipeline

To stop all processes:
1. Terminal 1 (auto-move): Press `Ctrl+C`
2. Terminal 2 (training watcher): Press `Ctrl+C`
3. Terminal 3 (monitor): Press `Ctrl+C`
4. Terminal 4 (X-AnyLabeling): `File` → `Exit`

All progress is automatically saved. You can resume anytime by restarting the terminals.

## 🔧 Troubleshooting

### Auto-move not working
```bash
# Check logs
tail -f logs/auto_move.log

# Verify file stability threshold
# Files must be unchanged for 60 seconds before moving
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
- **Analysis notebook**: `notebooks/analysis.ipynb`
- **Config files**: `configs/pipeline_config.yaml`, `configs/yolo_config.yaml`
- **Training history**: `logs/training_history.json`
- **Priority queue**: `logs/priority_queue.txt`

## 🎉 You're Ready!

Open 4 terminals, run the commands, and start annotating. The pipeline handles everything else automatically!

Questions? Check `README.md` or the inline help:
```bash
yolo-pipeline-train --help
yolo-pipeline-watch --help
yolo-pipeline-monitor --help
```
