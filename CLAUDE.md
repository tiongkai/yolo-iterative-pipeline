# YOLO Iterative Training Pipeline - Project Memory

## Project Overview

Active learning pipeline for iterative YOLO training that turns annotation into continuous model improvement. Dual-purpose system: assists annotation while producing production-ready detection models.

## Key Context

**Problem:** User has medium dataset (500-2000 images) with SAM3 bounding box annotations that are inaccurate:
- Boxes too tight/loose
- Missing objects
- False positives

**Goal:** Reduce manual annotation effort while training production YOLO model for detection

**Hardware:** 2x NVIDIA A5500 GPUs (excellent for fast iteration)

## Architecture Decisions

### Data Flow
- **raw/**: Original images
- **sam3_annotations/**: Initial SAM3 boxes (YOLO format, noisy)
- **working/**: X-AnyLabeling workspace for cleaning
- **verified/**: Human-verified annotations (triggers training)
- **eval/**: Auto-sampled 10-15% from verified (in-distribution metrics)
- **test/**: Pre-existing labeled data (fixed, generalization metrics)

### Key Design Choice: Dual Metrics
- **Eval set**: Sampled from verified annotations, measures performance on current working dataset
- **Test set**: Pre-existing labels user already has, measures generalization
- Why: Eval shows iteration improvement, test shows production readiness

### Active Learning Strategy
Priority score = 0.40×uncertainty + 0.35×disagreement + 0.25×diversity
- **Uncertainty**: Low model confidence → needs more examples
- **Disagreement**: Model vs SAM3 mismatch → model learned something new
- **Diversity**: Under-represented detection counts → ensure coverage

### Model Selection: YOLO11n (Current Implementation)
- Using YOLO11n (yolo26n not yet available in Ultralytics)
- Config: yolo11n.pt, imgsz=1280, batch_size=16, ~15-20min training on dual A5500
- Will upgrade to yolo26n when available (just change model name in config)

### Small Object Optimizations
- imgsz: 1280 (vs standard 640) → 4x more pixels
- copy_paste: 0.5 → duplicate small objects in training
- close_mosaic: 10 → disable mosaic in final epochs (better small object precision)

### Automation Strategy
- File watcher monitors `data/verified/` folder (NOT working/)
- Triggers training after N new images (default: 50, early iterations: 25)
- Auto-samples eval set (stratified by class, 15% of verified)
- Trains YOLO11n, evaluates on eval + test
- Promotes model to active/ only if eval mAP improves
- Re-scores priority queue with new model
- X-AnyLabeling loads improved model from models/active/best.pt

## Tool Integration: X-AnyLabeling

**Why X-AnyLabeling:**
- Native YOLO model loading
- Auto-annotation with custom models
- Fast keyboard shortcuts (W=draw, D=next, Delete=remove)
- Saves in YOLO format
- Shows confidence scores

**Workflow (Option A: Automatic - Recommended):**
1. User annotates in X-AnyLabeling → saves to data/working/
2. Auto-move script (`scripts/auto_move_verified.py`) watches working/
   - Waits 60s for file stability
   - Validates YOLO format
   - Moves to data/verified/
3. Training watcher (`yolo-pipeline-watch`) monitors verified/
4. After 50 verified images → training auto-triggers
5. New model promoted to models/active/best.pt (if improved)
6. User reloads model in X-AnyLabeling (AI → Load Model)
7. Repeat with better predictions

**Alternative Workflows:**
- **Option B (Manual)**: Use `./scripts/move_verified.sh` to manually move batches
- **Option C (Direct)**: Save directly to data/verified/ (skip working/ step)

**CRITICAL:** `yolo-pipeline-watch` monitors `verified/` not `working/`. Movement from working/ to verified/ requires either auto-move script or manual movement.

## Metrics Tracked

**Per Model:**
- mAP@0.5, mAP@0.5:0.95
- Precision, Recall, **F1 Score** (user requested)
- Both eval and test sets
- Per-class breakdown

**Training History:**
- Version, timestamp, image counts
- Metrics progression
- Improvement over previous
- Training time

## Expected Performance

**Iteration 1-2** (100-200 verified):
- Eval mAP: 0.75-0.82
- Large jumps, model becomes useful

**Iteration 3-5** (300-500 verified):
- Eval mAP: 0.82-0.88
- Production-ready with oversight

**Iteration 6+** (700+ verified):
- Eval mAP: 0.88-0.93
- Diminishing returns, polish phase

## Production Deployment

Models can be deployed during annotation (not just at end):
- After iteration 2-3: Good enough for screening, low-stakes detection
- After iteration 4-5: Production with human oversight
- After iteration 7+: Autonomous deployment

**Export formats:** ONNX, TensorRT (A5500 optimized), TorchScript

## Critical Implementation Notes

1. **Bootstrap training**: Train on SAM3 annotations first (noisy baseline) OR clean 50-100 images manually first
2. **Eval sampling**: Only start when verified/ ≥ 100 images, otherwise train on all
3. **Model promotion**: Only update active/ if eval mAP improves (prevent regression)
4. **Priority re-scoring**: After each training (new model = new uncertainties)
5. **Atomic file moves**: Prevent race conditions in file watching
6. **Lock files**: Prevent simultaneous training runs

## Configuration Tuning

**Trigger threshold:**
- Early iterations: 25 images (faster feedback)
- Mid iterations: 50 images (balanced)
- Late iterations: 100 images (diminishing returns)

**Eval split ratio:** 15% (adjust to 10% if dataset < 500, 20% if > 1500)

**Active learning weights:** Can adjust if specific signal more important:
- Increase uncertainty if model very uncertain
- Increase disagreement if SAM3 systematically wrong
- Increase diversity if dataset very imbalanced

## Error Handling

**OOM:** Auto-retry with batch_size/2, fallback to imgsz/2
**Corrupted annotations:** Skip, log, continue with valid subset
**Training crash:** Resume from checkpoint
**Model promotion fails:** Keep previous active model, log event
**Empty priority queue:** Prompt to import more or train final model

## Monitoring Commands

```bash
# Real-time status
python pipeline/monitor.py

# Health check
python pipeline/monitor.py --health-check

# Re-score manually
python pipeline/active_learning.py --rescore

# Export for production
python pipeline/export.py --version v007 --formats onnx tensorrt
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

## Implementation Status

✅ **COMPLETE** - All 8 tasks implemented and tested (March 2, 2026)

### What Was Built

**Core Pipeline (Tasks 1-5):**
- Project structure with setup.py and pytest configuration
- Configuration management (PipelineConfig, YOLOConfig with YAML serialization)
- Data utilities (validation, eval set sampling with stratification)
- Active learning scoring (uncertainty, disagreement, diversity)
- Training pipeline (YOLO11n training, dual eval/test metrics, F1 scoring, model promotion)

**Automation (Tasks 6-7):**
- File watcher service (`yolo-pipeline-watch`) - monitors verified/, triggers training
- Auto-move script (`scripts/auto_move_verified.py`) - handles working/ → verified/ movement
- Manual move script (`scripts/move_verified.sh`) - batch movement utility
- Status monitor (`yolo-pipeline-monitor`) - Rich-based real-time dashboard
- Model export (`yolo-pipeline-export`) - ONNX, TensorRT, TorchScript

**Documentation (Task 8):**
- Comprehensive README.md with 3 workflow options
- QUICKSTART.md for Option 2 (automatic workflow)
- Jupyter notebook for metrics visualization (`notebooks/analysis.ipynb`)
- Helper script (`scripts/start_pipeline.sh`) with all commands

**Test Coverage:** 41/41 tests passing (100%)
- 27 unit tests (config, data_utils, metrics, active_learning)
- 5 integration tests (watcher)
- ~44% code coverage (focused on critical paths)

### Current Dataset: Willow Boat Detection

**Status:** Ready for annotation
- **1,568 images** with converted YOLO labels in `data/working/`
- **3 classes:** boat (0), human (1), outboard motor (2)
- **Source:** Converted from detections.json (SAM3 format → YOLO format)
- **Next step:** Review/correct annotations in X-AnyLabeling

### Quick Start (4-Terminal Workflow)

```bash
# Terminal 1: Auto-move watcher
cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
python scripts/auto_move_verified.py

# Terminal 2: Training watcher
yolo-pipeline-watch

# Terminal 3: Status monitor
watch -n 5 yolo-pipeline-monitor

# Terminal 4: Annotation
x-anylabeling  # Open Dir: data/working/
```

See `QUICKSTART.md` for detailed instructions.

## References

- **Implementation Plan**: `docs/plans/2026-02-26-yolo-iterative-pipeline-implementation.md`
- **Design Doc**: `docs/plans/2026-02-26-yolo-iterative-pipeline-design.md`
- **Training History**: `logs/training_history.json` (created after first training)
- **Priority Queue**: `logs/priority_queue.txt` (updated after each training)
