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

### Model Selection: YOLO26
- Latest Ultralytics model (Jan 2026 release)
- **STAL**: Small-Target-Aware Label Assignment (built-in)
- **ProgLoss**: Progressive Loss Balancing (stable training)
- **MuSGD**: Hybrid SGD+Muon optimizer
- Config: yolo26n, imgsz=1280, batch_size=16, ~15-20min training

### Small Object Optimizations
- imgsz: 1280 (vs standard 640) → 4x more pixels
- copy_paste: 0.5 → duplicate small objects in training
- close_mosaic: 10 → disable mosaic in final epochs (better small object precision)
- STAL built into YOLO26

### Automation Strategy
- File watcher monitors verified/ folder
- Triggers training after N new images (default: 50)
- Auto-samples eval set (stratified by class)
- Trains YOLO26, evaluates on eval + test
- Promotes model to active/ only if eval mAP improves
- Re-scores priority queue with new model
- X-AnyLabeling loads improved model

## Tool Integration: X-AnyLabeling

**Why X-AnyLabeling:**
- Native YOLO model loading
- Auto-annotation with custom models
- Fast keyboard shortcuts
- Saves in YOLO format
- Shows confidence scores

**Workflow:**
1. User saves annotations to working/
2. Background script auto-moves to verified/ (confidence-based: file modified + has labels)
3. After 50 verified images, training auto-triggers
4. New model symlinked to models/active/
5. User reloads model in X-AnyLabeling
6. Repeat

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

## Next Steps

See full design: `docs/plans/2026-02-26-yolo-iterative-pipeline-design.md`

Next phase: Create detailed implementation plan using writing-plans skill
