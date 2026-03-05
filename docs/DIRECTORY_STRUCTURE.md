# Directory Structure - YOLO Dataset Format

## Overview

The pipeline now uses standard YOLO dataset format with separate `images/` and `labels/` subdirectories. This makes it easier for YOLO training and follows ML best practices.

## New Structure

```
data/
├── working/                    # Annotation workspace
│   ├── images/                 # Images for annotation (1,568 files)
│   ├── labels/                 # Label files YOLO format (1,568 files)
│   └── classes.txt             # Class names (one per line)
│
├── verified/                   # Manually verified annotations
│   ├── images/                 # Verified images
│   └── labels/                 # Verified labels
│
├── eval/                       # Auto-sampled evaluation set
│   ├── images/
│   └── labels/
│
├── test/                       # Fixed test set
│   ├── images/
│   └── labels/
│
└── raw/                        # Original unprocessed images
```

## File Pairing

Each image must have a corresponding label file with the same name:

```
working/images/image001.png  ←→  working/labels/image001.txt
working/images/image002.jpg  ←→  working/labels/image002.txt
```

## Auto-Move Script Behavior

### What It Monitors

```
Monitors: data/working/labels/*.txt
```

When a label file is modified (you save in X-AnyLabeling):
1. Waits 60 seconds for stability
2. Finds corresponding image in `working/images/`
3. Validates YOLO format
4. Moves **both files** to verified:
   - `working/images/X.png` → `verified/images/X.png`
   - `working/labels/X.txt` → `verified/labels/X.txt`

### Directory Creation

The script automatically creates subdirectories in verified/ when moving files:
- `verified/images/` (created on first move)
- `verified/labels/` (created on first move)

## X-AnyLabeling Integration

### Opening the Directory

When you open `data/working/` in X-AnyLabeling:
- X-AnyLabeling **auto-detects** `images/` and `labels/` subdirectories
- Displays images from `images/`
- Loads annotations from `labels/`
- Saves back to `labels/`

### Setup Steps

1. **File → Open Dir**
   - Navigate to: `/path/to/data/working/`
   - Click "Select Folder"

2. **X-AnyLabeling automatically:**
   - Finds `working/images/` for images
   - Finds `working/labels/` for annotations
   - Loads `working/classes.txt` for class names

3. **When you save (Ctrl+S):**
   - Updates label file in `working/labels/image.txt`
   - File modification time updates to NOW
   - Auto-move script detects change after 60s

## Classes.txt Format

**Location:** `data/working/classes.txt`

**Format:** One class name per line, no numbers
```
human
boat
outboard motor
```

**Line number = Class ID:**
- Line 0 (first line) = Class ID 0 (human)
- Line 1 (second line) = Class ID 1 (boat)
- Line 2 (third line) = Class ID 2 (outboard motor)

## YOLO Label Format

**Location:** `data/working/labels/*.txt`

**Format:** One box per line
```
<class_id> <center_x> <center_y> <width> <height>
```

**Example:** `working/labels/image001.txt`
```
0 0.854688 0.244141 0.262500 0.078125
1 0.625781 0.175781 0.114062 0.035156
2 0.450000 0.600000 0.100000 0.080000
```

- All coordinates normalized to 0-1 (relative to image size)
- Center coordinates (not corner coordinates)

## Training Integration

### YOLO Dataset Configuration

The verified/ directory is ready for YOLO training:

```yaml
# dataset.yaml
path: /path/to/data/verified
train: images
val: images

names:
  0: human
  1: boat
  2: outboard motor
```

YOLO automatically:
- Reads images from `verified/images/`
- Reads labels from `verified/labels/`
- Matches files by name (same stem)

### Training Command

```bash
yolo train data=/path/to/data/verified/dataset.yaml model=yolo11n.pt
```

## Migration from Old Structure

**Old structure (flat):**
```
working/
├── image001.png
├── image001.txt
├── image002.png
├── image002.txt
└── classes.txt
```

**New structure (nested):**
```
working/
├── images/
│   ├── image001.png
│   └── image002.png
├── labels/
│   ├── image001.txt
│   └── image002.txt
└── classes.txt
```

**Migration script:**
```bash
# Already completed for your dataset
# 1,568 images moved to working/images/
# 1,568 labels moved to working/labels/
# classes.txt remains in working/
```

## Advantages of New Structure

✅ **YOLO Standard:** Follows official YOLO dataset format
✅ **Cleaner Organization:** Images and labels separated
✅ **Easier Training:** Direct compatibility with YOLO commands
✅ **Better Tooling:** Most ML tools expect this format
✅ **Scalability:** Clearer with thousands of files

## Summary

**Key Changes:**
1. Labels monitored in `working/labels/` (not `working/`)
2. Images stored in `working/images/`
3. Auto-move creates `verified/images/` and `verified/labels/`
4. X-AnyLabeling opens `working/` (auto-detects subdirs)
5. classes.txt stays at `working/classes.txt`

**Workflow:**
1. Open X-AnyLabeling → `data/working/`
2. Review images, save labels
3. Auto-move monitors `working/labels/`, moves both files
4. Training uses `verified/` directory directly

---

**Last Updated:** March 3, 2026
