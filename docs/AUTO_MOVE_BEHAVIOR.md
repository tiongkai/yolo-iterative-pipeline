# Auto-Move Script Behavior - Timestamp-Based Verification

## Problem Solved

**Original issue:** When you ran `python scripts/auto_move_verified.py`, it immediately moved all pre-labeled files to verified/ even though you hadn't manually reviewed them.

**Root cause:** Script was checking "file stable for 60 seconds" (not modified recently), but ALL pre-labeled files were created hours ago, so they were all immediately stable.

## Solution: Timestamp-Based Verification

The script now records its **start timestamp** and only moves files modified **AFTER** that time.

### How It Works

```python
# When script starts
script_start_time = time.time()  # e.g., 2026-03-02 17:30:00

# For each file in data/working/
file_mtime = file.stat().st_mtime

# Only move if modified AFTER script started
if file_mtime < script_start_time:
    # Skip (pre-labeled, not manually reviewed)
    continue
else:
    # Move (manually reviewed in X-AnyLabeling)
    move_to_verified()
```

### Timeline Example

```
14:26:00 - Pre-labeled files created (from detections.json conversion)
17:30:00 - You start auto_move_verified.py (records start time)
17:30:01 - Script checks files:
             - Pre-labeled (14:26:00) → SKIPPED (before 17:30:00)
             - Status: "1568 pre-labeled files waiting for review"

17:35:00 - You open image001 in X-AnyLabeling
17:35:30 - You review, correct boxes, save (Ctrl+S)
17:35:30 - File modification time updates to 17:35:30
17:36:30 - Script checks again (60s later):
             - image001 (17:35:30) → MOVED (after 17:30:00 + 60s stable)
             - Other pre-labeled → STILL SKIPPED
```

## Key Benefits

✅ **Prevents accidental moves:** Pre-labeled files never auto-move without review

✅ **Guarantees manual verification:** Only files you opened/saved in X-AnyLabeling are moved

✅ **Simple logic:** Just compare timestamps (modified before vs after script start)

✅ **No false positives:** Can't accidentally move unreviewed files

## What You See

When you run the script:

```bash
python scripts/auto_move_verified.py
```

Output:
```
Starting auto-move watcher with verification tracking
  Working dir: data/working
  Verified dir: data/verified
  Check interval: 60s
  Stability threshold: 60s
  Script start time: 2026-03-02 17:30:00

⚠️  IMPORTANT: Only files modified AFTER script start will be moved
⚠️  Pre-labeled files will NOT auto-move until you open/save them in X-AnyLabeling

  Initial status: 0 verified, 1568 unverified

Status: 1568 pre-labeled files waiting for review, 0 files recently modified
```

## Testing

To verify it works:

1. **Start the script:**
   ```bash
   python scripts/auto_move_verified.py
   ```

2. **Check it's not moving files:**
   ```bash
   # In another terminal
   ls data/working/*.txt | wc -l   # Should stay at 1568
   ls data/verified/*.txt | wc -l  # Should stay at 0
   ```

3. **Open X-AnyLabeling and review ONE file:**
   ```bash
   x-anylabeling
   # Open Dir: data/working/
   # Review one image
   # Press Ctrl+S to save
   # Press D to go to next image
   ```

4. **Wait 60 seconds, then check:**
   ```bash
   ls data/working/*.txt | wc -l   # Should decrease by 1 (now 1567)
   ls data/verified/*.txt | wc -l  # Should increase by 1 (now 1)
   ```

5. **Check logs:**
   ```bash
   tail -f logs/auto_move.log
   # Should show: "✓ Moved: image_name → verified/"
   ```

## Code Location

The implementation is in `scripts/auto_move_verified.py`:

- **Line 152:** Records start timestamp
- **Lines 161-162:** Logs warning about timestamp-based verification
- **Lines 180-185:** Checks file modification time against start time
- **Line 183:** Skips files modified before script start

## Documentation Updated

All documentation now explains this behavior:

- ✅ **README.md** - Option A workflow description
- ✅ **QUICKSTART.md** - Terminal 1 setup and workflow diagram
- ✅ **IMPLEMENTATION_SUMMARY.md** - Technical details and workflow

## Summary

**Before:** All stable files (60s old) were moved → Pre-labeled files moved immediately

**After:** Only files modified AFTER script start are moved → Pre-labeled files skipped until you review them

**Guarantee:** Script can ONLY move files you've opened and saved in X-AnyLabeling!

---

**Last Updated:** March 2, 2026
