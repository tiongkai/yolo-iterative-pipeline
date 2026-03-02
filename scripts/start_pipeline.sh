#!/bin/bash
# Helper script to show commands for starting the 4-terminal workflow
# This doesn't actually launch the terminals, but shows you what to run in each

cat << 'EOF'
╔═══════════════════════════════════════════════════════════════════╗
║            YOLO ITERATIVE PIPELINE - 4 TERMINAL SETUP             ║
║                     Option 2: Automatic Movement                  ║
╚═══════════════════════════════════════════════════════════════════╝

📊 Current Status:
EOF

# Show current counts
echo "   Working directory:  $(ls data/working/*.txt 2>/dev/null | wc -l) images ready for annotation"
echo "   Verified directory: $(ls data/verified/*.txt 2>/dev/null | wc -l) images verified"
echo ""

cat << 'EOF'
┌───────────────────────────────────────────────────────────────────┐
│ TERMINAL 1: Auto-Move Watcher (working/ → verified/)              │
└───────────────────────────────────────────────────────────────────┘
Run these commands:

cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
python scripts/auto_move_verified.py

This watches data/working/ and automatically moves stable files to verified/
Log: logs/auto_move.log

┌───────────────────────────────────────────────────────────────────┐
│ TERMINAL 2: Training Watcher (verified/ → training)               │
└───────────────────────────────────────────────────────────────────┘
Run these commands:

cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
yolo-pipeline-watch

This monitors data/verified/ and triggers training after 50 images
Log: logs/watcher.log

┌───────────────────────────────────────────────────────────────────┐
│ TERMINAL 3: Status Monitor (dashboard)                            │
└───────────────────────────────────────────────────────────────────┘
Run these commands:

cd /home/lenovo6/TiongKai/yolo-iterative-pipeline
source venv/bin/activate
watch -n 5 yolo-pipeline-monitor

This auto-refreshes the status every 5 seconds
Press Ctrl+C to exit

Alternative (manual refresh):
yolo-pipeline-monitor --history

┌───────────────────────────────────────────────────────────────────┐
│ TERMINAL 4: X-AnyLabeling (annotation interface)                  │
└───────────────────────────────────────────────────────────────────┘
Run this command:

x-anylabeling

In X-AnyLabeling:
1. File → Open Dir → /home/lenovo6/TiongKai/yolo-iterative-pipeline/data/working/
2. Edit → Label Settings → Add classes:
   - boat
   - human
   - outboard motor
3. (Optional) AI → Load Model → models/active/best.pt
4. Start reviewing/correcting annotations!

Keyboard shortcuts:
- D: Next image
- A: Previous image
- W: Create rectangle
- Delete: Remove selected box
- Ctrl+S: Save
- Space: Mark as verified

╔═══════════════════════════════════════════════════════════════════╗
║                         WORKFLOW FLOW                             ║
╚═══════════════════════════════════════════════════════════════════╝

1. Annotate in X-AnyLabeling (Terminal 4) → saves to data/working/
   ↓
2. Auto-move watches and validates → moves to data/verified/
   ↓
3. Training watcher counts files → triggers at 50 images
   ↓
4. Training runs → new model in models/active/
   ↓
5. Reload model in X-AnyLabeling → better predictions!
   ↓
6. Repeat!

╔═══════════════════════════════════════════════════════════════════╗
║                      STOPPING THE PIPELINE                        ║
╚═══════════════════════════════════════════════════════════════════╝

To stop:
- Terminal 1: Ctrl+C (auto-move watcher)
- Terminal 2: Ctrl+C (training watcher)
- Terminal 3: Ctrl+C (monitor)
- Terminal 4: File → Exit (X-AnyLabeling)

All progress is saved automatically!

EOF
