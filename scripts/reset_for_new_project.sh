#!/bin/bash
# Reset pipeline for a new project
# Usage: ./scripts/reset_for_new_project.sh

set -e

echo "🔄 Resetting pipeline for new project..."

# 1. Backup old data
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

if [ -d "data/verified/images" ] && [ "$(ls -A data/verified/images)" ]; then
    echo "  📦 Backing up verified data to $BACKUP_DIR/verified/"
    cp -r data/verified "$BACKUP_DIR/"
fi

if [ -d "models/checkpoints" ] && [ "$(ls -A models/checkpoints)" ]; then
    echo "  📦 Backing up checkpoints to $BACKUP_DIR/checkpoints/"
    cp -r models/checkpoints "$BACKUP_DIR/"
fi

if [ -f "logs/training_history.json" ]; then
    echo "  📦 Backing up training history to $BACKUP_DIR/"
    cp logs/training_history.json "$BACKUP_DIR/"
fi

# 2. Clear verified data
echo "  🗑️  Clearing verified data..."
rm -rf data/verified/images/* data/verified/labels/*

# 3. Clear models
echo "  🗑️  Clearing active models..."
rm -f models/active/best.pt models/active/best.onnx

echo "  🗑️  Clearing checkpoints..."
rm -rf models/checkpoints/*

# 4. Clear logs
echo "  🗑️  Clearing training history..."
rm -f logs/training_history.json
rm -f logs/priority_queue.txt
rm -f logs/.training.lock

# 5. Clear YOLO cache files
echo "  🗑️  Clearing YOLO cache..."
find data -name "*.cache" -type f -delete

# 6. Reset verification tracking
echo "  🗑️  Resetting verification tracking..."
cat > logs/verification_status.json << 'EOF'
{
  "verified": [],
  "unverified": []
}
EOF

echo ""
echo "✅ Reset complete!"
echo ""
echo "📋 Next steps:"
echo "  1. Update data/verified/classes.txt with your new classes"
echo "  2. Copy to data/working/classes.txt"
echo "  3. Add your new images to data/working/images/"
echo "  4. Update configs/pipeline_config.yaml (optional)"
echo "  5. Run: python -m pipeline.train --from-scratch"
echo ""
echo "💾 Backup saved to: $BACKUP_DIR"
