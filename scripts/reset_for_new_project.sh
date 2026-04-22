#!/bin/bash
# Reset pipeline for a new project or retrain from scratch
# Usage: ./scripts/reset_for_new_project.sh [--models-only]
#   --models-only  Reset models and training state only (keeps all data/)

set -e

MODELS_ONLY=false
if [ "$1" = "--models-only" ]; then
    MODELS_ONLY=true
fi

if [ "$MODELS_ONLY" = true ]; then
    echo "🔄 Resetting models and training state (keeping data/)..."
else
    echo "🔄 Resetting pipeline for new project..."
fi

# 1. Backup old data
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="backups/$TIMESTAMP"
mkdir -p "$BACKUP_DIR"

if [ "$MODELS_ONLY" = false ]; then
    if [ -d "data/verified/images" ] && [ "$(ls -A data/verified/images)" ]; then
        echo "  📦 Backing up verified data to $BACKUP_DIR/verified/"
        cp -r data/verified "$BACKUP_DIR/"
    fi
fi

if [ -d "models/checkpoints" ] && [ "$(ls -A models/checkpoints)" ]; then
    echo "  📦 Backing up checkpoints to $BACKUP_DIR/checkpoints/"
    cp -r models/checkpoints "$BACKUP_DIR/"
fi

if [ -f "logs/training_history.json" ]; then
    echo "  📦 Backing up training history to $BACKUP_DIR/"
    cp logs/training_history.json "$BACKUP_DIR/"
fi

# 2. Clear data (full reset only)
if [ "$MODELS_ONLY" = false ]; then
    echo "  🗑️  Clearing verified data..."
    rm -rf data/verified/images/* data/verified/labels/*

    echo "  🗑️  Clearing working labels..."
    rm -rf data/working/labels/*
fi

# 3. Clear models
echo "  🗑️  Clearing active models..."
rm -f models/active/best.pt models/active/best.onnx

# Reset model config with new classes from verified/classes.txt
if [ -f "data/verified/classes.txt" ]; then
    echo "  🔄 Rebuilding models/active/config.yaml from data/verified/classes.txt..."
    CLASSES_FILE="data/verified/classes.txt"
    NC=$(grep -c . "$CLASSES_FILE")
    {
        echo "type: yolo11"
        echo "name: detection-yolo11n"
        echo "display_name: YOLO11n Detection"
        echo "model_path: best.onnx"
        echo "input_width: 1280"
        echo "input_height: 1280"
        echo "stride: 32"
        echo "nc: $NC"
        echo "classes:"
        while IFS= read -r class_name; do
            [ -n "$class_name" ] && echo "  - $class_name"
        done < "$CLASSES_FILE"
    } > models/active/config.yaml
else
    echo "  ⚠️  data/verified/classes.txt not found — update models/active/config.yaml manually after adding classes"
fi

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

# 6. Reset verification tracking (full reset only)
if [ "$MODELS_ONLY" = false ]; then
    echo "  🗑️  Resetting verification tracking..."
    cat > logs/verification_status.json << 'EOF'
{
  "verified": [],
  "unverified": []
}
EOF
fi

echo ""
echo "✅ Reset complete!"
echo ""
if [ "$MODELS_ONLY" = true ]; then
    echo "📋 Next steps:"
    echo "  1. Run: yolo-pipeline-run"
    echo "  2. Training will retrain from scratch on existing verified data"
else
    echo "📋 Next steps:"
    echo "  1. Update data/verified/classes.txt with your new classes"
    echo "  2. Copy to data/working/classes.txt"
    echo "  3. Add your new images to data/working/images/"
    echo "  4. Update configs/pipeline_config.yaml (optional)"
    echo "  5. Run: python -m pipeline.train --from-scratch"
fi
echo ""
echo "💾 Backup saved to: $BACKUP_DIR"
