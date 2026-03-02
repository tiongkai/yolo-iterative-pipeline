#!/bin/bash
# Move verified annotations from working to verified directory
#
# Usage:
#   ./scripts/move_verified.sh              # Interactive mode
#   ./scripts/move_verified.sh --all        # Move all files
#   ./scripts/move_verified.sh pattern      # Move files matching pattern

set -e

WORKING_DIR="data/working"
VERIFIED_DIR="data/verified"

# Create verified directory if it doesn't exist
mkdir -p "$VERIFIED_DIR"

# Function to move a file pair (image + label)
move_file_pair() {
    local txt_file="$1"
    local base_name=$(basename "$txt_file" .txt)

    # Find corresponding image
    for ext in .png .jpg .jpeg .PNG .JPG .JPEG; do
        img_file="$WORKING_DIR/${base_name}${ext}"
        if [ -f "$img_file" ]; then
            # Move both files
            mv "$txt_file" "$VERIFIED_DIR/"
            mv "$img_file" "$VERIFIED_DIR/"
            echo "✓ Moved: $base_name"
            return 0
        fi
    done

    echo "⚠ Warning: No image found for $base_name"
    return 1
}

# Interactive mode - show files and ask for confirmation
interactive_mode() {
    echo "Files in $WORKING_DIR:"
    local txt_files=("$WORKING_DIR"/*.txt)

    if [ ! -e "${txt_files[0]}" ]; then
        echo "No annotation files found in $WORKING_DIR"
        exit 0
    fi

    local count=0
    for txt_file in "${txt_files[@]}"; do
        count=$((count + 1))
        echo "  $count. $(basename "$txt_file" .txt)"
    done

    echo ""
    echo "Found $count annotated images"
    read -p "Move all to verified/? (y/n): " confirm

    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        local moved=0
        for txt_file in "${txt_files[@]}"; do
            if move_file_pair "$txt_file"; then
                moved=$((moved + 1))
            fi
        done
        echo ""
        echo "✓ Moved $moved file pairs to $VERIFIED_DIR/"
        echo "Verified count: $(ls $VERIFIED_DIR/*.txt 2>/dev/null | wc -l)"
    else
        echo "Cancelled."
    fi
}

# Move all files
move_all() {
    local txt_files=("$WORKING_DIR"/*.txt)

    if [ ! -e "${txt_files[0]}" ]; then
        echo "No annotation files found in $WORKING_DIR"
        exit 0
    fi

    local moved=0
    for txt_file in "${txt_files[@]}"; do
        if move_file_pair "$txt_file"; then
            moved=$((moved + 1))
        fi
    done

    echo "✓ Moved $moved file pairs to $VERIFIED_DIR/"
    echo "Verified count: $(ls $VERIFIED_DIR/*.txt 2>/dev/null | wc -l)"
}

# Move files matching pattern
move_pattern() {
    local pattern="$1"
    local txt_files=("$WORKING_DIR"/*${pattern}*.txt)

    if [ ! -e "${txt_files[0]}" ]; then
        echo "No files matching pattern '$pattern' found"
        exit 0
    fi

    local moved=0
    for txt_file in "${txt_files[@]}"; do
        if move_file_pair "$txt_file"; then
            moved=$((moved + 1))
        fi
    done

    echo "✓ Moved $moved file pairs to $VERIFIED_DIR/"
    echo "Verified count: $(ls $VERIFIED_DIR/*.txt 2>/dev/null | wc -l)"
}

# Main logic
if [ $# -eq 0 ]; then
    # No arguments - interactive mode
    interactive_mode
elif [ "$1" = "--all" ] || [ "$1" = "-a" ]; then
    # Move all files
    move_all
else
    # Move files matching pattern
    move_pattern "$1"
fi
