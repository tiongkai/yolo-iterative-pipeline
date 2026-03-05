#!/usr/bin/env python3
"""
Convert detections.json to YOLO format labels.

Reads SAM3 detection output and converts to YOLO format:
- Converts bbox [x1, y1, x2, y2] to [cx, cy, w, h] normalized
- Maps class names to class IDs
- Skips empty detections
"""

import json
from pathlib import Path
import shutil


def convert_bbox_to_yolo(bbox, img_width, img_height):
    """Convert [x1, y1, x2, y2] to YOLO format [cx, cy, w, h] normalized."""
    x1, y1, x2, y2 = bbox

    # Calculate center and dimensions
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1

    # Normalize to 0-1
    cx_norm = cx / img_width
    cy_norm = cy / img_height
    w_norm = w / img_width
    h_norm = h / img_height

    return cx_norm, cy_norm, w_norm, h_norm


def main():
    # Paths
    json_path = Path('processed-data/willow/willow-boat-clean-output/detections.json')
    image_dir = Path('processed-data/willow/willow-boat-clean-output')
    output_dir = Path('data/working')

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load detections
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Create class mapping
    classes = data['text_prompts']  # ['human', 'boat', 'outboard motor']
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    print(f"Classes: {classes}")
    print(f"Class mapping: {class_to_id}")

    # Write classes.txt
    with open(output_dir / 'classes.txt', 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")

    # Convert each image
    converted = 0
    skipped_no_detections = 0
    skipped_missing_image = 0

    for img_data in data['images']:
        filename = img_data['filename']
        width = img_data['width']
        height = img_data['height']
        detections = img_data.get('detections', [])

        # Skip if no detections
        if not detections:
            skipped_no_detections += 1
            continue

        # Check if image exists
        # Try with and without _detected suffix
        base_name = filename.replace('.png', '')
        img_path = image_dir / filename
        img_path_detected = image_dir / filename.replace('.png', '_detected.png')

        if img_path.exists():
            source_img = img_path
        elif img_path_detected.exists():
            source_img = img_path_detected
        else:
            skipped_missing_image += 1
            continue

        # Convert detections to YOLO format
        yolo_lines = []
        for det in detections:
            label = det['label']
            bbox = det['bbox']
            class_id = class_to_id[label]

            cx, cy, w, h = convert_bbox_to_yolo(bbox, width, height)
            yolo_lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        # Write label file
        label_name = base_name + '.txt'
        label_path = output_dir / label_name
        with open(label_path, 'w') as f:
            f.write('\n'.join(yolo_lines))

        # Copy image
        dest_img = output_dir / source_img.name
        shutil.copy2(source_img, dest_img)

        converted += 1
        if converted % 100 == 0:
            print(f"Converted {converted} images...")

    print(f"\nConversion complete!")
    print(f"  Converted: {converted} images")
    print(f"  Skipped (no detections): {skipped_no_detections}")
    print(f"  Skipped (missing image): {skipped_missing_image}")
    print(f"\nOutput directory: {output_dir}")


if __name__ == '__main__':
    main()
