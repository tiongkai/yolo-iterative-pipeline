#!/usr/bin/env python3
"""
Migration script to convert data directories from flat to YOLO layout.

Old structure (flat):
    data/working/
        image_001.jpg
        image_001.txt
        image_002.jpg
        image_002.txt

New structure (YOLO):
    data/working/
        images/
            image_001.jpg
            image_002.jpg
        labels/
            image_001.txt
            image_002.txt
"""

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Set
import sys


# Image file extensions to look for
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


class MigrationError(Exception):
    """Custom exception for migration errors."""
    pass


def is_yolo_layout(directory: Path) -> bool:
    """Check if directory already has YOLO layout.

    Args:
        directory: Directory to check

    Returns:
        True if directory has images/ and labels/ subdirectories
    """
    images_dir = directory / "images"
    labels_dir = directory / "labels"
    return images_dir.is_dir() and labels_dir.is_dir()


def get_file_pairs(directory: Path) -> Dict[str, Dict[str, Path]]:
    """Get image/label file pairs from flat directory.

    Args:
        directory: Directory to scan

    Returns:
        Dictionary mapping stem to {"image": path, "label": path}
    """
    pairs = {}

    # Find all files directly in directory (not in subdirs)
    for file_path in directory.iterdir():
        if not file_path.is_file():
            continue

        stem = file_path.stem
        ext = file_path.suffix

        if ext in IMAGE_EXTENSIONS:
            if stem not in pairs:
                pairs[stem] = {}
            pairs[stem]["image"] = file_path
        elif ext == ".txt":
            if stem not in pairs:
                pairs[stem] = {}
            pairs[stem]["label"] = file_path

    return pairs


def validate_migration(directory: Path, pairs: Dict[str, Dict[str, Path]]) -> List[str]:
    """Validate that migration can proceed safely.

    Args:
        directory: Directory being migrated
        pairs: File pairs dictionary

    Returns:
        List of warning messages (empty if all OK)
    """
    warnings = []
    images_dir = directory / "images"
    labels_dir = directory / "labels"

    # Check for conflicts
    if images_dir.exists():
        existing_images = {f.name for f in images_dir.iterdir() if f.is_file()}
        for stem, files in pairs.items():
            if "image" in files and files["image"].name in existing_images:
                warnings.append(f"Image already exists: {files['image'].name}")

    if labels_dir.exists():
        existing_labels = {f.name for f in labels_dir.iterdir() if f.is_file()}
        for stem, files in pairs.items():
            if "label" in files and files["label"].name in existing_labels:
                warnings.append(f"Label already exists: {files['label'].name}")

    # Check for orphaned files
    for stem, files in pairs.items():
        if "image" in files and "label" not in files:
            warnings.append(f"Image without label: {files['image'].name}")
        elif "label" in files and "image" not in files:
            warnings.append(f"Label without image: {files['label'].name}")

    return warnings


def migrate_directory(
    directory: Path,
    dry_run: bool = False,
    backup: bool = False,
    force: bool = False
) -> Dict[str, int]:
    """Migrate a single directory from flat to YOLO layout.

    Args:
        directory: Directory to migrate
        dry_run: If True, only show what would be done
        backup: If True, copy files instead of moving
        force: If True, proceed despite warnings

    Returns:
        Dictionary with statistics
    """
    stats = {
        "images_moved": 0,
        "labels_moved": 0,
        "errors": 0,
        "warnings": 0
    }

    if not directory.exists():
        print(f"⚠️  Directory does not exist: {directory}")
        return stats

    if not directory.is_dir():
        print(f"⚠️  Not a directory: {directory}")
        return stats

    # Check if already in YOLO layout
    if is_yolo_layout(directory):
        print(f"✓ Already in YOLO layout: {directory}")
        return stats

    # Get file pairs
    pairs = get_file_pairs(directory)
    if not pairs:
        print(f"ℹ️  No files to migrate in: {directory}")
        return stats

    # Validate migration
    warnings = validate_migration(directory, pairs)
    if warnings:
        stats["warnings"] = len(warnings)
        print(f"\n⚠️  Found {len(warnings)} warning(s) for {directory}:")
        for warning in warnings[:10]:  # Show first 10
            print(f"   - {warning}")
        if len(warnings) > 10:
            print(f"   ... and {len(warnings) - 10} more")

        if not force and not dry_run:
            print("\nUse --force to proceed despite warnings, or --dry-run to preview changes.")
            raise MigrationError("Migration validation failed")

    # Create target directories
    images_dir = directory / "images"
    labels_dir = directory / "labels"

    if dry_run:
        print(f"\n🔍 DRY RUN - Would migrate {directory}:")
        print(f"   Would create: {images_dir}")
        print(f"   Would create: {labels_dir}")
    else:
        print(f"\n📦 Migrating {directory}:")
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        print(f"   Created: {images_dir}")
        print(f"   Created: {labels_dir}")

    # Move files
    operation = "copy" if backup else "move"
    action = "Would copy" if dry_run else "Copying" if backup else "Moving"

    for stem, files in pairs.items():
        # Move image
        if "image" in files:
            src = files["image"]
            dst = images_dir / src.name
            try:
                if not dry_run:
                    if backup:
                        shutil.copy2(src, dst)
                    else:
                        src.rename(dst)
                stats["images_moved"] += 1
            except Exception as e:
                print(f"   ❌ Error {operation}ing {src.name}: {e}")
                stats["errors"] += 1

        # Move label
        if "label" in files:
            src = files["label"]
            dst = labels_dir / src.name
            try:
                if not dry_run:
                    if backup:
                        shutil.copy2(src, dst)
                    else:
                        src.rename(dst)
                stats["labels_moved"] += 1
            except Exception as e:
                print(f"   ❌ Error {operation}ing {src.name}: {e}")
                stats["errors"] += 1

    # Print summary
    verb = "Would " + operation if dry_run else operation.capitalize() + "d"
    print(f"   {verb} {stats['images_moved']} images")
    print(f"   {verb} {stats['labels_moved']} labels")
    if stats["errors"] > 0:
        print(f"   ⚠️  {stats['errors']} errors occurred")

    return stats


def migrate_to_yolo_layout(
    root: Path,
    directories: List[str] = None,
    dry_run: bool = False,
    backup: bool = False,
    force: bool = False
) -> Dict[str, Dict[str, int]]:
    """Migrate multiple data directories from flat to YOLO layout.

    Args:
        root: Project root directory
        directories: List of directory names to migrate (default: ["working", "verified", "eval", "test"])
        dry_run: If True, only show what would be done
        backup: If True, copy files instead of moving
        force: If True, proceed despite warnings

    Returns:
        Dictionary mapping directory names to migration statistics
    """
    if directories is None:
        directories = ["working", "verified", "eval", "test"]

    data_root = root / "data"
    if not data_root.exists():
        raise MigrationError(f"Data directory not found: {data_root}")

    print("=" * 70)
    print("YOLO Layout Migration Script")
    print("=" * 70)
    print(f"\nRoot directory: {root}")
    print(f"Data directory: {data_root}")
    print(f"Directories to migrate: {', '.join(directories)}")
    if dry_run:
        print("\n🔍 DRY RUN MODE - No files will be modified")
    if backup:
        print("\n📋 BACKUP MODE - Files will be copied (not moved)")
    print()

    results = {}
    total_images = 0
    total_labels = 0
    total_errors = 0
    total_warnings = 0

    for dir_name in directories:
        directory = data_root / dir_name
        try:
            stats = migrate_directory(directory, dry_run, backup, force)
            results[dir_name] = stats
            total_images += stats["images_moved"]
            total_labels += stats["labels_moved"]
            total_errors += stats["errors"]
            total_warnings += stats["warnings"]
        except MigrationError as e:
            print(f"❌ Migration failed for {dir_name}: {e}")
            results[dir_name] = {"images_moved": 0, "labels_moved": 0, "errors": 1, "warnings": 0}
            total_errors += 1

    # Print overall summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    verb = "Would be" if dry_run else "Were"
    print(f"Total images {verb.lower()}: {total_images}")
    print(f"Total labels {verb.lower()}: {total_labels}")
    if total_warnings > 0:
        print(f"Total warnings: {total_warnings}")
    if total_errors > 0:
        print(f"Total errors: {total_errors}")

    if dry_run:
        print("\n💡 Run without --dry-run to perform the migration")
    elif backup:
        print("\n✓ Files were copied. Original files are still in place.")
        print("  Review the migration and delete original files if satisfied.")
    else:
        print("\n✓ Migration complete!")

    return results


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Migrate data directories from flat to YOLO layout",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Preview migration (dry run)
  %(prog)s --dry-run

  # Migrate with backup (copy files)
  %(prog)s --backup

  # Migrate specific directories
  %(prog)s --dirs working verified

  # Force migration despite warnings
  %(prog)s --force

  # Migrate from custom root
  %(prog)s --root /path/to/project
        """
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Project root directory (default: current directory)"
    )
    parser.add_argument(
        "--dirs",
        nargs="+",
        default=["working", "verified", "eval", "test"],
        help="Directories to migrate (default: working verified eval test)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without modifying files"
    )
    parser.add_argument(
        "--backup",
        action="store_true",
        help="Copy files instead of moving (creates backup)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Proceed despite warnings"
    )

    args = parser.parse_args()

    try:
        results = migrate_to_yolo_layout(
            root=args.root,
            directories=args.dirs,
            dry_run=args.dry_run,
            backup=args.backup,
            force=args.force
        )

        # Exit with error if any migration failed
        if any(stats["errors"] > 0 for stats in results.values()):
            sys.exit(1)

    except MigrationError as e:
        print(f"\n❌ Migration failed: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration cancelled by user", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
