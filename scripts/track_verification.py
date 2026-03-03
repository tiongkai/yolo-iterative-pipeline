#!/usr/bin/env python3
"""
Track verification status of images in data/working/.

Maintains a log of which images have been manually verified vs. unverified.
Integrates with X-AnyLabeling's verification flag (Space key).

Usage:
    python scripts/track_verification.py --mark-verified image001.png
    python scripts/track_verification.py --list-verified
    python scripts/track_verification.py --list-unverified
    python scripts/track_verification.py --stats
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


class VerificationTracker:
    """Track verification status of images."""

    def __init__(self, log_path: Path = Path('logs/verification_status.json')):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        """Load verification status from log file."""
        if self.log_path.exists():
            with open(self.log_path, 'r') as f:
                return json.load(f)
        return {
            'verified': [],
            'unverified': [],
            'last_updated': None
        }

    def _save(self):
        """Save verification status to log file."""
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.log_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def mark_verified(self, image_name: str):
        """Mark an image as manually verified."""
        if image_name in self.data['verified']:
            logger.info(f"Already verified: {image_name}")
            return

        # Remove from unverified if present
        if image_name in self.data['unverified']:
            self.data['unverified'].remove(image_name)

        # Add to verified
        self.data['verified'].append(image_name)
        self._save()
        logger.info(f"✓ Marked as verified: {image_name}")

    def mark_unverified(self, image_name: str):
        """Mark an image as unverified (pre-labeled only)."""
        if image_name in self.data['unverified']:
            return

        # Remove from verified if present
        if image_name in self.data['verified']:
            self.data['verified'].remove(image_name)

        # Add to unverified
        self.data['unverified'].append(image_name)
        self._save()

    def is_verified(self, image_name: str) -> bool:
        """Check if an image has been manually verified."""
        return image_name in self.data['verified']

    def get_verified(self) -> List[str]:
        """Get list of verified images."""
        return self.data['verified']

    def get_unverified(self) -> List[str]:
        """Get list of unverified images."""
        return self.data['unverified']

    def scan_working_dir(self, working_dir: Path):
        """
        Scan working directory and update unverified list.

        Images not in verified list are added to unverified list.
        """
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
            image_files.extend(working_dir.glob(ext))

        image_names = [img.name for img in image_files]

        # Add new images to unverified
        for name in image_names:
            if name not in self.data['verified'] and name not in self.data['unverified']:
                self.data['unverified'].append(name)

        # Remove images that no longer exist
        self.data['verified'] = [n for n in self.data['verified'] if n in image_names]
        self.data['unverified'] = [n for n in self.data['unverified'] if n in image_names]

        self._save()
        logger.info(f"Scanned {len(image_names)} images")

    def get_stats(self) -> Dict:
        """Get verification statistics."""
        return {
            'total_verified': len(self.data['verified']),
            'total_unverified': len(self.data['unverified']),
            'total': len(self.data['verified']) + len(self.data['unverified']),
            'verification_rate': (
                len(self.data['verified']) /
                (len(self.data['verified']) + len(self.data['unverified']))
                if (len(self.data['verified']) + len(self.data['unverified'])) > 0
                else 0
            ) * 100
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Track verification status of images"
    )
    parser.add_argument(
        '--mark-verified',
        type=str,
        help='Mark image as verified (e.g., image001.png)'
    )
    parser.add_argument(
        '--mark-unverified',
        type=str,
        help='Mark image as unverified'
    )
    parser.add_argument(
        '--list-verified',
        action='store_true',
        help='List all verified images'
    )
    parser.add_argument(
        '--list-unverified',
        action='store_true',
        help='List all unverified images'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show verification statistics'
    )
    parser.add_argument(
        '--scan',
        action='store_true',
        help='Scan working directory and update status'
    )
    parser.add_argument(
        '--working-dir',
        type=Path,
        default=Path('data/working'),
        help='Working directory path (default: data/working)'
    )

    args = parser.parse_args()

    tracker = VerificationTracker()

    if args.mark_verified:
        tracker.mark_verified(args.mark_verified)

    if args.mark_unverified:
        tracker.mark_unverified(args.mark_unverified)

    if args.scan:
        tracker.scan_working_dir(args.working_dir)
        print(f"\nScan complete!")

    if args.stats or not any([args.mark_verified, args.mark_unverified,
                               args.list_verified, args.list_unverified, args.scan]):
        stats = tracker.get_stats()
        print(f"\n{'='*50}")
        print(f"VERIFICATION STATUS")
        print(f"{'='*50}")
        print(f"✓ Verified:   {stats['total_verified']}")
        print(f"⚠ Unverified: {stats['total_unverified']}")
        print(f"  Total:      {stats['total']}")
        print(f"  Progress:   {stats['verification_rate']:.1f}%")
        print(f"{'='*50}\n")

    if args.list_verified:
        verified = tracker.get_verified()
        print(f"\n✓ Verified images ({len(verified)}):")
        for name in verified:
            print(f"  {name}")

    if args.list_unverified:
        unverified = tracker.get_unverified()
        print(f"\n⚠ Unverified images ({len(unverified)}):")
        for name in unverified:
            print(f"  {name}")


if __name__ == '__main__':
    main()
