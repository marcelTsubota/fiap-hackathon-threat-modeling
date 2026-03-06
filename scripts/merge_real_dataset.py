#!/usr/bin/env python3
"""
merge_real_dataset.py — Remap real_dataset class indices to project indices and merge into data/images/all + data/labels/all.

Real dataset class mapping (source):
  0: API Gateway     -> project class 2
  1: Database        -> project class 8
  2: Cache           -> project class 9
  3: Message Queue   -> project class 10
  4: User            -> project class 0
  5: Application Server -> project class 6
  6: Load Balancer   -> project class 3

Usage:
  python scripts/merge_real_dataset.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

# Mapping from real_dataset class index to project class index
REAL_TO_PROJECT = {
    0: 2,   # API Gateway
    1: 8,   # Database
    2: 9,   # Cache
    3: 10,  # Message Queue
    4: 0,   # User
    5: 6,   # Application Server
    6: 3,   # Load Balancer
}


def merge_real_dataset(
    real_dir: str = "data/real_dataset",
    data_dir: str = "data",
) -> None:
    real_root = Path(real_dir)
    data_root = Path(data_dir)

    real_images = real_root / "images"
    real_labels = real_root / "labels"
    all_images = data_root / "images" / "all"
    all_labels = data_root / "labels" / "all"

    if not real_images.exists() or not real_labels.exists():
        print(f"[ERROR] Real dataset not found at {real_root}")
        return

    all_images.mkdir(parents=True, exist_ok=True)
    all_labels.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        list(real_images.glob("*.png")) + list(real_images.glob("*.jpg"))
    )
    print(f"Found {len(image_files)} real images in {real_images}")

    copied = 0
    skipped = 0

    for img_path in image_files:
        lbl_path = real_labels / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  [WARN] No label for {img_path.name}, skipping")
            skipped += 1
            continue

        # Remap label indices
        remapped_lines = []
        valid = True
        for line in lbl_path.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            old_class = int(parts[0])
            new_class = REAL_TO_PROJECT.get(old_class)
            if new_class is None:
                print(f"  [WARN] Unknown class {old_class} in {lbl_path.name}, skipping file")
                valid = False
                break
            remapped_lines.append(f"{new_class} {' '.join(parts[1:])}")

        if not valid or not remapped_lines:
            skipped += 1
            continue

        # Use prefix to avoid name collisions with synthetic data
        dst_img = all_images / f"real_{img_path.name}"
        dst_lbl = all_labels / f"real_{img_path.stem}.txt"

        shutil.copy2(str(img_path), str(dst_img))
        dst_lbl.write_text("\n".join(remapped_lines) + "\n")
        copied += 1

    print(f"\nDone! Merged {copied} images+labels, skipped {skipped}")
    print(f"  Images -> {all_images}")
    print(f"  Labels -> {all_labels}")
    total = len(list(all_images.glob("*.png"))) + len(list(all_images.glob("*.jpg")))
    print(f"  Total images in all/: {total}")


if __name__ == "__main__":
    merge_real_dataset()
