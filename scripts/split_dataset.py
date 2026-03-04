#!/usr/bin/env python3
"""
split_dataset.py — Split generated dataset into train/val sets.

Moves images and labels from data/images/all + data/labels/all
into data/images/train, data/images/val, data/labels/train, data/labels/val.

Default split: 80% train / 20% validation (stratified is not needed for
random synthetic data, but we shuffle deterministically).

Usage:
  python scripts/split_dataset.py
  python scripts/split_dataset.py --ratio 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple


def split_dataset(
    data_dir: str = "data",
    train_ratio: float = 0.8,
    seed: int = 42,
) -> None:
    """Split all images+labels into train/val."""

    root = Path(data_dir)
    all_images = root / "images" / "all"
    all_labels = root / "labels" / "all"

    if not all_images.exists():
        print(f"[ERROR] Source images directory not found: {all_images}")
        print("        Run generate_dataset.py first.")
        return

    # Collect all image files
    image_files = sorted(list(all_images.glob("*.jpg")) + list(all_images.glob("*.png")))
    if not image_files:
        print("[ERROR] No images found in", all_images)
        return

    print(f"Found {len(image_files)} images in {all_images}")

    # Shuffle deterministically
    random.seed(seed)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"Split: {len(train_files)} train / {len(val_files)} val (ratio={train_ratio})")

    # Create output directories
    for subset in ("train", "val"):
        (root / "images" / subset).mkdir(parents=True, exist_ok=True)
        (root / "labels" / subset).mkdir(parents=True, exist_ok=True)

    def _copy_files(files: List[Path], subset: str) -> Tuple[int, int]:
        img_count = 0
        lbl_count = 0
        for img_path in files:
            # Copy image
            dst_img = root / "images" / subset / img_path.name
            shutil.copy2(str(img_path), str(dst_img))
            img_count += 1

            # Copy corresponding label
            lbl_name = img_path.stem + ".txt"
            src_lbl = all_labels / lbl_name
            if src_lbl.exists():
                dst_lbl = root / "labels" / subset / lbl_name
                shutil.copy2(str(src_lbl), str(dst_lbl))
                lbl_count += 1
            else:
                print(f"  [WARN] Label not found for {img_path.name}")

        return img_count, lbl_count

    ti, tl = _copy_files(train_files, "train")
    vi, vl = _copy_files(val_files, "val")

    print(f"\nDone!")
    print(f"  Train: {ti} images, {tl} labels → {root / 'images' / 'train'}")
    print(f"  Val:   {vi} images, {vl} labels → {root / 'images' / 'val'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split dataset into train/val sets for YOLO training",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Dataset root directory (default: data)")
    parser.add_argument("--ratio", type=float, default=0.8, help="Training set ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()
    split_dataset(data_dir=args.data_dir, train_ratio=args.ratio, seed=args.seed)


if __name__ == "__main__":
    main()
