#!/usr/bin/env python3
"""
train_yolo.py — YOLO Training Script for FIAP Hackathon Threat Modeling

Trains a YOLOv8n (or v11) model for architectural component detection.

Training parameters (aligned with roadmap):
  - 30–50 epochs (default 40)
  - early stopping (patience 10)
  - imgsz 640
  - batch 16
  - mAP@0.5 as primary metric

Usage:
  python scripts/train_yolo.py
  python scripts/train_yolo.py --epochs 50 --batch 8 --device cpu
  python scripts/train_yolo.py --model yolov8s.pt --epochs 40
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import datetime
from pathlib import Path

from ultralytics import YOLO  # type: ignore


def train(
    model_name: str = "models/yolov8n.pt",
    data_yaml: str = "data/data.yaml",
    epochs: int = 40,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 10,
    device: str = "",
    project: str = "runs/detect",
    name: str = "",
    resume: bool = False,
    workers: int = 4,
) -> None:
    """Train YOLO model and copy best weights to models/best.pt."""

    data_path = Path(data_yaml)
    if not data_path.exists():
        print(f"[ERROR] data.yaml not found: {data_path}")
        print("        Run generate_dataset.py and split_dataset.py first.")
        sys.exit(1)

    # Validate train/val directories exist
    data_root = data_path.parent
    for subset in ("train", "val"):
        img_dir = data_root / "images" / subset
        if not img_dir.exists() or not list(img_dir.iterdir()):
            print(f"[ERROR] {img_dir} does not exist or is empty.")
            print("        Run split_dataset.py first.")
            sys.exit(1)

    run_name = name or f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 60)
    print("YOLO Training — FIAP Hackathon Threat Modeling")
    print("=" * 60)
    print(f"  Model:     {model_name}")
    print(f"  Data:      {data_yaml}")
    print(f"  Epochs:    {epochs}")
    print(f"  Batch:     {batch}")
    print(f"  ImgSz:     {imgsz}")
    print(f"  Patience:  {patience}")
    print(f"  Device:    {device or 'auto'}")
    print(f"  Run name:  {run_name}")
    print("=" * 60)

    # Load model
    model = YOLO(model_name)

    # Train
    results = model.train(
        data=str(data_path.resolve()),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device or None,
        project=project,
        name=run_name,
        resume=resume,
        workers=workers,
        # Common hyperparameters for detection
        optimizer="auto",
        lr0=0.01,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,
        # Augmentation (reasonable defaults)
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=15.0,
        translate=0.1,
        scale=0.5,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.0,
        # Save
        save=True,
        save_period=-1,  # save only best
        exist_ok=True,
        verbose=True,
    )

    # Resolve actual run directory from training results
    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(project) / run_name

    # Copy best weights to models/best.pt
    best_src = run_dir / "weights" / "best.pt"
    if best_src.exists():
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        dst = models_dir / "best.pt"
        shutil.copy2(str(best_src), str(dst))
        print(f"\n✅ Best weights copied to: {dst}")
    else:
        print(f"\n⚠️  best.pt not found at {best_src}. Check training output.")

    # Print key metrics
    print("\n" + "=" * 60)
    print("Training complete! Key paths:")
    print(f"  Run dir:     {run_dir}")
    print(f"  Best model:  models/best.pt")
    print(f"  Last model:  {run_dir / 'weights' / 'last.pt'}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLOv8 for architectural component detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_yolo.py                          # defaults (40 epochs, batch 16)
  python scripts/train_yolo.py --epochs 50 --batch 8    # custom
  python scripts/train_yolo.py --device cpu              # force CPU
  python scripts/train_yolo.py --model yolov8s.pt        # use small model
        """,
    )
    parser.add_argument("--model", type=str, default="models/yolov8n.pt", help="Base model (default: models/yolov8n.pt)")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="data.yaml path")
    parser.add_argument("--epochs", type=int, default=40, help="Training epochs (default: 40)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (default: 10)")
    parser.add_argument("--device", type=str, default="", help="Device: 'cpu', '0', '0,1' (default: auto)")
    parser.add_argument("--name", type=str, default="", help="Run name (default: auto-generated)")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    parser.add_argument("--workers", type=int, default=4, help="DataLoader workers (default: 4)")

    args = parser.parse_args()

    train(
        model_name=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        name=args.name,
        resume=args.resume,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
