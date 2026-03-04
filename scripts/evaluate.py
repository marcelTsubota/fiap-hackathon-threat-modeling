#!/usr/bin/env python3
"""
evaluate.py — Model Evaluation Script for FIAP Hackathon

Evaluates a trained YOLO model and prints key metrics:
  - mAP@0.5 (primary metric)
  - mAP@0.5:0.95
  - Precision
  - Recall
  - Per-class AP
  - Confusion matrix (saved as image)

Also runs inference on sample images for visual inspection.

Usage:
  python scripts/evaluate.py
  python scripts/evaluate.py --model models/best.pt --data data/data.yaml
  python scripts/evaluate.py --model models/best.pt --sample-images data/images/val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from ultralytics import YOLO  # type: ignore


# Fixed class list (must match data.yaml)
CLASS_NAMES: List[str] = [
    "User",
    "External System",
    "API Gateway",
    "Load Balancer",
    "Web Application",
    "Mobile Application",
    "Application Server",
    "Serverless Function",
    "Database",
    "Cache",
    "Message Queue",
    "File Storage",
    "Authentication Service",
    "Security Service (WAF/Firewall/Shield)",
]


def evaluate(
    model_path: str = "models/best.pt",
    data_yaml: str = "data/data.yaml",
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    output_dir: str = "runs/evaluate",
    conf: float = 0.25,
    iou: float = 0.45,
) -> Dict[str, Any]:
    """
    Run YOLO validation and return metrics dict.
    """
    model_p = Path(model_path)
    data_p = Path(data_yaml)

    if not model_p.exists():
        print(f"[ERROR] Model not found: {model_p}")
        print("        Train a model first with train_yolo.py")
        sys.exit(1)

    if not data_p.exists():
        print(f"[ERROR] data.yaml not found: {data_p}")
        sys.exit(1)

    print("=" * 60)
    print("YOLO Evaluation — FIAP Hackathon Threat Modeling")
    print("=" * 60)
    print(f"  Model:   {model_path}")
    print(f"  Data:    {data_yaml}")
    print(f"  ImgSz:   {imgsz}")
    print(f"  Conf:    {conf}")
    print(f"  IoU:     {iou}")
    print("=" * 60)

    model = YOLO(model_path)

    # Run validation
    results = model.val(
        data=str(data_p.resolve()),
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device or None,
        project=output_dir,
        name="val",
        exist_ok=True,
        verbose=True,
    )

    # Extract metrics
    metrics: Dict[str, Any] = {}

    # Box metrics
    box = getattr(results, "box", None)
    if box is not None:
        metrics["mAP50"] = float(getattr(box, "map50", 0.0))
        metrics["mAP50_95"] = float(getattr(box, "map", 0.0))

        # Per-class metrics
        ap50_per_class = getattr(box, "ap50", None)
        if ap50_per_class is not None:
            per_class = {}
            for i, ap in enumerate(ap50_per_class):
                if i < len(CLASS_NAMES):
                    per_class[CLASS_NAMES[i]] = round(float(ap), 4)
            metrics["per_class_ap50"] = per_class

    # Overall P, R
    mp = getattr(results, "mp", None) or getattr(box, "mp", None) if box else None
    mr = getattr(results, "mr", None) or getattr(box, "mr", None) if box else None

    if mp is not None:
        metrics["precision"] = float(mp)
    if mr is not None:
        metrics["recall"] = float(mr)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(f"\n  mAP@0.5:      {metrics.get('mAP50', 'N/A'):.4f}" if isinstance(metrics.get('mAP50'), float) else "  mAP@0.5:      N/A")
    print(f"  mAP@0.5:0.95: {metrics.get('mAP50_95', 'N/A'):.4f}" if isinstance(metrics.get('mAP50_95'), float) else "  mAP@0.5:0.95: N/A")
    print(f"  Precision:    {metrics.get('precision', 'N/A'):.4f}" if isinstance(metrics.get('precision'), float) else "  Precision:    N/A")
    print(f"  Recall:       {metrics.get('recall', 'N/A'):.4f}" if isinstance(metrics.get('recall'), float) else "  Recall:       N/A")

    if "per_class_ap50" in metrics:
        print("\n  Per-class AP@0.5:")
        for cls_name, ap in metrics["per_class_ap50"].items():
            bar = "█" * int(ap * 30) + "░" * (30 - int(ap * 30))
            print(f"    {cls_name:45s} {ap:.4f} {bar}")

    print("=" * 60)

    # Save metrics JSON
    out_path = Path(output_dir) / "val" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics saved to: {out_path}")

    return metrics


def run_sample_inference(
    model_path: str = "models/best.pt",
    images_dir: str = "data/images/val",
    output_dir: str = "runs/evaluate/samples",
    conf: float = 0.35,
    max_images: int = 10,
) -> None:
    """Run inference on sample images and save annotated results."""
    model_p = Path(model_path)
    imgs_p = Path(images_dir)

    if not model_p.exists():
        print(f"[SKIP] Model not found: {model_p}")
        return

    if not imgs_p.exists():
        print(f"[SKIP] Images directory not found: {imgs_p}")
        return

    image_files = sorted(list(imgs_p.glob("*.jpg")) + list(imgs_p.glob("*.png")))[:max_images]
    if not image_files:
        print("[SKIP] No images found for sample inference.")
        return

    print(f"\nRunning sample inference on {len(image_files)} images...")

    model = YOLO(model_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for img_path in image_files:
        results = model.predict(
            source=str(img_path),
            conf=conf,
            save=True,
            project=str(out),
            name="predict",
            exist_ok=True,
            verbose=False,
        )

        # Print detection summary
        r = results[0]
        n_det = len(r.boxes) if r.boxes is not None else 0
        print(f"  {img_path.name}: {n_det} detections")

    print(f"\nAnnotated images saved to: {out / 'predict'}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained YOLO model and generate metrics report",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate.py
  python scripts/evaluate.py --model models/best.pt
  python scripts/evaluate.py --sample-images data/images/val --max-samples 20
        """,
    )
    parser.add_argument("--model", type=str, default="models/best.pt", help="Path to trained model (default: models/best.pt)")
    parser.add_argument("--data", type=str, default="data/data.yaml", help="data.yaml path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="", help="Device")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--sample-images", type=str, default="data/images/val", help="Sample images for inference")
    parser.add_argument("--max-samples", type=int, default=10, help="Max sample images")
    parser.add_argument("--output", type=str, default="runs/evaluate", help="Output directory")

    args = parser.parse_args()

    # Run validation
    metrics = evaluate(
        model_path=args.model,
        data_yaml=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        output_dir=args.output,
        conf=args.conf,
        iou=args.iou,
    )

    # Run sample inference
    run_sample_inference(
        model_path=args.model,
        images_dir=args.sample_images,
        output_dir=args.output,
        conf=args.conf,
        max_images=args.max_samples,
    )

    # Final verdict
    map50 = metrics.get("mAP50", 0.0)
    print("\n" + "=" * 60)
    if isinstance(map50, float) and map50 >= 0.7:
        print(f"✅ Model is PRODUCTION-READY (mAP@0.5 = {map50:.4f} >= 0.70)")
    elif isinstance(map50, float) and map50 >= 0.5:
        print(f"⚠️  Model is ACCEPTABLE (mAP@0.5 = {map50:.4f} >= 0.50)")
    else:
        print(f"❌ Model NEEDS IMPROVEMENT (mAP@0.5 = {map50:.4f} < 0.50)")
    print("=" * 60)


if __name__ == "__main__":
    main()
