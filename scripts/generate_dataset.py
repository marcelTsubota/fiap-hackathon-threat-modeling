#!/usr/bin/env python3
"""
generate_dataset.py — Synthetic Dataset Generator for FIAP Hackathon Threat Modeling

Generates YOLO-format training images by compositing architecture icons
onto randomized canvas backgrounds.

Features:
  - 1280x720 canvas (configurable)
  - 3–10 components per image (configurable)
  - Scale randomization (0.6x–1.4x)
  - Slight rotation (±15°)
  - Brightness jitter
  - Overlap control (IoU < 0.2)
  - Background color/gradient variation
  - YOLO format labels (class x_center y_center width height)

Usage:
  python scripts/generate_dataset.py --num-images 500 --output-dir data

Pre-requisites:
  - Place icon PNGs in data/raw_icons/<ClassName>/
    e.g. data/raw_icons/User/user_01.png
         data/raw_icons/Database/db_aws.png
  - Each subfolder name must match a class name from CLASS_NAMES below.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

# ---------------------------------------------------------------------------
# Fixed class list (DO NOT CHANGE ORDER — matches data.yaml)
# ---------------------------------------------------------------------------
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

CLASS_NAME_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Mapping of folder-friendly names to canonical names
_FOLDER_ALIASES: Dict[str, str] = {
    "SecurityService": "Security Service (WAF/Firewall/Shield)",
    "Security Service": "Security Service (WAF/Firewall/Shield)",
    "WAF": "Security Service (WAF/Firewall/Shield)",
    "Firewall": "Security Service (WAF/Firewall/Shield)",
    "AuthService": "Authentication Service",
    "Auth Service": "Authentication Service",
    "AuthenticationService": "Authentication Service",
    "AppServer": "Application Server",
    "ApplicationServer": "Application Server",
    "WebApp": "Web Application",
    "WebApplication": "Web Application",
    "MobileApp": "Mobile Application",
    "MobileApplication": "Mobile Application",
    "ExternalSystem": "External System",
    "External_System": "External System",
    "APIGateway": "API Gateway",
    "API_Gateway": "API Gateway",
    "LoadBalancer": "Load Balancer",
    "Load_Balancer": "Load Balancer",
    "ServerlessFunction": "Serverless Function",
    "Serverless": "Serverless Function",
    "MessageQueue": "Message Queue",
    "Message_Queue": "Message Queue",
    "FileStorage": "File Storage",
    "File_Storage": "File Storage",
}


# ---------------------------------------------------------------------------
# Fallback shape generator (when no icons are available)
# ---------------------------------------------------------------------------

_SHAPE_COLORS: Dict[str, str] = {
    "User": "#4A90D9",
    "External System": "#7B68EE",
    "API Gateway": "#FF6B35",
    "Load Balancer": "#2ECC71",
    "Web Application": "#3498DB",
    "Mobile Application": "#9B59B6",
    "Application Server": "#E74C3C",
    "Serverless Function": "#F39C12",
    "Database": "#1ABC9C",
    "Cache": "#E91E63",
    "Message Queue": "#FF9800",
    "File Storage": "#607D8B",
    "Authentication Service": "#8BC34A",
    "Security Service (WAF/Firewall/Shield)": "#F44336",
}

_SHORT_LABELS: Dict[str, str] = {
    "User": "USR",
    "External System": "EXT",
    "API Gateway": "GW",
    "Load Balancer": "LB",
    "Web Application": "WEB",
    "Mobile Application": "MOB",
    "Application Server": "APP",
    "Serverless Function": "FN",
    "Database": "DB",
    "Cache": "CACHE",
    "Message Queue": "MQ",
    "File Storage": "FS",
    "Authentication Service": "AUTH",
    "Security Service (WAF/Firewall/Shield)": "SEC",
}


def _generate_shape_icon(class_name: str, size: int = 80) -> Image.Image:
    """Generate a simple geometric icon for a given class when real icons are unavailable."""
    color = _SHAPE_COLORS.get(class_name, "#888888")
    label = _SHORT_LABELS.get(class_name, class_name[:3].upper())

    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Different shapes per class type
    margin = 4
    if class_name in ("User", "Mobile Application"):
        # Circle
        draw.ellipse([margin, margin, size - margin, size - margin], fill=color, outline="black", width=2)
    elif class_name in ("Database", "Cache"):
        # Cylinder-like (rounded rect)
        draw.rounded_rectangle([margin, margin, size - margin, size - margin], radius=size // 4, fill=color, outline="black", width=2)
    elif class_name in ("Security Service (WAF/Firewall/Shield)", "Authentication Service"):
        # Shield/hexagon-like
        pts = [
            (size // 2, margin),
            (size - margin, size // 4),
            (size - margin, 3 * size // 4),
            (size // 2, size - margin),
            (margin, 3 * size // 4),
            (margin, size // 4),
        ]
        draw.polygon(pts, fill=color, outline="black", width=2)
    elif class_name in ("Serverless Function", "Message Queue"):
        # Diamond
        mid = size // 2
        pts = [(mid, margin), (size - margin, mid), (mid, size - margin), (margin, mid)]
        draw.polygon(pts, fill=color, outline="black", width=2)
    else:
        # Rectangle
        draw.rectangle([margin, margin, size - margin, size - margin], fill=color, outline="black", width=2)

    # Draw label text
    try:
        font = ImageFont.truetype("arial.ttf", max(10, size // 6))
    except (IOError, OSError):
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    tx = (size - tw) // 2
    ty = (size - th) // 2
    draw.text((tx, ty), label, fill="white", font=font)

    return img


# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def _iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """Compute IoU between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def _random_background(width: int, height: int) -> Image.Image:
    """Generate a random background (solid, gradient, or noisy)."""
    choice = random.choice(["solid", "gradient_h", "gradient_v", "noisy"])

    if choice == "solid":
        r, g, b = random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)
        return Image.new("RGB", (width, height), (r, g, b))

    elif choice in ("gradient_h", "gradient_v"):
        arr = np.zeros((height, width, 3), dtype=np.uint8)
        c1 = np.array([random.randint(200, 255) for _ in range(3)])
        c2 = np.array([random.randint(200, 255) for _ in range(3)])
        for i in range(height if choice == "gradient_v" else width):
            t = i / (height if choice == "gradient_v" else width)
            color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
            if choice == "gradient_v":
                arr[i, :] = color
            else:
                arr[:, i] = color
        return Image.fromarray(arr)

    else:  # noisy
        base = random.randint(230, 250)
        noise = np.random.randint(-10, 10, (height, width, 3), dtype=np.int16)
        arr = np.clip(base + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def _augment_icon(icon: Image.Image, scale: float, rotation: float, brightness: float) -> Image.Image:
    """Apply scale, rotation, and brightness augmentation to an icon."""
    w, h = icon.size
    new_w = max(16, int(w * scale))
    new_h = max(16, int(h * scale))
    icon = icon.resize((new_w, new_h), Image.LANCZOS)

    if abs(rotation) > 0.5:
        icon = icon.rotate(rotation, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    if abs(brightness - 1.0) > 0.01:
        # Convert to RGB for brightness, then back
        rgb = icon.convert("RGB")
        rgb = ImageEnhance.Brightness(rgb).enhance(brightness)
        # Re-apply alpha
        if icon.mode == "RGBA":
            r, g, b = rgb.split()
            icon = Image.merge("RGBA", (r, g, b, icon.split()[3]))
        else:
            icon = rgb

    return icon


# ---------------------------------------------------------------------------
# Icon loader
# ---------------------------------------------------------------------------

def load_icons(raw_icons_dir: str) -> Dict[str, List[Image.Image]]:
    """
    Load icon images from data/raw_icons/<ClassName>/*.png

    Returns a dict mapping canonical class name -> list of PIL Images.
    If no icons directory exists, returns empty dict (fallback shapes will be used).
    """
    icons: Dict[str, List[Image.Image]] = {}
    root = Path(raw_icons_dir)

    if not root.exists():
        print(f"[WARN] raw_icons directory not found: {root}. Using generated shapes.")
        return icons

    for folder in root.iterdir():
        if not folder.is_dir():
            continue

        # Resolve folder name to canonical class name
        folder_name = folder.name
        canonical = None

        # Direct match
        if folder_name in CLASS_NAME_TO_IDX:
            canonical = folder_name
        # Alias match
        elif folder_name in _FOLDER_ALIASES:
            canonical = _FOLDER_ALIASES[folder_name]
        else:
            # Try case-insensitive
            for cn in CLASS_NAMES:
                if cn.lower().replace(" ", "") == folder_name.lower().replace(" ", "").replace("_", ""):
                    canonical = cn
                    break

        if canonical is None:
            print(f"[WARN] Skipping unknown icon folder: {folder_name}")
            continue

        imgs = []
        for f in sorted(folder.glob("*.png")):
            try:
                img = Image.open(f).convert("RGBA")
                imgs.append(img)
            except Exception as e:
                print(f"[WARN] Failed to load {f}: {e}")

        if imgs:
            icons[canonical] = imgs
            print(f"  Loaded {len(imgs)} icons for '{canonical}'")

    return icons


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_single_image(
    icons: Dict[str, List[Image.Image]],
    canvas_width: int = 1280,
    canvas_height: int = 720,
    min_components: int = 3,
    max_components: int = 10,
    max_iou: float = 0.2,
    icon_base_size: int = 80,
    scale_range: Tuple[float, float] = (0.6, 1.4),
    rotation_range: Tuple[float, float] = (-15.0, 15.0),
    brightness_range: Tuple[float, float] = (0.8, 1.2),
) -> Tuple[Image.Image, List[Tuple[int, float, float, float, float]]]:
    """
    Generate one synthetic diagram image with YOLO-format labels.

    Returns:
      (image, labels) where labels is list of (class_idx, x_center, y_center, w, h) normalized to [0,1].
    """
    bg = _random_background(canvas_width, canvas_height)
    num_components = random.randint(min_components, max_components)

    # Pick which classes to place (can repeat)
    chosen_classes = random.choices(CLASS_NAMES, k=num_components)

    placed_boxes: List[Tuple[int, int, int, int]] = []
    labels: List[Tuple[int, float, float, float, float]] = []

    for class_name in chosen_classes:
        class_idx = CLASS_NAME_TO_IDX[class_name]

        # Get icon (real or generated)
        if class_name in icons and icons[class_name]:
            base_icon = random.choice(icons[class_name]).copy()
            # Resize to base size while keeping aspect ratio
            base_icon.thumbnail((icon_base_size * 2, icon_base_size * 2), Image.LANCZOS)
        else:
            base_icon = _generate_shape_icon(class_name, size=icon_base_size)

        # Augment
        scale = random.uniform(*scale_range)
        rotation = random.uniform(*rotation_range)
        brightness = random.uniform(*brightness_range)
        augmented = _augment_icon(base_icon, scale, rotation, brightness)

        iw, ih = augmented.size

        # Try to place without excessive overlap
        placed = False
        for _ in range(50):  # max attempts
            x = random.randint(0, max(0, canvas_width - iw))
            y = random.randint(0, max(0, canvas_height - ih))
            box = (x, y, x + iw, y + ih)

            # Check IoU with existing boxes
            overlap_ok = all(_iou(box, pb) < max_iou for pb in placed_boxes)
            if overlap_ok:
                placed = True
                placed_boxes.append(box)

                # Paste onto canvas
                if augmented.mode == "RGBA":
                    bg.paste(augmented, (x, y), augmented)
                else:
                    bg.paste(augmented, (x, y))

                # YOLO label (normalized)
                x_center = (x + iw / 2) / canvas_width
                y_center = (y + ih / 2) / canvas_height
                w_norm = iw / canvas_width
                h_norm = ih / canvas_height
                labels.append((class_idx, x_center, y_center, w_norm, h_norm))
                break

        if not placed:
            # Could not place without overlap; skip this component
            pass

    return bg, labels


def generate_dataset(
    num_images: int,
    output_dir: str,
    raw_icons_dir: str,
    canvas_width: int = 1280,
    canvas_height: int = 720,
    min_components: int = 3,
    max_components: int = 10,
    seed: Optional[int] = None,
) -> None:
    """Generate the full synthetic dataset."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    out = Path(output_dir)
    images_dir = out / "images" / "all"
    labels_dir = out / "labels" / "all"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_images} synthetic images...")
    print(f"  Canvas: {canvas_width}x{canvas_height}")
    print(f"  Components per image: {min_components}-{max_components}")
    print(f"  Output: {out}")
    print()

    # Load icons
    icons = load_icons(raw_icons_dir)
    if not icons:
        print("[INFO] No real icons found; using generated geometric shapes.")
        print("       For better results, add icons to data/raw_icons/<ClassName>/\n")

    class_counts: Dict[int, int] = {i: 0 for i in range(len(CLASS_NAMES))}

    for i in range(num_images):
        img, labels = generate_single_image(
            icons=icons,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            min_components=min_components,
            max_components=max_components,
        )

        # Save image
        img_name = f"diagram_{i:05d}.jpg"
        img_path = images_dir / img_name
        img.save(str(img_path), quality=95)

        # Save label
        lbl_name = f"diagram_{i:05d}.txt"
        lbl_path = labels_dir / lbl_name
        with open(lbl_path, "w") as f:
            for cls_idx, xc, yc, w, h in labels:
                f.write(f"{cls_idx} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
                class_counts[cls_idx] += 1

        if (i + 1) % 50 == 0 or i == 0:
            print(f"  [{i + 1}/{num_images}] generated")

    # Summary
    print(f"\nDataset generation complete!")
    print(f"  Images: {images_dir}")
    print(f"  Labels: {labels_dir}")
    print(f"\nClass distribution:")
    for idx, count in sorted(class_counts.items()):
        print(f"  {idx:2d} {CLASS_NAMES[idx]:45s} -> {count:5d} instances")

    total = sum(class_counts.values())
    print(f"\n  Total instances: {total}")
    print(f"  Avg per image:   {total / num_images:.1f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic dataset for YOLO training (architectural diagrams)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_dataset.py --num-images 500
  python scripts/generate_dataset.py --num-images 300 --seed 42
  python scripts/generate_dataset.py --num-images 500 --icons data/raw_icons
        """,
    )
    parser.add_argument("--num-images", type=int, default=500, help="Number of images to generate (default: 500)")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory (default: data)")
    parser.add_argument("--icons", type=str, default="data/raw_icons", help="Path to raw icons directory")
    parser.add_argument("--width", type=int, default=1280, help="Canvas width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Canvas height (default: 720)")
    parser.add_argument("--min-components", type=int, default=3, help="Min components per image (default: 3)")
    parser.add_argument("--max-components", type=int, default=10, help="Max components per image (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    generate_dataset(
        num_images=args.num_images,
        output_dir=args.output_dir,
        raw_icons_dir=args.icons,
        canvas_width=args.width,
        canvas_height=args.height,
        min_components=args.min_components,
        max_components=args.max_components,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
