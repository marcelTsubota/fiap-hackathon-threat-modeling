from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from PIL import Image
from ultralytics import YOLO  # type: ignore

from backend.schemas import BBox, ComponentClass, Detection, DiagramDetections, ImageMeta


# -----------------------------
# Exceptions
# -----------------------------


class ModelNotFoundError(FileNotFoundError):
    """Raised when the YOLO weights file cannot be found."""


class InvalidImageError(ValueError):
    """Raised when the provided input cannot be decoded as a valid image."""


# -----------------------------
# Config / helpers
# -----------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """
    Runtime configuration for the detector.
    - conf: confidence threshold to keep a detection
    - iou: IoU threshold used by Ultralytics NMS
    - imgsz: inference image size (Ultralytics will handle resize/letterbox)
    - max_det: maximum detections per image
    """
    weights_path: str
    conf: float = 0.25
    iou: float = 0.45
    imgsz: int = 640
    max_det: int = 300


def _safe_component_label(label: str) -> ComponentClass:
    """
    Convert a YOLO class name (string) into the fixed ComponentClass enum.
    Fails fast if the model outputs an unexpected class.
    """
    try:
        return ComponentClass(label)
    except ValueError as exc:
        raise ValueError(
            f"YOLO model produced unknown class label: {label!r}. "
            "Ensure your model was trained with the project's fixed class list."
        ) from exc



def _load_image_from_path(path: Union[str, Path]) -> Image.Image:
    p = Path(path)
    if not p.exists():
        raise InvalidImageError(f"Image file not found: {str(p)}")
    try:
        img = Image.open(p)
        img = img.convert("RGB")
        return img
    except Exception as exc:
        raise InvalidImageError(f"Invalid image file: {str(p)}") from exc


def _pil_to_numpy_rgb(img: Image.Image) -> np.ndarray:
    # Ultralytics accepts numpy arrays (H, W, 3) in uint8
    arr = np.array(img)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InvalidImageError("Decoded image is not a valid RGB image.")
    return arr


def _make_detection_id(prefix: str = "cmp") -> str:
    # Stable-enough identifier for a component within a single inference run.
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# -----------------------------
# Main detector
# -----------------------------


class YoloDetector:
    """
    Thin wrapper around Ultralytics YOLO for inference and conversion to schemas.
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config = config
        self._model = self._load_model(config.weights_path)

    @staticmethod
    def _load_model(weights_path: str) -> YOLO:
        p = Path(weights_path)
        if not p.exists():
            raise ModelNotFoundError(
                f"YOLO weights not found at: {str(p)}. "
                "Provide a valid path (e.g., 'models/best.pt')."
            )
        # Ultralytics will use CPU automatically if CUDA not available.
        return YOLO(str(p))

    def detect_from_path(
        self,
        image_path: Union[str, Path],
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> DiagramDetections:
        img = _load_image_from_path(image_path)
        fn = filename or Path(image_path).name
        return self._detect(img=img, filename=fn, content_type=content_type)

    def detect_from_bytes(
        self,
        image_bytes: bytes,
        filename: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> DiagramDetections:
        # Avoid adding non-requested dependencies; use a local import.
        import io  # local import to keep module top clean

        try:
            img = Image.open(io.BytesIO(image_bytes))
            img = img.convert("RGB")
        except Exception as exc:
            raise InvalidImageError("Invalid image bytes. Could not decode image.") from exc

        return self._detect(img=img, filename=filename, content_type=content_type)

    def _detect(
        self,
        img: Image.Image,
        filename: Optional[str],
        content_type: Optional[str],
    ) -> DiagramDetections:
        arr = _pil_to_numpy_rgb(img)

        width, height = img.size
        image_meta = ImageMeta(
            filename=filename,
            width=width,
            height=height,
            content_type=content_type,
        )

        results = self._model.predict(
            source=arr,
            conf=self.config.conf,
            iou=self.config.iou,
            imgsz=self.config.imgsz,
            max_det=self.config.max_det,
            verbose=False,
        )

        detections: List[Detection] = self._convert_results(results, image_meta=image_meta)

        return DiagramDetections(image=image_meta, detections=detections)

    def _convert_results(self, results: Sequence, image_meta: ImageMeta) -> List[Detection]:
        """
        Convert Ultralytics results to our Detection list.

        Notes:
        - We assume a single image per inference call, so results[0] is used.
        - Coordinates from Ultralytics are in xyxy (absolute pixels for the input image).
        """
        if not results:
            return []

        r0 = results[0]

        # If there are no boxes, return empty list.
        if getattr(r0, "boxes", None) is None or len(r0.boxes) == 0:
            return []

        names = getattr(r0, "names", None)
        if not isinstance(names, dict):
            raise RuntimeError("Ultralytics result missing 'names' mapping.")

        dets: List[Detection] = []
        for b in r0.boxes:
            # Ultralytics: b.xyxy, b.conf, b.cls
            xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2]
            conf = float(b.conf[0].item()) if hasattr(b.conf[0], "item") else float(b.conf[0])
            cls_idx = int(b.cls[0].item()) if hasattr(b.cls[0], "item") else int(b.cls[0])

            label_name = names.get(cls_idx)
            if not isinstance(label_name, str):
                raise RuntimeError(f"Could not resolve class name for cls index: {cls_idx}")

            label = _safe_component_label(label_name)

            x1, y1, x2, y2 = (float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]))

            # Clamp to image bounds defensively (avoids negative values from rounding/letterbox effects).
            x1 = max(0.0, min(x1, float(image_meta.width)))
            y1 = max(0.0, min(y1, float(image_meta.height)))
            x2 = max(0.0, min(x2, float(image_meta.width)))
            y2 = max(0.0, min(y2, float(image_meta.height)))

            # Skip invalid / degenerate boxes after clamping.
            if x2 <= x1 or y2 <= y1:
                continue

            dets.append(
                Detection(
                    id=_make_detection_id(),
                    label=label,
                    confidence=conf,
                    bbox=BBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        # Sort by confidence desc for deterministic ordering (helps debugging and tests).
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets
