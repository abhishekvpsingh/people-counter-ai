# utils/yolo_utils.py
from __future__ import annotations
from typing import List, Tuple, Optional
import numpy as np

from ultralytics import YOLO


def load_model(
    model_path: str = "models/yolov8n.pt",
    imgsz: int = 640,
    device: Optional[str] = None,
):
    """
    Load a YOLOv8 model. We accept imgsz for API compatibility with callers,
    but inference size is applied in detect_people().
    """
    model = YOLO(model_path)
    # Note: In Ultralytics v8, you typically pass `device` to .predict(...)
    # If you want to hard-pin device here, you could use: model.to(device)
    # but it's optional and not always available depending on build.
    return model


def detect_people(
    model,
    frame_bgr: np.ndarray,
    imgsz: int = 640,
    conf: float = 0.5,
    iou: float = 0.45,
    max_det: int = 100,
) -> List[Tuple[int, int, int, int]]:
    """
    Run person detection and return a list of (x1, y1, x2, y2) pixel boxes.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return []

    # Ultralytics expects images in RGB or BGR ndarray; BGR is fine.
    # class 0 = 'person'
    results = model.predict(
        source=frame_bgr,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        classes=[0],
        max_det=max_det,
        verbose=False,
    )

    if not results:
        return []

    r = results[0]
    if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
        return []

    xyxy = r.boxes.xyxy.cpu().numpy()  # (N,4) float
    boxes: List[Tuple[int, int, int, int]] = []
    h, w = frame_bgr.shape[:2]
    for x1, y1, x2, y2 in xyxy:
        xi1 = max(0, min(w - 1, int(x1)))
        yi1 = max(0, min(h - 1, int(y1)))
        xi2 = max(0, min(w - 1, int(x2)))
        yi2 = max(0, min(h - 1, int(y2)))
        if xi2 > xi1 and yi2 > yi1:
            boxes.append((xi1, yi1, xi2, yi2))
    return boxes
