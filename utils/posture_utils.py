# utils/posture_utils.py
from __future__ import annotations
import os
from typing import Tuple

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks imports (no VisionRunningMode import needed)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions

# ========== Config ==========
# Default to the FULL model you downloaded; allow override via env var.
_DEFAULT_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_full.task")
)
_MODEL_PATH = os.environ.get("POSE_TASK_PATH", _DEFAULT_MODEL)

# BlazePose landmark indices
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26

# ========== Singleton landmarker ==========
_LANDMARKER = None  # created lazily


def _get_landmarker() -> PoseLandmarker:
    """Create (once) and return a PoseLandmarker in IMAGE mode (default)."""
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER

    if not os.path.isfile(_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at '{_MODEL_PATH}'. "
            "Set POSE_TASK_PATH to the .task file or place it under models/."
        )

    base = BaseOptions(model_asset_path=_MODEL_PATH)
    # NOTE: We do NOT set running_mode — default is IMAGE, which matches .detect(...)
    opts = PoseLandmarkerOptions(
        base_options=base,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _LANDMARKER = PoseLandmarker.create_from_options(opts)
    return _LANDMARKER


def _safe_crop(frame: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop frame to bbox=(x1,y1,x2,y2) with clamping; return original frame if invalid."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return frame
    return frame[y1:y2, x1:x2]


def detect_posture(frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> str:
    """
    Return 'standing' | 'sitting' | 'unknown' for a single-person bbox.
      - frame_bgr: OpenCV frame in BGR order (as from cv2.VideoCapture)
      - bbox_xyxy: (x1, y1, x2, y2) in pixel coords
    """
    landmarker = _get_landmarker()

    roi_bgr = _safe_crop(frame_bgr, bbox_xyxy)
    if roi_bgr.size == 0:
        return "unknown"

    # MediaPipe expects SRGB (RGB). Convert BGR -> RGB.
    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # Wrap into MediaPipe Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)

    result = landmarker.detect(mp_image)
    if not result.pose_landmarks:
        return "unknown"

    # Normalized landmarks in ROI
    lm = result.pose_landmarks[0]
    hip_y = (lm[L_HIP].y + lm[R_HIP].y) / 2.0
    knee_y = (lm[L_KNEE].y + lm[R_KNEE].y) / 2.0

    gap = knee_y - hip_y  # positive if knees are lower than hips
    return "sitting" if gap < 0.08 else "standing"
