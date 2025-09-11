# utils/posture_utils.py
from __future__ import annotations
import os
from typing import Tuple, Optional, List

import cv2
import numpy as np
import mediapipe as mp

# MediaPipe Tasks imports (no VisionRunningMode needed)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions


# =========================
# Config / thresholds
# =========================
# Model path (downloaded earlier). Override with POSE_TASK_PATH if you want heavy/lite variants.
_DEFAULT_MODEL = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "pose_landmarker_full.task")
)
_MODEL_PATH = os.environ.get("POSE_TASK_PATH", _DEFAULT_MODEL)

# Heuristics (tune as needed)
KNEE_STANDING_MIN_DEG = float(os.environ.get("KNEE_STANDING_MIN_DEG", 165))  # knees ~ straight
KNEE_SITTING_MAX_DEG  = float(os.environ.get("KNEE_SITTING_MAX_DEG", 140))  # knees bent
GAP_STANDING_MIN      = float(os.environ.get("GAP_STANDING_MIN", 0.12))     # knee below hip (normalized ROI y)
GAP_SITTING_MAX       = float(os.environ.get("GAP_SITTING_MAX", 0.05))      # knee near hip height
TORSO_TILT_MAX_STAND  = float(os.environ.get("TORSO_TILT_MAX_STAND", 25))   # degrees from vertical
ROI_EXPAND_SCALE      = float(os.environ.get("ROI_EXPAND_SCALE", 1.15))     # enlarge bbox so ankles/hips fit


# BlazePose indices (MP Tasks uses the same convention)
L_SHOULDER, R_SHOULDER = 11, 12
L_HIP, R_HIP           = 23, 24
L_KNEE, R_KNEE         = 25, 26
L_ANKLE, R_ANKLE       = 27, 28


# =========================
# Landmarker singleton
# =========================
_LANDMARKER: Optional[PoseLandmarker] = None

def _get_landmarker() -> PoseLandmarker:
    global _LANDMARKER
    if _LANDMARKER is not None:
        return _LANDMARKER
    if not os.path.isfile(_MODEL_PATH):
        raise FileNotFoundError(
            f"Pose model not found at '{_MODEL_PATH}'. "
            "Set POSE_TASK_PATH to the .task file or place it under models/."
        )
    base = BaseOptions(model_asset_path=_MODEL_PATH)
    opts = PoseLandmarkerOptions(
        base_options=base,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    _LANDMARKER = PoseLandmarker.create_from_options(opts)
    return _LANDMARKER


# =========================
# Geometry helpers
# =========================
def _expand_bbox(bbox: Tuple[int, int, int, int], w: int, h: int, scale: float) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    bw, bh = (x2 - x1) * scale, (y2 - y1) * scale
    nx1 = int(max(0, min(w - 1, round(cx - bw / 2))))
    ny1 = int(max(0, min(h - 1, round(cy - bh / 2))))
    nx2 = int(max(0, min(w - 1, round(cx + bw / 2))))
    ny2 = int(max(0, min(h - 1, round(cy + bh / 2))))
    if nx2 <= nx1 or ny2 <= ny1:
        return bbox
    return (nx1, ny1, nx2, ny2)

def _crop(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return frame[y1:y2, x1:x2]

def _angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle ABC in degrees with vertex at B.
    a, b, c are 2D points [x, y] in same coordinate system.
    Returns 0..180.
    """
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1) + 1e-8
    n2 = np.linalg.norm(v2) + 1e-8
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def _torso_tilt_deg(hip_mid: np.ndarray, sh_mid: np.ndarray) -> float:
    """
    Angle between torso vector (hip->shoulder) and vertical axis (0,-1).
    0° = perfectly upright, larger = more tilted.
    """
    v = sh_mid - hip_mid  # pointing up ideally
    # vertical reference vector (pointing up): (0, -1) in image coords where y increases downward
    ref = np.array([0.0, -1.0], dtype=np.float32)
    n1 = np.linalg.norm(v) + 1e-8
    n2 = np.linalg.norm(ref)
    cosang = np.clip(np.dot(v / n1, ref / n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


# =========================
# Posture detection
# =========================
def _landmarks_from_image(image_rgb: np.ndarray):
    lmkr = _get_landmarker()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    result = lmkr.detect(mp_image)
    if not result.pose_landmarks:
        return None
    return result.pose_landmarks[0]  # list of normalized landmarks with .x/.y

def _points_from_landmarks(lm, w: int, h: int) -> dict:
    """
    Convert normalized landmarks to pixel coordinates within the ROI (w,h).
    """
    def P(idx: int) -> np.ndarray:
        pt = lm[idx]
        return np.array([pt.x * w, pt.y * h], dtype=np.float32)
    return {
        "l_sh": P(L_SHOULDER), "r_sh": P(R_SHOULDER),
        "l_hip": P(L_HIP), "r_hip": P(R_HIP),
        "l_knee": P(L_KNEE), "r_knee": P(R_KNEE),
        "l_ank": P(L_ANKLE), "r_ank": P(R_ANKLE),
    }

def _classify_from_points(P: dict, h_norm: float) -> str:
    """
    Classify posture using multiple signals.
    h_norm: ROI height used to normalize vertical gaps (just use 1.0 if using normalized).
    """
    # Midpoints
    hip_mid = (P["l_hip"] + P["r_hip"]) / 2.0
    knee_mid = (P["l_knee"] + P["r_knee"]) / 2.0
    sh_mid   = (P["l_sh"] + P["r_sh"]) / 2.0

    # Angles (degrees)
    # Knee angles: angle at knee between Hip-Knee-Ankle
    lknee = _angle_deg(P["l_hip"], P["l_knee"], P["l_ank"])
    rknee = _angle_deg(P["r_hip"], P["r_knee"], P["r_ank"])
    knee_angle = np.nanmean([lknee, rknee])

    # Hip–knee vertical gap in normalized ROI coordinates (y grows downward)
    gap = (knee_mid[1] - hip_mid[1]) / max(h_norm, 1e-6)

    # Torso tilt from vertical
    torso_tilt = _torso_tilt_deg(hip_mid, sh_mid)

    # --- Rules ---
    standing = (
        knee_angle >= KNEE_STANDING_MIN_DEG and
        gap >= GAP_STANDING_MIN and
        torso_tilt <= TORSO_TILT_MAX_STAND
    )
    sitting = (
        knee_angle <= KNEE_SITTING_MAX_DEG and
        gap <= GAP_SITTING_MAX
    )

    if standing and not sitting:
        return "standing"
    if sitting and not standing:
        return "sitting"

    # If ambiguous, bias using knee angle first, then gap
    if knee_angle >= (KNEE_STANDING_MIN_DEG - 5):
        return "standing"
    if knee_angle <= (KNEE_SITTING_MAX_DEG + 5):
        return "sitting"

    return "unknown"


def detect_posture(frame_bgr: np.ndarray, bbox_xyxy: Tuple[int, int, int, int]) -> str:
    """
    Return 'standing' | 'sitting' | 'unknown' for a single-person bbox.
    Steps:
      1) Expand bbox to include hips/ankles, crop ROI
      2) Run Pose Landmarker; if fails, retry on whole frame
      3) Compute knee angles, hip–knee gap, torso tilt; classify
    """
    h, w = frame_bgr.shape[:2]

    # 1) Expand ROI and crop
    x1, y1, x2, y2 = bbox_xyxy
    x1e, y1e, x2e, y2e = _expand_bbox((x1, y1, x2, y2), w, h, ROI_EXPAND_SCALE)
    roi_bgr = _crop(frame_bgr, (x1e, y1e, x2e, y2e))
    if roi_bgr.size == 0:
        return "unknown"

    roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)

    # 2) Landmarks on ROI; fallback to full frame if needed
    lm = _landmarks_from_image(roi_rgb)
    used_roi = True
    if lm is None:
        full_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        lm = _landmarks_from_image(full_rgb)
        used_roi = False
    if lm is None:
        return "unknown"

    # 3) Convert normalized landmarks to pixels in the coordinate system we used
    if used_roi:
        Pw, Ph = roi_rgb.shape[1], roi_rgb.shape[0]
        P = _points_from_landmarks(lm, Pw, Ph)
    else:
        Pw, Ph = frame_bgr.shape[1], frame_bgr.shape[0]
        P = _points_from_landmarks(lm, Pw, Ph)

    label = _classify_from_points(P, h_norm=float(Ph))
    return label
