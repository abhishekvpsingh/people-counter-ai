#!/usr/bin/env python3
"""
Streamlined Streamlit app for People + Posture detection.

Minimal UI:
- Choose source: Webcam (index) or Video file (path)
- Start / Stop toggle
- On‑frame counts (People / Standing / Sitting) and simple live metrics

All other tuning is hidden and uses sensible defaults.
"""

import platform
import time
from typing import List, Optional, Tuple

import cv2
import streamlit as st

# Prefer utils.* layout; fall back to top-level files if needed
try:
    from utils.yolo_utils import load_model, detect_people
    from utils.posture_utils import detect_posture
    from utils.debounce import PostureDebouncer
    from utils.overlay import FPSSmoother, draw_boxes_and_labels, draw_fps
except ModuleNotFoundError:
    from yolo_utils import load_model, detect_people
    from posture_utils import detect_posture
    from debounce import PostureDebouncer
    from overlay import FPSSmoother, draw_boxes_and_labels, draw_fps


# ------------------ Defaults (hidden from UI) ------------------
YOLO_MODEL_PATH = "models/yolov8n.pt"
IMGSZ = 640
CONF = 0.30
IOU = 0.45
MAX_DET = 100


# ------------------ Helpers ------------------
_BACKENDS_ORDER = []
if platform.system() == "Darwin":  # macOS
    _BACKENDS_ORDER = [
        getattr(cv2, "CAP_AVFOUNDATION", None),
        None,  # "any"
        getattr(cv2, "CAP_FFMPEG", None),
        getattr(cv2, "CAP_GSTREAMER", None),
    ]
else:
    _BACKENDS_ORDER = [
        None,  # "any"
        getattr(cv2, "CAP_FFMPEG", None),
        getattr(cv2, "CAP_GSTREAMER", None),
    ]

def open_capture_auto(source: str, width: Optional[int] = None, height: Optional[int] = None):
    """Try a few backends automatically for robustness; return opened VideoCapture or None."""
    for api in _BACKENDS_ORDER:
        if source.isdigit():
            idx = int(source)
            cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
        else:
            cap = cv2.VideoCapture(source) if api is None else cv2.VideoCapture(source, api)

        if not cap or not cap.isOpened():
            if cap: cap.release()
            continue

        if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
    return None

def call_debouncer_compat(debouncer, boxes, raw_labels):
    """Compatibility shim for debouncer method naming differences."""
    if hasattr(debouncer, "process"):
        return debouncer.process(boxes, raw_labels)
    if hasattr(debouncer, "update"):
        return debouncer.update(boxes, raw_labels)
    if hasattr(debouncer, "__call__"):
        return debouncer(boxes, raw_labels)
    return raw_labels

def draw_panel(frame, lines, topleft=(8, 8), pad=6, line_h=24, alpha=0.35):
    """Draw a semi‑transparent info panel AFTER boxes so text isn't covered."""
    import cv2 as _cv2
    max_chars = max((len(s) for s in lines), default=0)
    width = max(160, max_chars * 10 + pad * 2)
    height = pad * 2 + line_h * len(lines)
    x, y = topleft
    x2, y2 = x + width, y + height
    overlay = frame.copy()
    _cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    _cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, text in enumerate(lines):
        ty = y + pad + line_h * (i + 1) - 6
        _cv2.putText(frame, text, (x + pad, ty), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, _cv2.LINE_AA)


# ------------------ Minimal Streamlit UI ------------------
st.set_page_config(page_title="People Counter + Posture", layout="wide")
st.title("People Counter + Posture Detection")

# Simple controls (no sidebar)
source_mode = st.radio("Select source", ["Webcam", "Video file"], horizontal=True)
if source_mode == "Webcam":
    source = st.number_input("Webcam index", min_value=0, max_value=8, value=0, step=1, help="0 = default camera")
else:
    source = st.text_input("Video file path", value="samples/office.mp4")

run = st.toggle("Start / Stop", value=False, key="run_toggle")

# Model load (hidden settings)
if "yolo_model" not in st.session_state:
    try:
        st.session_state["yolo_model"] = load_model(model_path=YOLO_MODEL_PATH, imgsz=IMGSZ, device=None)
    except Exception as e:
        st.error(f"Failed to load YOLO model from '{YOLO_MODEL_PATH}': {e}")
        st.stop()

if "debouncer" not in st.session_state:
    st.session_state["debouncer"] = PostureDebouncer(switch_frames=3, grid=48, max_miss=30)
if "fps_smoother" not in st.session_state:
    st.session_state["fps_smoother"] = FPSSmoother(alpha=0.12)

# Fixed placeholders so UI never stacks
metrics_row = st.container()
col1, col2, col3 = metrics_row.columns(3)
m_people_ph = col1.empty()
m_stand_ph  = col2.empty()
m_sit_ph    = col3.empty()

frame_holder = st.empty()
log_box = st.empty()


def run_stream():
    src_str = str(source) if isinstance(source, int) else source
    cap = open_capture_auto(src_str)
    if cap is None:
        st.error("Could not open the selected source. If this is macOS, make sure no other app is using the camera.")
        return

    model = st.session_state["yolo_model"]
    deb   = st.session_state["debouncer"]
    fps_sm = st.session_state["fps_smoother"]
    last_log = ""

    while st.session_state.get("run_toggle", False):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            last_log = "End of stream or read failure."
            break

        # Detect people
        try:
            boxes = detect_people(model, frame_bgr, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET)
        except Exception as e:
            boxes = []
            last_log = f"detect_people error: {e}"

        # Posture raw labels (always on in minimal UI)
        raw_labels: List[str] = []
        if boxes:
            for xyxy in boxes:
                try:
                    raw_labels.append(detect_posture(frame_bgr, xyxy))
                except Exception as e:
                    raw_labels.append("unknown")
                    last_log = f"detect_posture error: {e}"

        labels = call_debouncer_compat(deb, boxes, raw_labels) if boxes else raw_labels

        # Counts
        num_people = len(boxes)
        num_stand  = sum(1 for l in labels if l == "standing")
        num_sit    = sum(1 for l in labels if l == "sitting")

        # Draw detections then overlay compact panel
        draw_boxes_and_labels(frame_bgr, boxes, labels)
        fps = fps_sm.tick()
        draw_fps(frame_bgr, fps)
        draw_panel(frame_bgr, [f"People: {num_people}", f"Standing: {num_stand}", f"Sitting: {num_sit}"], topleft=(8, 8))

        # Render
        frame_holder.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        m_people_ph.metric("People", num_people)
        m_stand_ph.metric("Standing", num_stand)
        m_sit_ph.metric("Sitting", num_sit)

        if last_log:
            log_box.info(last_log)
        time.sleep(0.001)

    cap.release()


if run:
    run_stream()
else:
    frame_holder.info("Toggle **Start / Stop** to begin. Choose Webcam or Video file above.")
