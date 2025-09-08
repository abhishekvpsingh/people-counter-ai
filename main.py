#!/usr/bin/env python3
"""
CLI app for People + Posture detection (feature parity with minimal webcam_demo).

Features:
- Source: webcam index, video file, or network URL (RTSP/HTTP)
- Backend selection (auto, avfoundation, ffmpeg, gstreamer)
- Optional camera width/height hints
- YOLOv8 params: model path, imgsz, conf, iou, max_det
- Posture classification + debouncer (compatible with process/update/__call__)
- FPS smoothing + overlay
- On‑frame semi‑transparent panel showing People / Standing / Sitting
- Keyboard: press 'q' or 'Q' or ESC to quit; closing the window also exits
- Probe utility: --probe (tests indices/backends quickly)
"""

import argparse
from typing import List, Tuple, Optional

import cv2

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


_BACKENDS = {
    "any": None,
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
}


def open_capture(source: str, backend_key: str = "any",
                 width: Optional[int] = None, height: Optional[int] = None):
    """Open a cv2.VideoCapture with optional backend and resolution hints."""
    api = _BACKENDS.get(backend_key, None)

    if source.isdigit():
        idx = int(source)
        cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
    else:
        cap = cv2.VideoCapture(source) if api is None else cv2.VideoCapture(source, api)

    if not cap or not cap.isOpened():
        return None

    if width:  cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    if height: cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

    ok, _ = cap.read()
    if not ok:
        cap.release()
        return None
    return cap


def probe_cameras(max_index: int = 3, backends: Optional[List[str]] = None) -> List[Tuple[int, str, bool]]:
    """Try opening camera indices with various backends; return (index, backend, ok)."""
    if backends is None:
        backends = ["avfoundation", "any", "ffmpeg", "gstreamer"]
    out = []
    for idx in range(0, max(1, max_index + 1)):
        for be in backends:
            api = _BACKENDS.get(be, None)
            cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
            ok = bool(cap.isOpened())
            if ok:
                ok, _ = cap.read()
            out.append((idx, be, bool(ok)))
            cap.release()
    return out


def call_debouncer_compat(debouncer, boxes, raw_labels):
    """Compatibility shim: process/update/__call__."""
    if hasattr(debouncer, "process"):
        return debouncer.process(boxes, raw_labels)
    if hasattr(debouncer, "update"):
        return debouncer.update(boxes, raw_labels)
    if hasattr(debouncer, "__call__"):
        return debouncer(boxes, raw_labels)
    return raw_labels


def draw_panel(frame, lines, topleft=(8, 8), pad=6, line_h=24, alpha=0.35):
    """Draw a semi-transparent info panel AFTER boxes so text isn't covered."""
    max_chars = max((len(s) for s in lines), default=0)
    width = max(160, max_chars * 10 + pad * 2)
    height = pad * 2 + line_h * len(lines)
    x, y = topleft
    x2, y2 = x + width, y + height
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    for i, text in enumerate(lines):
        ty = y + pad + line_h * (i + 1) - 6
        cv2.putText(frame, text, (x + pad, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="People + Posture detection (CLI)")
    p.add_argument("--source", type=str, default="0",
                   help="Webcam index (e.g., '0'), video path, or URL")
    p.add_argument("--backend", type=str, default="any", choices=list(_BACKENDS.keys()),
                   help="OpenCV backend. macOS: try 'avfoundation'.")
    p.add_argument("--width", type=int, default=0, help="Camera width hint")
    p.add_argument("--height", type=int, default=0, help="Camera height hint")

    # YOLO settings
    p.add_argument("--model", type=str, default="models/yolov8n.pt", help="YOLO model path")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    p.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    p.add_argument("--max-det", type=int, default=100, help="Max detections")

    # Posture / debounce
    p.add_argument("--no-posture", action="store_true", help="Disable posture classification")
    p.add_argument("--switch-frames", type=int, default=3, help="Debounce frames to switch label")
    p.add_argument("--grid", type=int, default=48, help="Debounce grid size")
    p.add_argument("--max-miss", type=int, default=30, help="Debounce track expiry in frames")

    # Probe
    p.add_argument("--probe", action="store_true", help="Probe camera indices/backends and exit")
    p.add_argument("--probe-max-index", type=int, default=3, help="Max index to probe")

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.probe:
        results = probe_cameras(args.probe_max_index)
        print("== Camera probe results ==")
        for idx, be, ok in results:
            print(f"index={idx:>2} backend={be:<12} -> {'OK' if ok else 'FAIL'}")
        return 0

    # Load model
    try:
        model = load_model(model_path=args.model, imgsz=args.imgsz, device=None)
    except Exception as e:
        print(f"[ERROR] Failed to load YOLO model: {e}")
        return 2

    # Debouncer + FPS smoother
    debouncer = PostureDebouncer(switch_frames=int(args.switch_frames),
                                 grid=int(args.grid),
                                 max_miss=int(args.max_miss))
    fps_sm = FPSSmoother(alpha=0.12)

    # Open capture
    width = args.width if args.width > 0 else None
    height = args.height if args.height > 0 else None
    cap = open_capture(args.source, backend_key=args.backend, width=width, height=height)
    if cap is None:
        print(f"[ERROR] Failed to open source={args.source} with backends [{args.backend} -> any]")
        print("Tips: try a different camera index (0/1/2), ensure no other app uses the camera, "
              "or set --backend avfoundation on macOS.")
        return 3

    window_name = "People Counter + Posture  —  Press Q/ESC to quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # If the user clicked the window's close button, exit cleanly
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("[WARN] End of stream or read failure.")
                break

            # Detect people
            try:
                boxes = detect_people(model, frame_bgr, imgsz=args.imgsz, conf=args.conf,
                                      iou=args.iou, max_det=int(args.max_det))
            except Exception as e:
                boxes = []
                # Optionally print once; keep loop going
                # print(f"[WARN] detect_people error: {e}")

            # Posture
            if not args.no_posture and boxes:
                raw_labels: List[str] = []
                for xyxy in boxes:
                    try:
                        raw_labels.append(detect_posture(frame_bgr, xyxy))
                    except Exception:
                        raw_labels.append("unknown")
            else:
                raw_labels = ["person"] * len(boxes)

            # Debounce
            labels = call_debouncer_compat(debouncer, boxes, raw_labels) if not args.no_posture else raw_labels

            # Counts
            num_people = len(boxes)
            num_stand = sum(1 for l in labels if l == "standing")
            num_sit = sum(1 for l in labels if l == "sitting")

            # Draw
            draw_boxes_and_labels(frame_bgr, boxes, labels)
            fps = fps_sm.tick()
            draw_fps(frame_bgr, fps)
            draw_panel(frame_bgr, [
                f"People:   {num_people}",
                f"Standing: {num_stand}",
                f"Sitting:  {num_sit}",
            ], topleft=(8, 8))

            cv2.imshow(window_name, frame_bgr)

            # Handle quit keys robustly (q / Q / ESC)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 27 = ESC
                break

    except KeyboardInterrupt:
        # Allow Ctrl+C to exit cleanly from terminal
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
