# utils/capture.py
from __future__ import annotations
import platform
import time
from typing import Union, Optional, List, Tuple
import cv2


_BACKENDS = {
    "any": None,
    "avfoundation": getattr(cv2, "CAP_AVFOUNDATION", None),
    "ffmpeg": getattr(cv2, "CAP_FFMPEG", None),
    "gstreamer": getattr(cv2, "CAP_GSTREAMER", None),
}

def _is_macos() -> bool:
    return platform.system().lower() == "darwin"

def _try_open(index_or_path: Union[int, str], backend: Optional[str]) -> cv2.VideoCapture:
    if backend is None or backend == "any":
        return cv2.VideoCapture(index_or_path)
    api = _BACKENDS.get(backend)
    if api is None:
        return cv2.VideoCapture(index_or_path)
    return cv2.VideoCapture(index_or_path, api)

def _warmup(cap: cv2.VideoCapture, seconds: float = 0.8) -> bool:
    t0 = time.time()
    while time.time() - t0 < seconds:
        ret, _ = cap.read()
        if ret:
            return True
        time.sleep(0.05)
    return False

def open_capture(
    source: Union[int, str],
    cam_w: int | None = None,
    cam_h: int | None = None,
    backend: str = "auto",  # "auto" | "any" | "avfoundation" | "ffmpeg" | "gstreamer"
    warmup_sec: float = 0.8,
) -> cv2.VideoCapture:
    """
    Open a camera index or video path with sensible defaults on macOS.
    Tries AVFoundation for cameras on mac by default, then falls back.
    Adds a short warm-up to let the camera initialize.
    """
    is_cam = isinstance(source, int)
    tried: List[str] = []

    # Build attempt list
    attempts: List[Optional[str]] = []
    if is_cam:
        if backend == "auto":
            attempts = ["avfoundation", "any"] if _is_macos() else ["any"]
        else:
            attempts = [backend, "any"]
    else:
        attempts = ["any"]

    for be in attempts:
        tried.append(be or "any")
        cap = _try_open(source, be)
        if not cap or not cap.isOpened():
            if cap:
                cap.release()
            continue

        # Apply camera size if it's a camera
        if is_cam:
            if cam_w:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
            if cam_h:
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

        # Warm-up
        if _warmup(cap, warmup_sec):
            return cap

        cap.release()

    tried_str = " -> ".join(tried)
    raise RuntimeError(f"Failed to open source={source} with backends [{tried_str}]")

def probe_cameras(max_index: int = 4, backends: Tuple[str, ...] = ("avfoundation", "any")) -> None:
    """
    Quick probe to see which indices/backends open on this system.
    """
    print("\n[PROBE] Camera indices/backends")
    for idx in range(max(1, max_index + 1)):
        for be in backends:
            api = _BACKENDS.get(be)
            cap = cv2.VideoCapture(idx) if api is None else cv2.VideoCapture(idx, api)
            ok = cap.isOpened()
            # Try one read to confirm
            if ok:
                ret, _ = cap.read()
                ok = bool(ret)
            status = "OK " if ok else "FAIL"
            print(f"  index={idx} backend={be:<12} -> {status}")
            cap.release()
    print("[PROBE] Done\n")
