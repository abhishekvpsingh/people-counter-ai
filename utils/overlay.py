# utils/overlay.py
from __future__ import annotations
from typing import List, Tuple
import cv2
from time import perf_counter

class FPSSmoother:
    def __init__(self, alpha: float = 0.12):
        self.alpha = alpha
        self._ema = None
        self._t_prev = perf_counter()

    def tick(self) -> float:
        now = perf_counter()
        fps_inst = 1.0 / max(1e-6, (now - self._t_prev))
        self._t_prev = now
        if self._ema is None:
            self._ema = fps_inst
        else:
            self._ema = (1 - self.alpha) * self._ema + self.alpha * fps_inst
        return self._ema

def draw_boxes_and_labels(frame, boxes: List[Tuple[int,int,int,int]], labels: List[str]):
    for (x1, y1, x2, y2), label in zip(boxes, labels):
        if label == "standing":
            color = (0, 255, 0)
        elif label == "sitting":
            color = (0, 165, 255)
        else:
            color = (128, 128, 128)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2)
