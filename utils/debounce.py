# utils/debounce.py
from __future__ import annotations
from typing import Tuple, List

class PostureDebouncer:
    """
    Debounces posture changes per person using a coarse position key (grid-quantized centroid).
    A posture switches only after N consecutive frames disagree with the current label.
    Tracks expire if not seen for `max_miss` frames.
    """
    def __init__(self, switch_frames: int = 3, grid: int = 48, max_miss: int = 30):
        self.switch_frames = max(1, int(switch_frames))
        self.grid = max(8, int(grid))
        self.max_miss = max(1, int(max_miss))
        self._tracks = {}  # key -> {"label": str|None, "streak": int, "age": int}

    def _key_for_bbox(self, bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        return (cx // self.grid, cy // self.grid)

    def update(self, boxes: List[Tuple[int, int, int, int]], raw_labels: List[str]) -> List[str]:
        out_labels: List[str] = []
        seen_keys = set()

        for bbox, raw in zip(boxes, raw_labels):
            k = self._key_for_bbox(bbox)
            state = self._tracks.get(k, {"label": None, "streak": 0, "age": 0})

            if raw != "unknown":
                if state["label"] is None:
                    state["label"] = raw
                    state["streak"] = 0
                elif raw == state["label"]:
                    state["streak"] = 0
                else:
                    state["streak"] += 1
                    if state["streak"] >= self.switch_frames:
                        state["label"] = raw
                        state["streak"] = 0

            state["age"] = 0
            self._tracks[k] = state
            seen_keys.add(k)
            out_labels.append(state["label"] or raw)

        for k in list(self._tracks.keys()):
            if k not in seen_keys:
                self._tracks[k]["age"] += 1
                if self._tracks[k]["age"] > self.max_miss:
                    del self._tracks[k]

        return out_labels
