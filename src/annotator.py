"""
annotator.py
Draws bounding boxes, track IDs, confidence scores, and motion trails
onto video frames. Stateless — operates purely on the frame + track data.
"""

import cv2
import numpy as np
from typing import List
from src.tracker import TrackedObject
from src.utils import id_to_color


class Annotator:
    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.6,
        show_confidence: bool = True,
        show_trail: bool = True,
        trail_thickness: int = 2,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.show_confidence = show_confidence
        self.show_trail = show_trail
        self.trail_thickness = trail_thickness

    def draw(self, frame: np.ndarray, tracked_objects: List[TrackedObject]) -> np.ndarray:
        """
        Draw all tracked objects onto the frame.
        Returns a new annotated copy (does not modify input in-place).
        """
        annotated = frame.copy()

        for obj in tracked_objects:
            color = id_to_color(obj.track_id)
            x1, y1, x2, y2 = map(int, obj.bbox)

            # --- Bounding box ---
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            # --- Label: "ID:5  0.82" ---
            label = f"ID:{obj.track_id}"
            if self.show_confidence:
                label += f"  {obj.confidence:.2f}"

            label_size, baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, 1
            )
            # filled rectangle behind label for readability
            cv2.rectangle(
                annotated,
                (x1, y1 - label_size[1] - baseline - 4),
                (x1 + label_size[0] + 4, y1),
                color,
                thickness=-1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 2, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            # --- Motion trail ---
            if self.show_trail and len(obj.trail) > 1:
                pts = np.array(obj.trail, dtype=np.int32)
                # fade older points by drawing thinner lines for older segments
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    thickness = max(1, int(self.trail_thickness * alpha))
                    cv2.line(annotated, tuple(pts[i - 1]), tuple(pts[i]), color, thickness)

        # Frame counter (top-left corner)
        return annotated