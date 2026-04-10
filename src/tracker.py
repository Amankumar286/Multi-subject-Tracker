"""
tracker.py
Multi-object tracker using YOLOv8's built-in ByteTrack integration.
Manages persistent IDs across frames including during occlusion.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Represents one tracked subject across multiple frames."""
    track_id: int
    bbox: np.ndarray            # [x1, y1, x2, y2] current frame
    confidence: float
    class_name: str
    trail: List[Tuple[int, int]] = field(default_factory=list)   # (cx, cy) history


class Tracker:
    """
    Wraps YOLO's built-in ByteTrack tracker.

    ByteTrack works in two stages:
      1. High-confidence detections are matched to existing tracks via IoU.
      2. Unmatched tracks are re-checked against LOW-confidence detections,
         recovering objects that briefly become hard to detect (occlusion, blur).
    This two-stage matching is why ByteTrack handles crowded sports scenes well.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
        trail_length: int = 40,
        tracker_config: str = "bytetrack.yaml",
    ):
        self.trail_length = trail_length
        self.device = device
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.tracker_config = tracker_config

        # trail_history[track_id] = list of (cx, cy) tuples
        self.trail_history: Dict[int, List[Tuple[int, int]]] = defaultdict(list)

        logger.info(f"Loading tracker model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(device)

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Run detection + tracking on one frame.

        Args:
            frame: BGR numpy array from OpenCV.

        Returns:
            List of TrackedObject for all active tracks in this frame.
        """
        results = self.model.track(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            tracker=self.tracker_config,
            device=self.device,
            persist=True,      # CRITICAL: keeps track state between calls
            verbose=False,
        )

        tracked_objects = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box in result.boxes:
                track_id = int(box.id[0].cpu())
                bbox = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cls_name = self.model.names[cls_id]

                # Update trail (centre point of bounding box)
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                self.trail_history[track_id].append((cx, cy))

                # Keep only the last `trail_length` positions
                if len(self.trail_history[track_id]) > self.trail_length:
                    self.trail_history[track_id].pop(0)

                tracked_objects.append(TrackedObject(
                    track_id=track_id,
                    bbox=bbox,
                    confidence=conf,
                    class_name=cls_name,
                    trail=list(self.trail_history[track_id]),
                ))

        return tracked_objects

    def reset(self):
        """Clear all track history (call between unrelated video segments)."""
        self.trail_history.clear()
        # Reset internal YOLO tracker state
        self.model.predictor = None