"""
detector.py
YOLOv8-based object detector.
Returns standardised Detection objects (bbox, confidence, class_id) per frame.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from ultralytics import YOLO
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Single detection result for one object in one frame."""
    bbox: np.ndarray        # [x1, y1, x2, y2] in pixel coords
    confidence: float
    class_id: int
    class_name: str


class Detector:
    """
    Wraps YOLOv8 for person (and optionally object) detection.
    Designed to be initialised once and called per frame.
    """

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.35,
        iou_threshold: float = 0.45,
        classes: Optional[List[int]] = None,
        device: str = "cpu",
    ):
        """
        Args:
            model_path:    YOLOv8 weights. First call auto-downloads from ultralytics.
            confidence:    Minimum detection confidence (0-1).
            iou_threshold: NMS IoU threshold.
            classes:       Filter to specific COCO class IDs. None = all classes.
                           Use [0] for persons only.
            device:        'cpu' or 'cuda:0'
        """
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device

        logger.info(f"Loading YOLO model: {model_path} on {device}")
        self.model = YOLO(model_path)
        self.model.to(device)
        logger.info("Model loaded.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run detection on a single BGR frame (OpenCV format).

        Args:
            frame: H x W x 3 numpy array in BGR colour space.

        Returns:
            List of Detection objects. Empty list if nothing detected.
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            classes=self.classes,
            device=self.device,
            verbose=False,   # suppress per-frame console spam
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                bbox = box.xyxy[0].cpu().numpy()    # [x1, y1, x2, y2]
                conf = float(box.conf[0].cpu())
                cls_id = int(box.cls[0].cpu())
                cls_name = self.model.names[cls_id]

                detections.append(Detection(
                    bbox=bbox,
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        return detections