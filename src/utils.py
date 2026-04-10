"""
utils.py
Small helper utilities used across the pipeline.
"""

import cv2
import numpy as np
from typing import Tuple


# Generates a visually distinct BGR colour for each track ID.
# Uses HSV colour space so IDs far apart in number still look different.
def id_to_color(track_id: int) -> Tuple[int, int, int]:
    hue = (track_id * 37) % 180      # spread IDs across the hue wheel
    color_hsv = np.uint8([[[hue, 220, 220]]])
    color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2])


def resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(frame, (width, height))


def get_video_properties(cap: cv2.VideoCapture):
    """Return (fps, width, height, total_frames) from an open VideoCapture."""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return fps, width, height, total