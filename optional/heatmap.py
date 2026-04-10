"""
heatmap.py
Generates a movement heatmap from accumulated bounding box data.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_heatmap(
    heatmap_data: np.ndarray,
    output_path: str,
    background_frame: np.ndarray = None,
    alpha: float = 0.6,
):
    """
    Saves a heatmap PNG. Optionally blends over a background frame.

    Args:
        heatmap_data:     2D float array of accumulated presence (from pipeline).
        output_path:      Where to save the PNG.
        background_frame: Optional BGR frame to overlay the heatmap onto.
        alpha:            Opacity of heatmap over background.
    """
    # Normalise to 0-255
    norm = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX)
    norm = norm.astype(np.uint8)

    # Apply JET colormap
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

    if background_frame is not None:
        # Resize background to match heatmap dimensions
        bg = cv2.resize(background_frame, (colored.shape[1], colored.shape[0]))
        blended = cv2.addWeighted(bg, 1 - alpha, colored, alpha, 0)
        cv2.imwrite(output_path, blended)
    else:
        cv2.imwrite(output_path, colored)

    print(f"Heatmap saved to: {output_path}")