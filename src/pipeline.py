"""
pipeline.py
End-to-end orchestrator: reads video → tracks → annotates → writes output.
Also collects data for optional enhancements (heatmap, count-over-time).
"""

import cv2
import os
import logging
import yaml
import numpy as np
from collections import defaultdict
from typing import Optional

from src.tracker import Tracker
from src.annotator import Annotator
from src.utils import get_video_properties, resize_frame

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_pipeline(
    video_path: str,
    config_path: str = "config.yaml",
    progress_callback=None,
) -> dict:
    """
    Main pipeline entry point.

    Args:
        video_path:        Path to the input video file.
        config_path:       Path to config.yaml.
        progress_callback: Optional callable(frame_idx, total_frames) for progress.

    Returns:
        stats dict with keys: total_frames, processed_frames, unique_ids,
                               heatmap_data, count_history
    """
    cfg = load_config(config_path)
    det_cfg = cfg["detection"]
    trk_cfg = cfg["tracking"]
    ann_cfg = cfg["annotation"]
    out_cfg = cfg["output"]
    enh_cfg = cfg["enhancements"]

    # ── Initialise components ──────────────────────────────────────────────
    tracker = Tracker(
        model_path=det_cfg["model"],
        confidence=det_cfg["confidence"],
        iou_threshold=det_cfg["iou_threshold"],
        classes=det_cfg["classes"],
        device=det_cfg["device"],
        trail_length=trk_cfg["trail_length"],
        tracker_config=trk_cfg["tracker_config"],
    )

    annotator = Annotator(
        box_thickness=ann_cfg["box_thickness"],
        font_scale=ann_cfg["font_scale"],
        show_confidence=ann_cfg["show_confidence"],
        show_trail=ann_cfg["show_trail"],
        trail_thickness=ann_cfg["trail_thickness"],
    )

    # ── Open video ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps, orig_w, orig_h, total_frames = get_video_properties(cap)
    logger.info(f"Video: {orig_w}x{orig_h} @ {fps:.1f}fps — {total_frames} frames")

    # Output resolution
    res = out_cfg["resolution"]
    out_w, out_h = (res[0], res[1]) if res else (orig_w, orig_h)
    out_fps = out_cfg.get("fps", fps)

    # ── Set up video writer ───────────────────────────────────────────────
    os.makedirs(os.path.dirname(out_cfg["annotated_video"]), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_cfg["annotated_video"], fourcc, out_fps, (out_w, out_h))

    # ── Data collectors for optional enhancements ─────────────────────────
    heatmap_accumulator = np.zeros((out_h, out_w), dtype=np.float32)
    count_history = []          # (frame_idx, count) tuples
    all_track_ids = set()

    frame_skip = trk_cfg["frame_skip"]
    frame_idx = 0
    last_tracked_objects = []   # reuse last result for skipped frames

    logger.info("Starting tracking pipeline...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Resize to output resolution
        if (orig_w, orig_h) != (out_w, out_h):
            frame = resize_frame(frame, out_w, out_h)

        # ── Detection + Tracking (every N-th frame) ───────────────────────
        if frame_idx % frame_skip == 0:
            tracked_objects = tracker.update(frame)
            last_tracked_objects = tracked_objects
        else:
            # On skipped frames, reuse previous tracks (boxes stay in place)
            tracked_objects = last_tracked_objects

        # ── Collect stats ─────────────────────────────────────────────────
        for obj in tracked_objects:
            all_track_ids.add(obj.track_id)
            # Accumulate heatmap: increment pixels under each bounding box
            if enh_cfg.get("heatmap"):
                x1, y1, x2, y2 = map(int, np.clip(obj.bbox, 0,
                    [out_w, out_h, out_w, out_h]))
                heatmap_accumulator[y1:y2, x1:x2] += 1.0

        count_history.append((frame_idx, len(tracked_objects)))

        # ── Annotate frame ────────────────────────────────────────────────
        annotated_frame = annotator.draw(frame, tracked_objects)

        # Overlay frame number
        cv2.putText(
            annotated_frame,
            f"Frame {frame_idx} | IDs: {len(tracked_objects)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        writer.write(annotated_frame)

        if progress_callback:
            progress_callback(frame_idx, total_frames)
        elif frame_idx % 50 == 0:
            logger.info(f"  Frame {frame_idx}/{total_frames} | "
                        f"Active tracks: {len(tracked_objects)} | "
                        f"Unique IDs seen: {len(all_track_ids)}")

    cap.release()
    writer.release()
    logger.info(f"Done. Output saved to: {out_cfg['annotated_video']}")
    logger.info(f"Unique track IDs assigned: {len(all_track_ids)}")

    return {
        "total_frames": frame_idx,
        "processed_frames": frame_idx // frame_skip,
        "unique_ids": len(all_track_ids),
        "heatmap_data": heatmap_accumulator,
        "count_history": count_history,
        "output_path": out_cfg["annotated_video"],
    }