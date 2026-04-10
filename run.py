"""
run.py — main entry point
Usage:
  python run.py --url "https://youtube.com/watch?v=..." --config config.yaml
  python run.py --local path/to/video.mp4 --config config.yaml
"""

import argparse
import logging
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

from src.downloader import download_video, use_local_video
from src.pipeline import run_pipeline, load_config
from optional.heatmap import save_heatmap
from optional.count_over_time import plot_count_over_time


def main():
    parser = argparse.ArgumentParser(description="Sports Multi-Object Tracker")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", type=str, help="Public video URL to download")
    group.add_argument("--local", type=str, help="Path to a local video file")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Step 1: Get video ─────────────────────────────────────────────────
    if args.url:
        video_path = download_video(
            url=args.url,
            output_path=cfg["video"]["output_path"],
            max_duration_sec=cfg["video"]["max_duration_sec"],
        )
    else:
        video_path = use_local_video(args.local)

    # ── Step 2: Run tracking pipeline ────────────────────────────────────
    stats = run_pipeline(video_path, config_path=args.config)

    print("\n" + "="*50)
    print(f"  Total frames processed : {stats['total_frames']}")
    print(f"  Unique IDs assigned    : {stats['unique_ids']}")
    print(f"  Output video           : {stats['output_path']}")
    print("="*50)

    # ── Step 3: Optional enhancements ────────────────────────────────────
    enh_cfg = cfg["enhancements"]

    if enh_cfg.get("heatmap") and stats["heatmap_data"] is not None:
        save_heatmap(
            heatmap_data=stats["heatmap_data"],
            output_path=enh_cfg["heatmap_path"],
        )

    if enh_cfg.get("count_plot") and stats["count_history"]:
        plot_count_over_time(
            count_history=stats["count_history"],
            output_path=enh_cfg["count_plot_path"],
        )


if __name__ == "__main__":
    main()