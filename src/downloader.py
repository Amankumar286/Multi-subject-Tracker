"""
downloader.py
Downloads a public video (YouTube or direct URL) using yt-dlp.
Supports duration trimming so we don't download hour-long matches.
"""

import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def download_video(url: str, output_path: str, max_duration_sec: int = 120) -> str:
    """
    Download a video from a public URL using yt-dlp.

    Args:
        url:              Public video URL (YouTube, etc.)
        output_path:      Where to save the downloaded file (e.g. 'output/input_video.mp4')
        max_duration_sec: Clip to this many seconds. Set to 0 for full video.

    Returns:
        Absolute path to the downloaded file.

    Raises:
        RuntimeError: If yt-dlp is not installed or download fails.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Build yt-dlp command
    cmd = [
        "yt-dlp",
        "--format", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/best",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--no-playlist",
    ]

    # Trim to max_duration_sec using postprocessor args (requires ffmpeg)
    if max_duration_sec and max_duration_sec > 0:
        cmd += ["--download-sections", f"*0-{max_duration_sec}"]

    cmd.append(url)

    logger.info(f"Downloading: {url}")
    logger.info(f"Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed with return code {result.returncode}. "
            "Make sure yt-dlp and ffmpeg are installed."
        )

    abs_path = os.path.abspath(output_path)
    logger.info(f"Downloaded to: {abs_path}")
    return abs_path


def use_local_video(local_path: str) -> str:
    """
    Use an already-downloaded local video instead of downloading.
    Validates the file exists and returns its absolute path.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local video not found: {local_path}")
    return os.path.abspath(local_path)