"""
count_over_time.py
Plots number of detected subjects per frame over the video duration.
"""

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
from typing import List, Tuple


def plot_count_over_time(
    count_history: List[Tuple[int, int]],
    output_path: str,
    fps: float = 30.0,
):
    """
    Args:
        count_history: List of (frame_idx, count) tuples from the pipeline.
        output_path:   Where to save the PNG.
        fps:           Video FPS (used to convert frames → seconds on x-axis).
    """
    frames, counts = zip(*count_history)
    times = [f / fps for f in frames]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(times, counts, linewidth=1.2, color="#4A90D9", alpha=0.85)
    ax.fill_between(times, counts, alpha=0.15, color="#4A90D9")
    ax.set_xlabel("Time (seconds)", fontsize=11)
    ax.set_ylabel("Detected subjects", fontsize=11)
    ax.set_title("Number of detected subjects over time", fontsize=13)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Count plot saved to: {output_path}")