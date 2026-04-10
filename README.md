# Multi-Object Detection and Persistent ID Tracking in Sports Footage

A computer vision pipeline that detects all moving subjects in a public sports video and assigns each one a **unique, persistent ID** across every frame — even under occlusion, motion blur, scale changes, and camera movement.

Built with **YOLOv8** (detection) and **ByteTrack** (multi-object tracking).


## Overview

This project implements a full end-to-end pipeline for multi-object tracking in public sports event footage. The pipeline:

1. Downloads a publicly available sports video (YouTube or direct URL) using `yt-dlp`
2. Extracts frames using OpenCV at a configurable interval
3. Runs **YOLOv8** object detection on each sampled frame to locate all subjects (players, athletes, etc.)
4. Passes detections into **ByteTrack**, which assigns and maintains persistent unique IDs
5. Draws annotated bounding boxes, ID labels, and motion trails onto every frame
6. Assembles all annotated frames back into an output MP4 video
7. Optionally generates a movement heatmap and an object-count-over-time chart

The pipeline is designed to be modular — each stage (download, detect, track, annotate, enhance) lives in its own file and can be swapped or extended independently.

---

## Pipeline Architecture

```
Video Input (YouTube URL)
        │
        ▼
Frame Extraction (yt-dlp + OpenCV)
        │
        ▼
Object Detection (YOLOv8n — bounding boxes + confidence scores)
        │
        ▼
Multi-Object Tracking (ByteTrack — persistent ID assignment)
        │
        ▼
ID Manager (trail history + IoU matching across frames)
        │
        ▼
Frame Annotation (boxes, ID labels, motion trails drawn per frame)
        │
        ▼
Output Video (MP4 — annotated at original FPS)
        │
        ▼
Optional Enhancements (heatmap, count-over-time chart)
```

**Key design decision:** Detection and tracking are decoupled. `detector.py` handles pure inference; `tracker.py` handles state management across frames. This makes it easy to swap either component without touching the other.

---

## Project Structure

```
sports-tracker/
│
├── src/
│   ├── __init__.py
│   ├── downloader.py        # Download video via yt-dlp with optional duration trim
│   ├── detector.py          # YOLOv8 detection wrapper — returns Detection objects per frame
│   ├── tracker.py           # ByteTrack wrapper — returns TrackedObject list with trail history
│   ├── annotator.py         # Draws bounding boxes, ID labels, confidence, motion trails
│   ├── pipeline.py          # Orchestrates all stages end-to-end; collects stats
│   └── utils.py             # Helpers: colour-per-ID, resize, video property reader
│
├── optional/
│   ├── heatmap.py           # Generates a movement heatmap from accumulated bbox data
│   └── count_over_time.py   # Plots number of detected subjects per frame over time
│
├── notebooks/
│   └── demo.ipynb           # Interactive Jupyter walkthrough of the full pipeline
│
├── output/                  # Auto-created — all generated files land here
├── assets/                  # Sample screenshots for submission / README
│
├── config.yaml              # All tunable parameters in one place (model, thresholds, paths)
├── run.py                   # Main CLI entry point
├── requirements.txt         # All Python dependencies
├── report.md                # Short technical report (model choices, challenges, results)
└── README.md                # This file
```

### What each source file does

**`src/downloader.py`**
Uses `yt-dlp` to download a public video at up to 720p. Supports the `--download-sections` flag to grab only the first N seconds, which keeps test runs fast. Also exposes `use_local_video()` so you can skip downloading if you already have the file.

**`src/detector.py`**
Wraps `ultralytics.YOLO` to run inference on a single BGR frame. Returns a list of `Detection` dataclasses, each with a bounding box (`[x1, y1, x2, y2]`), confidence score, COCO class ID, and class name. Stateless — no frame-to-frame memory here.

**`src/tracker.py`**
Wraps YOLO's built-in ByteTrack integration (`model.track(..., persist=True)`). The `persist=True` flag is critical — it tells YOLO to maintain ByteTrack's internal state between calls so IDs remain consistent across frames. Also maintains a `trail_history` dict mapping each `track_id` to a rolling list of centre-point coordinates, which the annotator uses to draw motion trails.

**`src/annotator.py`**
Stateless drawing module. Given a frame and a list of `TrackedObject` instances, it draws a colour-coded bounding box and label for each object, plus a fading motion trail. Each track ID gets a deterministic colour via HSV hue rotation (`id_to_color` in utils.py), so the same player always appears in the same colour.

**`src/pipeline.py`**
Reads the input video frame by frame, calls the tracker every N frames (configurable via `frame_skip`), collects bounding boxes into a heatmap accumulator and count history, annotates each frame, and writes the result to an output MP4. Returns a stats dict consumed by `run.py` to trigger optional enhancements.

**`src/utils.py`**
Three small helpers: `id_to_color` (converts a track ID to a unique BGR colour), `resize_frame`, and `get_video_properties`.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- `ffmpeg` (required by yt-dlp for video trimming and merging)

Install `ffmpeg`:
```bash

# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html and add to PATH

```

### 1. Clone the repository

```bash
git clone <your-repo-link>
cd sports-tracker
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

The full `requirements.txt`:
```
ultralytics>=8.0.0
opencv-python>=4.8.0
yt-dlp>=2024.1.0
numpy>=1.24.0
PyYAML>=6.0
matplotlib>=3.7.0
scipy>=1.11.0
pandas>=2.0.0
jupyter>=1.0.0
ipykernel>=6.0.0
```

> **Note:** `ultralytics` includes ByteTrack internally — no separate tracker install is required. YOLOv8 model weights (`yolov8n.pt`, ~6 MB) are downloaded automatically on first run.

### 4. Verify the setup

```bash
python -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('Setup OK')"
```

You should see `Setup OK` after the model downloads.

---

## Configuration

All parameters are controlled from `config.yaml`. Edit this file before running — no code changes needed.

```yaml
# ── Video ──────────────────────────────────────────────────────────────────
video:
  url: "https://www.youtube.com/watch?v=YOUR_VIDEO_ID"  # public video URL
  output_path: "output/input_video.mp4"
  max_duration_sec: 120          # download only first 2 min (0 = full video)

```

## Running the Pipeline

### Option A — Download from YouTube

```bash
python run.py --url "https://www.youtube.com/watch?v=YOUR_VIDEO_ID" --config config.yaml
```

### Option B — Use a local video file

```bash
python run.py --local /path/to/your/match.mp4 --config config.yaml
```

### What happens when you run it

1. The video is downloaded (or validated if local) and saved to `output/input_video.mp4`
2. OpenCV reads the video frame by frame
3. Every 3rd frame (configurable) is passed to the YOLOv8 + ByteTrack pipeline
4. Each detected person gets a bounding box and a unique persistent ID
5. Annotated frames are written to `output/tracked_output.mp4` in real time
6. After all frames are processed, the heatmap and count chart are generated
7. A summary is printed to the console:

```
==================================================
  Total frames processed : 1800
  Unique IDs assigned    : 23
  Output video           : output/tracked_output.mp4
==================================================
```

### Running the Jupyter notebook

```bash
jupyter notebook notebooks/demo.ipynb
```

The notebook walks through each pipeline stage interactively and displays annotated frames inline.

---

## Output Files

| File | Description |
|---|---|
| `output/tracked_output.mp4` | Annotated video — bounding boxes, persistent IDs, motion trails |
| `output/heatmap.png` | Colour heatmap of where subjects spent the most time |
| `output/count_over_time.png` | Line chart of detected subject count per frame |
| `output/input_video.mp4` | The downloaded/original input video |

---