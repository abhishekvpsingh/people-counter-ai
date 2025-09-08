# People Counter + Posture (YOLOv8 + MediaPipe)

A small project that detects people in a video stream and classifies posture (standing vs sitting). It shows live counts on the video and simple metrics in the UI. You can run it either as a minimal Streamlit app or as a plain Python CLI.

---

## What it does

- Detects people using YOLOv8.
- Classifies posture for each person (standing or sitting) using MediaPipe.
- Debounces posture labels so they do not flicker.
- Overlays clean, readable counts on the video: People, Standing, Sitting.
- Shows FPS on the video.
- Two ways to run:
  - **Streamlit app** (`webcam_demo.py`) with UI.
  - **CLI** (`main.py`) with command‑line options.

---

## Requirements

- Python 3.10 or later
- macOS, Windows, or Linux
- A webcam if you want to use live camera input
- `requirements.txt` in this repo installs all needed Python packages (OpenCV, Streamlit, Ultralytics/YOLO, MediaPipe, etc.)

> Tip for macOS: the default OpenCV backend for the camera is `avfoundation`. If your camera does not open, see the troubleshooting section below.

---

## Setup

### 1) Clone the project
```bash
git clone <your-repo-url> people-counter-ai
cd people-counter-ai
```

### 2) Create and activate a virtual environment

**Option A: venv (built‑in)**
```bash
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate  # Windows PowerShell
```

**Option B: conda**
```bash
conda create -n pycv python=3.11 -y
conda activate pycv
```

### 3) Install libraries
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) YOLO model weights
By default the apps use `models/yolov8n.pt`. If that file does not exist, Ultralytics will usually download it the first time. If you prefer to place it yourself:

```bash
mkdir -p models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
curl -L -o models/pose_landmarker_full.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task
```

If you want to use a different YOLOv8 model file, you can change the model path in the CLI options or in the code.

---

## How to run

### Option 1: Minimal Streamlit app (recommended for quick testing)
Start the app:
```bash
streamlit run webcam_demo.py
```

In the app:
- Select **Webcam** and choose the camera index (0 is usually the built‑in camera), or select **Video file** and enter a path.
- Toggle **Start / Stop**.
- The overlay will show **People / Standing / Sitting** and FPS. Metrics update at the top of the page.

### Option 2: Command‑line (no UI)
Run with a webcam:
```bash
python main.py --source 0 --backend avfoundation   # macOS example
```
Run with a recorded video:
```bash
python main.py --source samples/office.mp4
```

Useful flags:
- `--backend` one of: `any`, `avfoundation` (macOS), `ffmpeg`, `gstreamer`
- `--width` and `--height` to hint camera resolution (e.g., 1280x720)
- `--model` to point to a different YOLO file
- `--imgsz` (default 640), `--conf` (default 0.30), `--iou` (default 0.45), `--max-det` (default 100)
- `--no-posture` to disable posture classification
- Debounce knobs: `--switch-frames 3`, `--grid 48`, `--max-miss 30`

Probe cameras and backends:
```bash
python main.py --probe --probe-max-index 3
```

Quit the CLI window with the `q` key.

---

## Project layout (key files)

```
.
├── main.py
├── webcam_demo.py
├── requirements.txt
├── README.md
├── models/
│   ├── yolov8n.pt                 # YOLO weights (optional, auto-download if missing)
│   └── pose_landmarker_full.task  # MediaPipe Pose task file (required for posture)
└── utils/
    ├── capture.py                 # (optional) capture helpers
    ├── debounce.py                # Debouncer to stabilize posture labels
    ├── overlay.py                 # Drawing helpers (FPS, boxes, labels)
    ├── posture_utils.py           # Posture classification helpers (MediaPipe)
    └── yolo_utils.py              # YOLO model loading + person detection

```

---

## Troubleshooting

**The webcam does not open**
- Try a different index: 0, then 1 or 2.
- Close any app that might be using the camera (Zoom, Teams, FaceTime, browser).
- On macOS, pass `--backend avfoundation` when using `main.py`. For Streamlit, just try another index; the app auto‑tries common backends.
- Check camera permissions for your terminal or IDE (System Settings → Privacy & Security → Camera).

**The video is choppy or slow**
- Use a smaller `--imgsz` value (for example 416 or 320) in `main.py`.
- Prefer the `yolov8n` model (default). Larger models are slower on CPU.
- Close other CPU‑heavy apps.

**Posture seems incorrect sometimes**
- The MediaPipe signal may degrade at extreme angles or when people are partially out of frame.
- The debouncer reduces flicker; tune `--switch-frames` and `--max-miss` in `main.py` if needed.


---

## License

For personal or educational use. If you plan to use this commercially, review the licenses of YOLO/Ultralytics and MediaPipe.
