# 👥 People Counter and Posture Detector

This project uses YOLOv8 + MediaPipe to count people from live webcam feed and classify them as sitting or standing.

## ✅ Setup

```bash
python3 -m venv person-detect-env
source person-detect-env/bin/activate
pip install -r requirements.txt
```

Download the YOLOv8 nano model (free):

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

## 🚀 Run

- Live webcam:
```bash
python main.py
```

- With dashboard:
```bash
streamlit run webcam_demo.py
```

Press **q** to quit webcam view.
