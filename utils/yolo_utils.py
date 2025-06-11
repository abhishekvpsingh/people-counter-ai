from ultralytics import YOLO

def load_model(model_path="models/yolov8n.pt"):
    model = YOLO(model_path)
    return model

def detect_people(model, frame):
    results = model(frame)
    people = []
    for box in results[0].boxes.data:
        cls_id = int(box[5])
        if cls_id == 0:  # person class
            people.append(box[:4].cpu().numpy())  # x1, y1, x2, y2
    return people
