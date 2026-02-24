import runpod
from ultralytics import YOLO
import base64
import numpy as np
import cv2

model = YOLO("yolov8n.pt")


def handler(event):
    input_data = event.get("input", {})
    image_b64 = input_data.get("image")

    if not image_b64:
        return {"error": "No image provided"}

    image_bytes = base64.b64decode(image_b64)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(img)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append({
                "class_id": cls_id,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

    return {"detections": detections}


runpod.serverless.start({"handler": handler})
