from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import os

app = FastAPI()

model = YOLO("yolov8n.pt")

@app.get("/ping")
async def ping():
    return {"status": "healthy"}

@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    image_bytes = await file.read()
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "80"))
    uvicorn.run(app, host="0.0.0.0", port=port)
