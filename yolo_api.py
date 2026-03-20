from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import numpy as np
import cv2
import uvicorn

app = FastAPI()

# CORS config
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = YOLO("yolov8n.pt")  


vehicle_classes = ['car', 'bus', 'truck', 'motorbike']

@app.post("/detect-vehicles")
async def detect_vehicle(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    results = model(frame)[0]
    vehicle_count = 0

    for result in results.boxes.cls:
        cls_name = model.names[int(result)]
        if cls_name in vehicle_classes:
            vehicle_count += 1

    return {"vehicle_count": vehicle_count}


@app.post("/detect-video")
async def detect_from_video(file: UploadFile = File(...)):
    import requests

    contents = await file.read()
    video_path = "temp_video.mp4"
    with open(video_path, 'wb') as f:
        f.write(contents)

    cap = cv2.VideoCapture(video_path)

    timer_started = False
    signal_timer = 0
    vehicle_count = 0
    start_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Only run this ONCE at start
        if not timer_started:
            results = model(frame)[0]
            vehicle_count = 0

            for cls in results.boxes.cls:
                class_name = model.names[int(cls)]
                if class_name in vehicle_classes:
                    vehicle_count += 1

            # Send to Spring Boot to calculate timer
            response = requests.post("http://localhost:8080/api/traffic/status",
                                     json={"vehicleCount": vehicle_count})
            signal_timer = response.json().get("signalTime", 0)
            start_time = cv2.getTickCount()
            timer_started = True

        # Time passed since start
        elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        time_left = max(signal_timer - int(elapsed_time), 0)

        # Run YOLO on current frame for visualization (optional, not used for logic)
        results = model(frame)[0]
        annotated = results.plot()

        # Display vehicle count and signal timer
        cv2.putText(annotated, f"Vehicles Detected: {vehicle_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.putText(annotated, f"Signal Timer: {time_left}s", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Signal Detection - Video", annotated)

        # Exit when timer ends
        if time_left <= 0 or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Detection complete."}



@app.get("/detect-webcam")
def detect_from_webcam():
    import requests

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return {"error": "Could not open webcam"}

    # Initialize with default values
    vehicle_count = 0
    signal_timer = 15  # Minimum default timer
    timer_started = False
    start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if not timer_started:
            # Initial detection
            results = model(frame)[0]
            vehicle_count = sum(
                model.names[int(cls)] in vehicle_classes 
                for cls in results.boxes.cls
            )

            # Get timer from backend
            try:
                response = requests.post(
                    "http://localhost:8080/api/traffic/status",
                    json={"vehicleCount": vehicle_count},
                    timeout=2
                )
                signal_timer = max(response.json().get("signalTime", 15), 15)
            except:
                signal_timer = 15  # Fallback timer

            start_time = cv2.getTickCount()
            timer_started = True

        # Calculate remaining time
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        time_left = max(signal_timer - int(elapsed), 0)

        # Visualize detection
        results = model(frame)[0]
        annotated = results.plot()
        
        # Display info
        cv2.putText(annotated, f"Vehicles: {vehicle_count}", (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)
        cv2.putText(annotated, f"Timer: {time_left}s", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 2)

        cv2.imshow("Webcam Detection", annotated)

        # Exit conditions
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if time_left <= 0:
            # Reset for next cycle
            timer_started = False

    cap.release()
    cv2.destroyAllWindows()
    return {"message": "Webcam detection complete."}


@app.post("/detect-vehicle-from-video")
async def detect_vehicle_from_video(file: UploadFile = File(...)):
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    os.remove(video_path)

    if not ret:
        return {"vehicle_count": 0}

    results = model(frame)[0]
    vehicle_count = sum(
        model.names[int(cls)] in vehicle_classes for cls in results.boxes.cls
    )
    return {"vehicle_count": vehicle_count}



import os
port = int(os.environ.get("PORT", 8000))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
