
import torch
import cv2

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Define vehicle classes that we are interested in
vehicle_classes = ['car', 'bus', 'truck', 'motorcycle']

# Start webcam
cap = cv2.VideoCapture(0)  # 0 means default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)

    # Parse results
    df = results.pandas().xyxy[0]
    
    # Filter only vehicle classes
    vehicle_count = sum(df['name'].isin(vehicle_classes))

    # Render the results (bounding boxes)
    annotated_frame = results.render()[0].copy()


    # Show vehicle count on the frame
    cv2.putText(annotated_frame, f'Vehicle Count: {vehicle_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Live Traffic Detection", annotated_frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
