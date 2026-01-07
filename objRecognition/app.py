import cv2
import numpy as np
from ultralytics import YOLO

WEIGHTS_PATH = "yolov8n.pt"

try:
    model = YOLO(WEIGHTS_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the 'yolov8n.pt' file is in the same directory as this script.")
    exit()


# We generate a unique random color for each class the model can detect (80 classes for COCO)
# We set a seed so the colors remain the same every time you run the app
np.random.seed(42)
colors = np.random.uniform(0, 255, size=(len(model.names), 3))

#camera setup

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera. Check camera permissions.")
    exit()

print(f"YOLOv8 Model initialized. Detecting {len(model.names)} classes.")
print("Press 'q' to quit.")

#Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a "mirror" effect
    frame = cv2.flip(frame, 1)

    results = model.track(
        frame,
        conf=0.25,  # Standard confidence threshold (adjust lower if missing objects)
        persist=True,
        verbose=False,
        imgsz=480
    )

    for result in results:
        boxes = result.boxes

        for box in boxes:
            # Get coordinates, class ID, and confidence
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            # Get the correct label name directly from the model
            label = model.names[class_id].upper()

            # Get the specific color for this class
            color = colors[class_id]
            # OpenCV uses BGR, and our random colors are float, so we convert them to int tuple
            bgr_color = (int(color[0]), int(color[1]), int(color[2]))

            confidence_text = f"{label} ({round(conf * 100)}%)"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), bgr_color, 2)

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Position the text block above the box
            text_x = x1
            text_y = y1 - 10

            # Adjust text position if it runs off the top of the frame
            if text_y < text_height + 5:
                text_y = y1 + text_height + 15

            # Draw text background
            cv2.rectangle(frame, (text_x, text_y - text_height - 5), (text_x + text_width + 10, text_y + 5), bgr_color,
                          -1)

            # Draw text (Black text provides best contrast on bright random colors)
            cv2.putText(frame, confidence_text, (text_x + 5, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the result
    cv2.imshow("Full Detection (YOLOv8)", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
