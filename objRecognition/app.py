import cv2
from ultralytics import YOLO

# --- 1. CONFIGURATION ---
WEIGHTS_PATH = "yolov8n.pt"

# Initialize YOLO model (loads local weights)
try:
    model = YOLO(WEIGHTS_PATH)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the 'yolov8n.pt' file is in the same directory as this script.")
    exit()

# --- 2. CAMERA AND DETECTION LOOP SETUP ---


TARGET_CLASSES = [0, 67, 46]

# Drawing Configuration
COLORS = {
    0: (0, 255, 255), 
    67: (255, 0, 255),  
    46: (0, 255, 0)  
}
LABELS = {0: 'PERSON', 67: 'CELL PHONE', 46: 'SPOON'}

# Initialize camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera. Check camera permissions.")
    exit()

print("YOLOv8 Model initialized. Starting detection. Press 'q' to quit.")

# --- 3. DETECTION LOOP ---

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # --- Run YOLOv8 Detection ---
    results = model.track(
        frame,
        conf=0.15,  # *** Significantly lower confidence to detect faint/small objects ***
        classes=TARGET_CLASSES,
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

            label = LABELS.get(class_id, "UNKNOWN")
            color = COLORS.get(class_id, (255, 255, 255))

            confidence_text = f"{label} ({round(conf * 100)}%)"

            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Calculate text size for background
            (text_width, text_height), baseline = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

            # Position the text block above the box
            text_x = x1
            text_y = y1 - 10

            # Adjust text position if it runs off the top of the frame
            if text_y < text_height + 5:
                text_y = y1 + text_height + 15

            # Draw text background
            cv2.rectangle(frame, (text_x, text_y - text_height - 5), (text_x + text_width + 10, text_y + 5), color, -1)

            # Draw text
            cv2.putText(frame, confidence_text, (text_x + 5, text_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Display the result
    cv2.imshow("Real-Time Object Detection (YOLOv8)", frame)

    # Exit loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
