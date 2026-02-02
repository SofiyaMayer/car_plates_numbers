import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sort.sort import Sort
from datetime import datetime
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = 'license_plate_ncnn_model'  # Your optimized Pi 5 model folder
LOG_FILE = "detected_plates.txt"
FRAME_SKIP = 5              # Process inference every 5th frame
OCR_CONF_THRESH = 0.5       # Minimum confidence to accept a plate
TARGET_PLATE = "EL924CF"

# --- 2. ID MAPPING LOGIC ---
# Maps messy tracker IDs to clean, sequential numbers
id_map = {}
next_human_id = 1

def get_human_id(tracker_id):
    global next_human_id
    if tracker_id not in id_map:
        id_map[tracker_id] = next_human_id
        next_human_id += 1
    return id_map[tracker_id]

# --- 3. INITIALIZATION ---
# Load NCNN model (Task must be 'detect' for NCNN)
model = YOLO(MODEL_PATH, task='detect')

# Initialize PaddleOCR (PP-OCRv4 mobile)
ocr_reader = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)

# Initialize SORT tracker with higher max_age to prevent ID jumps
tracker = Sort(max_age=1, min_hits=2, iou_threshold=0.5)

# Camera Setup
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Data Storage
stable_plates = {}    # {human_id: {"text": "ABC", "count": 5}}
finalized_plates = set() 
last_tracked_objects = []
frame_count = 0

def log_plate(plate_text, human_id):
    """Appends unique detection to text file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ID: {human_id} | Plate: {plate_text}\n")
    print(f"📝 LOGGED: {plate_text} (ID: {human_id})")

# --- 4. MAIN LOOP ---
print(f"🚀 ALPR Active on Pi 5. Logging to {LOG_FILE}...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1

    # A. HEAVY INFERENCE (Every 5th frame)
    if frame_count % FRAME_SKIP == 0:
        # YOLO NCNN + imgsz=320 for maximum Pi 5 speed
        results = model(frame, imgsz=320, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                coords = box.xyxy[0].tolist()
                conf = box.conf.item()
                detections.append([*coords, conf])

        # Update Tracker
        detections_np = np.array(detections) if detections else np.empty((0, 5))
        last_tracked_objects = tracker.update(detections_np)

        # Process OCR for each car
        for obj in last_tracked_objects:
            x1, y1, x2, y2, raw_id = map(int, obj)
            h_id = get_human_id(raw_id)

            if h_id not in finalized_plates:
                # Clip crop to frame boundaries
                cx1, cy1 = max(0, x1), max(0, y1)
                cx2, cy2 = min(frame_width, x2), min(frame_height, y2)
                plate_crop = frame[cy1:cy2, cx1:cx2]
                
                if plate_crop.size > 0:
                    # Convert to RGB for PaddleOCR
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    ocr_res = ocr_reader.ocr(plate_rgb, cls=False)

                    if ocr_res and ocr_res[0]:
                        text = ocr_res[0][0][1][0].upper().replace(" ", "")
                        conf = ocr_res[0][0][1][1]

                        if conf >= OCR_CONF_THRESH:
                            # Update Stability Logic
                            if h_id not in stable_plates:
                                stable_plates[h_id] = {"text": text, "count": 1}
                            elif stable_plates[h_id]["text"] == text:
                                stable_plates[h_id]["count"] += 1
                            else:
                                # If text changes, reset count for the new text
                                stable_plates[h_id] = {"text": text, "count": 1}

                            # Finalize & Log at 5 matches
                            if stable_plates[h_id]["count"] == 5:
                                finalized_plates.add(h_id)
                                log_plate(text, h_id)

    # B. PERSISTENT DRAWING (Every frame - No blinking)
    for obj in last_tracked_objects:
        x1, y1, x2, y2, raw_id = map(int, obj)
        h_id = get_human_id(raw_id)
        
        display_text = stable_plates.get(h_id, {}).get("text", "Scanning...")
        # Green if finalized, Orange if scanning
        is_final = h_id in finalized_plates
        color = (0, 255, 0) if is_final else (0, 165, 255)
        
        # Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"ID:{h_id} {display_text}"
        cv2.putText(frame, label, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # C. SHOW OUTPUT
    #cv2.imshow('Pi 5 ALPR (Clean IDs)', frame)
    if cv2.waitKey(1) == ord('q'):
        break

    # Save the current frame for the webapp to see
    cv2.imwrite("static/latest.jpg", frame)
# --- 5. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print(f"✅ Session Complete. {next_human_id - 1} unique vehicles tracked.")
