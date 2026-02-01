import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from sort.sort import Sort
from datetime import datetime
import os

# --- 1. CONFIGURATION ---
TARGET_PLATE = "EL924CF"
MODEL_PATH = 'license_plate_ncnn_model' # Folder created by your export
LOG_FILE = "detected_plates.txt"
FRAME_SKIP = 5  # Process YOLO/OCR every 5th frame
OCR_CONF_THRESH = 0.7

# --- 2. INITIALIZATION ---
# Load optimized NCNN model for Pi 5
model = YOLO(MODEL_PATH, task='detect')

# Initialize PaddleOCR (Mobile version)
ocr_reader = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=False, show_log=False)

# Initialize SORT tracker
tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Video source (0 for webcam)
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Persistence storage
stable_plates = {}    # {track_id: {"text": "ABC", "count": 5}}
finalized_plates = set() 
last_tracked_objects = []
frame_count = 0

def log_plate(plate_text, track_id):
    """Saves unique detection to text file with a timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a") as f:
        f.write(f"[{timestamp}] ID: {track_id} | Plate: {plate_text}\n")
    print(f"📝 LOGGED: {plate_text} (ID: {track_id})")

# --- 3. MAIN LOOP ---
print("🚀 ALPR Running on Pi 5. Press 'q' to stop.")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame_count += 1

    # A. HEAVY INFERENCE (Every Nth frame)
    if frame_count % FRAME_SKIP == 0:
        # YOLO Detection with reduced imgsz for speed
        results = model(frame, imgsz=320, conf=0.25, verbose=False)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                detections.append([x1, y1, x2, y2, conf])

        # Update Tracker
        detections_np = np.array(detections) if detections else np.empty((0, 5))
        last_tracked_objects = tracker.update(detections_np)

        # Process OCR for tracked items
        for obj in last_tracked_objects:
            x1, y1, x2, y2, track_id = map(int, obj)
            track_id = int(track_id)

            if track_id not in finalized_plates:
                # Clip coordinates to frame
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)
                plate_crop = frame[y1:y2, x1:x2]
                
                if plate_crop.size > 0:
                    plate_rgb = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2RGB)
                    ocr_res = ocr_reader.ocr(plate_rgb, cls=False)

                    if ocr_res and ocr_res[0]:
                        text = ocr_res[0][0][1][0].upper().replace(" ", "")
                        conf = ocr_res[0][0][1][1]

                        if conf >= OCR_CONF_THRESH:
                            # Update Stability
                            if track_id not in stable_plates:
                                stable_plates[track_id] = {"text": text, "count": 1}
                            elif stable_plates[track_id]["text"] == text:
                                stable_plates[track_id]["count"] += 1
                            else:
                                stable_plates[track_id] = {"text": text, "count": 1}

                            # LOGGING: Write to file exactly once at the 5th stable detection
                            if stable_plates[track_id]["count"] == 5:
                                finalized_plates.add(track_id)
                                log_plate(text, track_id)

    # B. PERSISTENT DRAWING (Every frame - prevents blinking)
    for obj in last_tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj)
        
        display_text = stable_plates.get(track_id, {}).get("text", "Scanning...")
        is_stable = track_id in finalized_plates or stable_plates.get(track_id, {}).get("count", 0) >= 3
        
        color = (0, 255, 0) if is_stable else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID:{track_id} {display_text}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # C. VISUAL OUTPUT
    cv2.imshow('Raspberry Pi 5 ALPR', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# --- 4. CLEANUP ---
cap.release()
cv2.destroyAllWindows()
print(f"✅ Finished. Logs saved to {LOG_FILE}")
