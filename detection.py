from ultralytics import YOLO
import cv2
import numpy as np
from paddleocr import PaddleOCR
from sort.sort import Sort
import os

# Conditionally import Google Colab files module
try:
    from google.colab import files

    is_colab = True
except ImportError:
    is_colab = False

# Create directories for saving cropped plates
if not os.path.exists('cropped_plates'):
    os.makedirs('cropped_plates')


# Configuration
TARGET_PLATE = "EL924CF"
detected_frame_path = "target_car_detected.jpg"
video_path = 'input1.mp4'
output_video_path = "car_detection_output1.mp4"

# Load models (only once)
model = YOLO('model/runs/train/license_plates2/weights/license_plate.pt')
# Restrict characters to those commonly found on license plates
allowlist = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
ocr_reader = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=True,
    rec_batch_num=1,
    cls_thresh=0.9,
    det_db_thresh=0.3,
    det_db_box_thresh=0.5,
    show_log=False
)

# Detection parameters
conf_threshold = 0.2  # Confidence threshold for detection
iou_threshold = 0.6  # IOU threshold for NMS
box_thickness = 2  # Bounding box thickness
box_color = (0, 255, 0)  # Green color for boxes
ocr_conf_threshold = 0.7  # Confidence threshold for OCR result

# Load video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Prepare video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=10, iou_threshold=0.5)

# Tracking data storage
track_id_dict = {}
detection_data = []
target_car_detected = False

plate_history_dict = {}  # Track plate readings per vehicle
stable_plates = {}  # For storing stable license plate readings

print(f"Processing video {video_path}...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get detections
    results = model(frame, conf=conf_threshold, iou=iou_threshold)

    detections = []
    for result in results:
        for box in result.boxes:
            # Filter by class name
            if model.names[int(box.cls)] == "License_Plate":
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = box.conf.item()
                detections.append([x1, y1, x2, y2, conf])

    # Update tracker with current detections
    detections_np = np.array(detections) if detections else np.empty((0, 5))
    tracked_objects = tracker.update(detections_np)

    # Process tracked objects
    frame_data = {
        "frame_number": int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
        "cars": []
    }

    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = obj
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        track_id = int(track_id)

        # Initialize plate history for this track if needed
        if track_id not in plate_history_dict:
            plate_history_dict[track_id] = []

        # Crop license plate area
        car_crop = frame[y1:y2, x1:x2]
        if car_crop.size == 0:
            continue

        # Enhanced OCR preprocessing specifically for license plates
        car_crop_gray = cv2.cvtColor(car_crop, cv2.COLOR_BGR2GRAY)

        # Resize for better OCR if plate is too small or too large
        h, w = car_crop_gray.shape
        if h < 20:  # If too small
            scale_factor = 30.0 / h
            car_crop_gray = cv2.resize(car_crop_gray, None, fx=scale_factor, fy=scale_factor,
                                       interpolation=cv2.INTER_CUBIC)
        elif h > 100:  # If too large
            scale_factor = 60.0 / h
            car_crop_gray = cv2.resize(car_crop_gray, None, fx=scale_factor, fy=scale_factor,
                                       interpolation=cv2.INTER_AREA)

        # Define max_acceptable_conf value and initialize tracking variables
        max_acceptable_conf = 0.9
        best_text = ""
        best_conf = 0
        best_candidate = None
        best_name = ""


        def process_with_paddleocr(candidate):
            """Process image with PaddleOCR and return text and confidence"""
            result = ocr_reader.ocr(candidate, cls=True)

            if result and result[0]:
                # Extract highest confidence result
                best_result = None
                best_conf = 0

                for line in result[0]:
                    text, conf = line[1]
                    # Filter non-license plate characters
                    text = ''.join(c for c in text if c in allowlist)
                    if text and conf > best_conf:
                        best_result = (text, conf)
                        best_conf = conf

                if best_result:
                    return best_result

            return "", 0.0


        # Convert grayscale to RGB (PaddleOCR expects RGB input)
        if len(car_crop_gray.shape) == 2:  # If grayscale
            candidate_rgb = cv2.cvtColor(car_crop_gray, cv2.COLOR_GRAY2RGB)
        else:
            candidate_rgb = car_crop_gray

        # Process with PaddleOCR
        text, conf = process_with_paddleocr(candidate_rgb)

        # Update best result tracking
        if conf > best_conf:
            best_text = text
            best_conf = conf
            best_candidate = car_crop_gray
            best_name = "PaddleOCR"

        if best_candidate is not None:
            vis_filename = f"cropped_plates/vis_{track_id}_{frame_data['frame_number']}.jpg"

            # Create comparison visualization
            h, w = car_crop_gray.shape
            comparison = np.zeros((h, w * 2), dtype=np.uint8)
            comparison[:, :w] = car_crop_gray
            comparison[:, w:] = cv2.resize(best_candidate, (w, h)) if best_candidate.shape != (h, w) else best_candidate

            # Add text showing the OCR result and which method was best
            comparison_vis = cv2.cvtColor(comparison, cv2.COLOR_GRAY2BGR)
            cv2.putText(comparison_vis, f"Original | {best_text} ({best_conf:.2f})",
                        (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(comparison_vis, f"Method: {best_name}",
                        (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(vis_filename, comparison_vis)

        # Set license_plate variable based on best_text
        license_plate = best_text if best_conf >= ocr_conf_threshold else ""

        # Clean the license plate text (remove spaces, normalize)
        if license_plate:
            license_plate = ''.join(license_plate.split()).upper()

            # Update plate history for this track
            plate_history_dict[track_id].append(license_plate)
            # Keep only recent readings
            if len(plate_history_dict[track_id]) > 10:
                plate_history_dict[track_id] = plate_history_dict[track_id][-10:]

        # Update stable plates dictionary
        if license_plate:
            if track_id not in stable_plates:
                stable_plates[track_id] = {"text": license_plate, "count": 1}
            elif stable_plates[track_id]["text"] == license_plate:
                stable_plates[track_id]["count"] += 1
            elif stable_plates[track_id]["count"] < 3:  # Only replace if current stable reading is weak
                stable_plates[track_id] = {"text": license_plate, "count": 1}

        # Get the most stable plate reading for display
        display_plate = ""
        if track_id in stable_plates and stable_plates[track_id]["count"] >= 2:
            display_plate = stable_plates[track_id]["text"]
        elif license_plate:
            display_plate = license_plate + "?"  # Mark unstable readings with ?

        # Draw bounding box with enhanced text visibility
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, box_thickness)

        # Draw a background for the text for better visibility
        if display_plate:
            label = f"ID: {track_id} | LP: {display_plate}"
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Draw background rectangle
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_width, y1), (0, 0, 0), -1)
            # Draw text with bigger size
            cv2.putText(frame, label, (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            label = f"ID: {track_id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_width, y1), (0, 0, 0), -1)
            cv2.putText(frame, label, (x1, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Check for target plate
        if license_plate == TARGET_PLATE and len(plate_history_dict[track_id]) >= 3:
            recent_readings = plate_history_dict[track_id][-3:]
            plate_counts = {}
            for plate in recent_readings:
                plate_counts[plate] = plate_counts.get(plate, 0) + 1

            most_common = max(plate_counts, key=plate_counts.get)

            if most_common == TARGET_PLATE and plate_counts[most_common] >= 2:
                cv2.imwrite(detected_frame_path, frame)
                print(f"🚨 The car with number {TARGET_PLATE} detected! Track ID: {track_id}")
                if is_colab:
                    files.download(detected_frame_path)
                target_car_detected = True

        # Update track ID dictionary
        if track_id not in track_id_dict:
            track_id_dict[track_id] = {
                'license_plates': [],
                'first_seen': frame_data["frame_number"],
                'last_seen': frame_data["frame_number"]
            }
        else:
            track_id_dict[track_id]['last_seen'] = frame_data["frame_number"]

        if license_plate:
            track_id_dict[track_id]['license_plates'].append(license_plate)

        # Store tracking data
        frame_data["cars"].append({
            "track_id": track_id,
            "bbox": [x1, y1, x2, y2],
            "license_plate": license_plate
        })

    detection_data.append(frame_data)
    out.write(frame)

cap.release()
out.release()

# Cleanup old tracks
current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
max_track_age = 200  # frames to keep track after last detection
for track_id in list(track_id_dict.keys()):
    if current_frame - track_id_dict[track_id]['last_seen'] > max_track_age:
        del track_id_dict[track_id]

print(f"✅ Car detection and tracking complete! Output saved as {output_video_path}")
print(f"✅ Detected {len(track_id_dict)} unique vehicles in total")
print(f"✅ Detection data contains {len(detection_data)} frames")