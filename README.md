An efficient Automatic License Plate Recognition (ALPR) system combining YOLOv8 for detection, SORT for tracking, and PaddleOCR for text recognition. Designed for real-world deployment with optimized performance.

## Features

- 🚗 **Direct license plate detection** using custom-trained YOLOv8n model
- 🔍 **High-accuracy OCR** with PaddleOCR (superior to EasyOCR for angled/blurry plates)
- 🏷️ **Frame-to-frame tracking** using SORT algorithm
- ⚡ **Optimized performance** for edge devices and Google Colab
- 📹 **Video processing** with frame-by-frame analysis
- 📊 **Comprehensive outputs** including annotated videos and plate data

## Performance

| Metric            | Value   |
|-------------------|---------|
| mAP50             | 98.6%   |
| mAP50-95          | 70%     |
| OCR Confidence    | ≥0.7    |
| FPS (1080p)       | ~25-30  |

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection
```
2. Install dependencies
!pip install ultralytics cv2 paddleocr filterpy, sort

Detection on video

In model testing this video was used:
https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/

## Usage

1. Train yolov8n on your dataset with train_script.py or just use license_plate.pt for license plates recognition.
2. Use detection.py as a main script for detecting license plates numbers.



