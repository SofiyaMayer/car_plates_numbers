# Automatic License Plate Recognition (ALPR) System

![License Plate Detection Example](assets/demo.gif) <!-- Add a demo gif/screenshot here -->

An end-to-end license plate recognition system using YOLOv8 for detection, SORT for tracking, and PaddleOCR for text extraction. Optimized for real-time performance on edge devices.

## Key Features

- 🚗 **YOLOv8-based detection** - Custom-trained model (`license_plate.pt`) specifically for license plates
- 🔍 **PaddleOCR integration** - Superior accuracy for angled, blurry, or low-quality plates compared to EasyOCR
- 📹 **Video processing** - Handles MP4, AVI, and other common video formats
- 🏷️ **Real-time tracking** - SORT algorithm maintains consistent IDs across frames
- 📁 **Multiple outputs** - Generates annotated videos, cropped plates, and text logs

## Performance Metrics

| Metric            | Value   | Notes                          |
|-------------------|---------|--------------------------------|
| mAP50             | 98.6%   | Detection accuracy             |
| mAP50-95          | 70%     | Robustness across IoU thresholds|
| OCR Confidence    | ≥0.7    | Minimum threshold for valid reads |
| Processing Speed  | 25-30 FPS | On 1080p video (RTX 3060)     |

## Installation

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU (recommended) with CUDA 11.7 

### Steps
```bash
# Clone repository
git clone https://github.com/yourusername/license-plate-detection.git
cd license-plate-detection
```

```bash
# Install dependencies with uv https://docs.astral.sh/uv/
uv sync
```

Detection on video

In model testing this video was used:
https://www.pexels.com/video/traffic-flow-in-the-highway-2103099/

## Usage

1. Train yolov8n on your dataset with train_script.py or just use license_plate.pt for license plates recognition.
2. Use detection.py as a main script for detecting license plates numbers, upload your path to testing video first.

Dataset that was used for training YOLOv8n:
https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4 (download in your yolo format)



