# YOLO Object Detection System

A comprehensive computer vision project implementing YOLOv11 for real-time object detection with support for images, videos, and live camera feeds.

## Project Overview

This project demonstrates the implementation of YOLO (You Only Look Once) object detection algorithm for real-time computer vision applications. The system includes both a custom-trained model and scripts for inference on various input sources including USB cameras, video files, and image datasets.

## Features

- **Real-time Object Detection**: Process live video streams from USB or Picamera with real-time FPS display
- **Multiple Input Sources**: Support for images, image folders, video files, USB cameras, and Raspberry Pi cameras
- **Custom Model Training**: Train custom YOLO models on specific datasets
- **High Performance**: Optimized inference pipeline with average FPS tracking
- **Flexible Detection**: Adjustable confidence thresholds and resolution settings
- **Video Recording**: Record detection results with bounding boxes and labels
- **Interactive Controls**: Pause, capture, and quit during inference

## Model Performance

The custom model was trained using YOLOv11s for 60 epochs with the following specifications:

- **Model**: YOLOv11s (Small)
- **Training Epochs**: 60
- **Batch Size**: 16
- **Image Size**: 640x640
- **Final mAP50**: 99.5%
- **Final mAP50-95**: 91.8%
- **Precision**: 98.4%
- **Recall**: 98.5%

## Project Structure

```
yolo/
├── 1.py                    # Model training and detection interface
├── yolo_detect.py          # Main detection script with advanced features
├── best.pt                 # Trained model weights (best performance)
├── my_model.pt             # Custom trained model
├── README.md               # Project documentation
└── my_model/
    ├── my_model.pt
    ├── yolo_detect.py
    └── train/
        ├── args.yaml       # Training configuration
        ├── results.csv     # Training metrics and performance
        └── weights/
            ├── best.pt     # Best model checkpoint
            └── last.pt     # Last epoch checkpoint
```

## Installation

### Prerequisites

- Python 3.12 or higher
- Conda (recommended) or pip
- Webcam or camera (for live detection)

### Setup Environment

```bash
# Create and activate conda environment
conda create --name yolo-env1 python=3.12 -y
conda activate yolo-env1

# Install required packages
pip install ultralytics opencv-python numpy
```

For Raspberry Pi camera support:
```bash
pip install picamera2
```

## Usage

### Running Object Detection

Navigate to the project directory and run the detection script:

```bash
cd "c:\Users\User\OneDrive\Desktop\sem 4 projects\yolo"
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720
```

### Command Line Arguments

- `--model`: Path to YOLO model file (e.g., `my_model.pt` or `runs/detect/train/weights/best.pt`)
- `--source`: Input source options:
  - Image file: `test.jpg`
  - Image folder: `test_images/`
  - Video file: `video.mp4`
  - USB camera: `usb0` (or `usb1`, `usb2`, etc.)
  - Picamera: `picamera0`
- `--thresh`: Confidence threshold (default: 0.5)
- `--resolution`: Display resolution in WxH format (e.g., `1280x720`)
- `--record`: Record output video (requires `--resolution`)

### Examples

**Detect objects in an image:**
```bash
python yolo_detect.py --model my_model.pt --source image.jpg
```

**Process a video file:**
```bash
python yolo_detect.py --model my_model.pt --source video.mp4 --resolution 1920x1080
```

**Live webcam detection with recording:**
```bash
python yolo_detect.py --model my_model.pt --source usb0 --resolution 1280x720 --record
```

**Batch process images in a folder:**
```bash
python yolo_detect.py --model my_model.pt --source ./test_images/
```

### Interactive Controls

During inference:
- Press **'q'** to quit
- Press **'s'** to pause/resume
- Press **'p'** to save a screenshot of the current frame

## Training Custom Models

Use the `1.py` script for an interactive training and detection interface:

```bash
python 1.py
```

Follow the menu prompts to:
1. Train a new model with your dataset
2. Run detection with pretrained models
3. Run detection with custom trained models

## Technical Details

### Detection Pipeline

1. **Input Processing**: Frame acquisition from specified source
2. **Preprocessing**: Resolution adjustment and format conversion
3. **Inference**: YOLO model prediction with confidence filtering
4. **Post-processing**: Bounding box visualization and label annotation
5. **Output Display**: Real-time visualization with FPS counter

### Performance Optimization

- Tensor operations optimized for CPU/GPU
- Frame rate averaging over 200 frames for stable FPS display
- Efficient bounding box rendering with Tableau color scheme
- Minimal latency for real-time applications

## Results

The trained model achieves excellent performance metrics:
- **High Accuracy**: 99.5% mAP at IoU threshold 0.5
- **Robust Detection**: 91.8% mAP across IoU thresholds 0.5-0.95
- **Reliable Performance**: 98%+ precision and recall

## Technologies Used

- **YOLOv11**: State-of-the-art object detection model
- **Ultralytics**: YOLO implementation and training framework
- **OpenCV**: Computer vision and image processing
- **NumPy**: Numerical computations
- **Python**: Core programming language

## Future Enhancements

- Multi-object tracking implementation
- Custom dataset integration tools
- Model optimization for edge devices
- Web-based inference interface
- Mobile deployment support

## License

This project is developed for educational purposes as part of Semester 4 coursework.

## Author

Developed as part of academic project work

---

**Note**: Ensure your camera is properly connected and accessible before running live detection. For best results, use good lighting conditions and appropriate confidence thresholds for your use case.
