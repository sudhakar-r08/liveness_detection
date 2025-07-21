# Face Liveness Detection System

A real-time face liveness detection system that distinguishes between real faces and fake faces (photos, videos, masks) using computer vision and deep learning.

## Features

- **Real-time Detection**: Live video stream processing from webcam or IP camera
- **Face Detection**: OpenCV DNN-based face detection
- **Liveness Classification**: Deep learning model to classify real vs fake faces
- **API Integration**: Automatic status updates to backend API
- **Logging System**: Host-based activity logging
- **Multi-source Input**: Support for webcam, RTSP streams, and HTTP video feeds

## Requirements

### Python Dependencies
```
opencv-python
imutils
numpy
keras
tensorflow
requests
argparse
```

### System Requirements
- Python 3.6+
- Webcam or IP camera
- CUDA-compatible GPU (optional, for better performance)

## Installation

1. **Clone the repository**
```bash
git clone <your-repository-url>
cd liveness-detection
```

2. **Install dependencies**
```bash
pip install opencv-python imutils numpy keras tensorflow requests
```

3. **Download required models**
   - Place your trained liveness model as `liveness.model`
   - Place label encoder as `le.pickle`
   - Download OpenCV face detector files in `face_detector/` folder:
     - `deploy.prototxt`
     - `res10_300x300_ssd_iter_140000.caffemodel`

## Usage

### Basic Usage
```bash
python liveness_demo.py
```

### With Custom Parameters
```bash
python liveness_demo.py --model path/to/model.h5 --le path/to/encoder.pickle --detector path/to/face_detector --confidence 0.8
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | `liveness.model` | Path to trained liveness detection model |
| `--le` | `le.pickle` | Path to label encoder |
| `--detector` | `face_detector` | Path to OpenCV face detector directory |
| `--confidence` | `0.75` | Minimum confidence threshold for face detection |

## Project Structure

```
liveness-detection/
├── liveness_demo.py          # Main application script
├── liveness.model            # Trained liveness detection model
├── le.pickle                 # Label encoder
├── face_detector/            # OpenCV face detector files
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## How It Works

1. **Face Detection**: Uses OpenCV's DNN face detector to locate faces in video frames
2. **Preprocessing**: Detected faces are resized to 32x32 pixels and normalized
3. **Liveness Classification**: Preprocessed faces are fed to a trained CNN model
4. **Result Display**: Real faces are marked with green boxes, fake faces with red boxes
5. **API Updates**: Detection results are sent to configured API endpoints

## API Endpoints

The system integrates with the following APIs:

- **Liveness Status**: `POST http://localhost/liveness-websocket/api/liveness.php`
- **Activity Logging**: `POST http://localhost/liveness-websocket/api/log.php`
- **Validity Check**: `GET http://url.com/t={hostname}`

## Configuration

### Video Sources
The system supports multiple video input sources:

```python
# Webcam (default)
cap.open(0)

# RTSP Stream
cap.open("rtsp://admin:admin@12345@192.168.1.102:554/Streaming/channels/1/")

# HTTP Stream
cap.open("http://192.168.1.6:8082/video.mjpg?q=30&fps=33&id=0.2729321831683187&r=1586790060214")
```

### Detection Thresholds
- **Face Detection Confidence**: Default 0.75 (adjustable via `--confidence`)
- **Liveness Prediction Threshold**: 0.7 (hardcoded in script)

## Controls

- **Quit Application**: Press 'q' key to exit
- **Real-time Display**: Live video feed with detection results

## Output

- **Green Rectangle**: Real face detected
- **Red Rectangle**: Fake face detected
- **Label Format**: `{classification}: {confidence_score}`

## Troubleshooting

### Common Issues

1. **Camera not opening**
   - Check camera permissions
   - Verify camera index (try 0, 1, 2...)

2. **Model loading errors**
   - Ensure model files exist in correct paths
   - Check file permissions

3. **API connection errors**
   - Verify API endpoints are accessible
   - Check network connectivity

4. **Low detection accuracy**
   - Adjust confidence threshold
   - Ensure good lighting conditions
   - Retrain model with more diverse data

## Performance Tips

- Use GPU acceleration for better performance
- Adjust frame processing rate by modifying the fps calculation
- Lower video resolution for faster processing

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## Support

For issues and questions:
- Create an issue in the repository
