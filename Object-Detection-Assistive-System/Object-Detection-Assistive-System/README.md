# 🦯 Object Detection Assistive System for Visually Impaired People

An advanced real-time object detection system designed to help visually impaired individuals navigate their environment safely using audio feedback and spatial awareness cues.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## 🌟 Features

### Core Functionality
- ✅ **Real-time Object Detection** - YOLO-based detection with high accuracy
- 🎥 **Multiple Input Sources** - Camera (webcam) or video file support
- 📸 **Image Detection** - Detect objects in static pictures with confidence scores
- 🔊 **Advanced Audio System** - Priority-based announcements with spatial cues
- 📍 **Directional Beeps** - Different pitch for left/center/right (like parking sensors)
- ⚡ **Distance-Based Alerts** - Faster warnings as objects get closer
- 🎯 **Smart Priority System** - Critical objects (people, vehicles) announced first
- 🔄 **Motion Detection** - Identifies approaching vs. static objects
- 🖥️ **GPU Acceleration** - Automatic CUDA support for faster processing
- 📊 **Confidence/Accuracy Display** - See detection accuracy for each object

### Enhanced Features (v2.0)
- 📝 **Configuration File** - Customize all settings via `config.json`
- 🎮 **Command-line Interface** - Flexible control with arguments
- 📊 **Statistics Dashboard** - Real-time session stats overlay
- 🚀 **Performance Optimization** - Frame skipping, GPU detection
- 💾 **Portable** - No hardcoded paths, works anywhere
- 🔧 **Privacy Protection** - Optional face blurring
- 📸 **Image Detection Mode** - Process static pictures with batch support
- 📈 **Confidence Scores** - Accuracy percentage for every detection

---

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Operating System**: Windows (for beep sounds), Linux/Mac (audio only)
- **Camera**: Any USB webcam (for live detection)
- **GPU**: NVIDIA GPU with CUDA (optional, for better performance)

### Dependencies
See `requirements.txt` for full list:
- `ultralytics` - YOLO model
- `opencv-python` - Computer vision
- `torch` & `torchvision` - Deep learning
- `pyttsx3` - Text-to-speech
- `numpy` - Numerical computing

---

## 🚀 Installation

### 1. Clone or Download Repository
```bash
git clone <repository-url>
cd Object-Detection-Assistive-System-for-Visually-Impaired-People-main
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download or Train YOLO Model
Place your YOLO model file (`gpModel.pt`) in the project directory.

### 4. Verify Installation
```bash
python verify_setup.py
```

### 5. Test the System
```bash
# Test with camera
python main_enhanced.py --help

# Test with image
python detect_image.py --help
```

---

## 🎮 Usage

### Quick Start

#### 1. Real-time Camera Detection
```bash
python main_enhanced.py --camera
```

#### 2. Video File Processing
```bash
python main_enhanced.py --video test_video2.mp4
```

#### 3. Image Detection (NEW!)
```bash
# Single image
python detect_image.py --image photo.jpg

# Batch process multiple images
python detect_image.py --batch-dir ./photos
```

#### 4. Use Specific Camera
```bash
python main_enhanced.py --camera --camera-id 1
```

---

### Command-Line Options

```bash
python main_enhanced.py [OPTIONS]

Options:
  --camera              Use camera as input source
  --camera-id ID        Camera device ID (default: 0)
  --video PATH          Path to video file
  --model PATH          Path to YOLO model file
  --config PATH         Path to config file (default: config.json)
  --no-display          Disable visual display (audio only)
  --no-save             Don't save output video
  --gpu                 Force GPU usage
  --cpu                 Force CPU usage
  -h, --help            Show help message
```

---

### Keyboard Controls (During Execution)

| Key | Action | Description |
|-----|--------|-------------|
| **S** | START | Activate audio system (announcements begin) |
| **E** | STOP | Deactivate audio system (silence) |
| **P** | PAUSE | Pause/resume video playback |
| **Q** | QUIT | Exit the program |
| **R** | RESET | Reset statistics counter |

---

## ⚙️ Configuration

### Config File (`config.json`)

The system is highly customizable through the configuration file:

```json
{
  "video": {
    "source": "camera",           // "camera" or "file"
    "camera_id": 0,               // Camera device ID
    "video_file": "test.mp4",     // Video file path
    "output_file": "output.avi",  // Output save path
    "display_window": true,       // Show visual window
    "save_output": true,          // Save output video
    "fps": 20                     // Output video FPS
  },
  "model": {
    "path": "gpModel.pt",         // YOLO model path
    "confidence_threshold": 0.5,  // Detection confidence
    "use_gpu": true               // Use GPU if available
  },
  "audio": {
    "enabled": true,              // Enable audio system
    "start_on_launch": false,     // Auto-start audio
    "speech_rate": 300,           // Words per minute
    "volume": 1.0,                // Volume (0.0 - 1.0)
    "language": "en"              // Language code
  },
  "detection": {
    "max_distance": 12.5,         // Max detection range (m)
    "frame_skip": 0,              // Skip frames for performance
    "blur_faces": true            // Privacy protection
  },
  "alerts": {
    "beep_frequencies": {
      "left": 300,                // Low pitch (Hz)
      "center": 500,              // Mid pitch (Hz)
      "right": 800                // High pitch (Hz)
    },
    "cooldown_times": {
      "danger_zone": 0.3,         // <1.5m announcement interval (s)
      "very_close": 0.6,          // <3m announcement interval (s)
      "close": 1.0,               // <5m announcement interval (s)
      "far": 1.5                  // >5m announcement interval (s)
    },
    "distance_zones": {
      "danger": 1.5,              // Danger zone threshold (m)
      "very_close": 3.0,          // Very close zone (m)
      "close": 5.0,               // Close zone (m)
      "far": 8.0                  // Far zone (m)
    },
    "object_priorities": {
      "person": 2,                // Lower = higher priority
      "car": 2,
      "truck": 12,
      // ... add more objects
    }
  },
  "display": {
    "show_distance": true,        // Show distance labels
    "show_boxes": true,           // Show bounding boxes
    "show_statistics": true,      // Show stats overlay
    "show_controls": true,        // Show control hints
    "font_scale": 0.6,            // Text size
    "box_thickness": 2            // Box line thickness
  }
}
```

---

## 🔊 Audio System

### How It Works

1. **Spatial Audio Beeps**
   - **Left objects**: Low pitch (300 Hz) 🔉
   - **Center objects**: Mid pitch (500 Hz) 🔊
   - **Right objects**: High pitch (800 Hz) 🔔

2. **Distance-Based Warnings**
   - **≤1.5m**: "DANGER!" + rapid beeps (0.3s interval)
   - **≤3m**: "Warning" + fast beeps (0.6s interval)
   - **≤5m**: Normal announcement + medium beeps (1.0s)
   - **>5m**: Slow announcements (1.5s interval)

3. **Priority System**
   - Critical objects (people, cars) announced first
   - Approaching objects bypass cooldown timers
   - Multiple dangers = up to 4 simultaneous announcements

### Example Audio Timeline

```
Scenario: Person walking toward you from the left

Time    Distance    Audio Output
----    --------    ------------
0.0s    5.0m       🔊 "person left 5 meters" [beep-low]
1.0s    4.5m       🔊 "person left 4.5 approaching" [beep-low]
1.6s    4.0m       🔊 "person left 4 approaching" [beep-low]
2.2s    3.5m       🔊 "person left 3.5 approaching" [beep-beep-low]
2.8s    2.5m       🔊 "Warning person left 2.5 very close" [beep-beep-beep-low]
3.0s    2.0m       🔊 "Warning person left 2 very close" [beep-beep-beep-low]
3.3s    1.5m       🔊 "DANGER! person left 1.5 meters" [BEEP-BEEP-BEEP-BEEP-low]
3.6s    1.2m       🔊 "DANGER! person left 1.2 meters" [BEEP-BEEP-BEEP-BEEP-low]
```

---

## 📊 Visual Interface

### Status Indicators

```
┌──────────────────┐
│ AUDIO: ACTIVE    │ ← Green box (system ON)
└──────────────────┘

┌──────────────────┐
│ AUDIO: STOPPED   │ ← Red box (system OFF)
└──────────────────┘
```

### Statistics Panel (Top Right)
- Session duration
- Total detections
- Closest encounter (object & distance)
- Current FPS

### Information Overlay
- Frame counter
- Objects being tracked
- Closest object details
- Speaking indicator (red circle)

---

## 🛠️ Customization Guide

### Adjusting Speech Speed
```json
"audio": {
  "speech_rate": 300  // Range: 150-400 (words/minute)
}
```

### Changing Alert Zones
```json
"distance_zones": {
  "danger": 1.5,      // Immediate danger (m)
  "very_close": 3.0,  // Very close (m)
  "close": 5.0,       // Close (m)
  "far": 8.0          // Far (m)
}
```

### Adding New Objects
```json
"object_priorities": {
  "dog": 3,           // Priority level (1=highest)
  "cat": 5,
  "chair": 10
}
```

### Performance Tuning
```json
"detection": {
  "frame_skip": 2,    // Process every 3rd frame (0=no skip)
  "confidence_threshold": 0.6  // Higher = fewer false positives
}
```

---

## 📁 Project Structure

```
Object-Detection-Assistive-System/
├── main.py                      # Original version
├── main_enhanced.py             # Enhanced version (v2.0)
├── detect_image.py              # Image detection with confidence scores (NEW!)
├── verify_setup.py              # Setup verification script
├── config.json                  # Configuration file
├── config_examples.json         # Example configurations
├── requirements.txt             # Python dependencies
├── README.md                    # This file (main documentation)
├── QUICK_START.md               # 5-minute quick start guide
├── IMAGE_DETECTION_GUIDE.md     # Image detection documentation (NEW!)
├── AUDIO_IMPROVEMENTS.md        # Audio system documentation
├── CONTINUOUS_AUDIO_UPDATE.md   # Continuous feedback docs
├── CONTROLS_GUIDE.md            # Keyboard controls guide
├── ENHANCEMENTS_SUMMARY.md      # All enhancements summary
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── gpModel.pt                   # YOLO model (your trained model)
├── test_video2.mp4              # Sample video (optional)
├── output_with_boxes.avi        # Generated video output
└── output_images/               # Generated image detections (NEW!)
    ├── *_detected.jpg           # Annotated images
    └── detection_report.json    # Detection statistics
```

---

## 🧪 Testing

### Test Camera
```bash
python main_enhanced.py --camera
# Press 'S' to start audio
# Move in front of camera
# Listen for announcements
```

### Test Video File
```bash
python main_enhanced.py --video test_video2.mp4
# Press 'S' to start audio
# Observe detections
```

### Test Audio Only (No Display)
```bash
python main_enhanced.py --camera --no-display
# For blind users: audio-only mode
```

---

## 🔧 Troubleshooting

### Issue: No audio output
**Solution:**
- Check system volume is not muted
- Verify `pyttsx3` is installed: `pip install pyttsx3`
- Try different speech engine (edit config)
- Check console for "Audio System Activated" message

### Issue: Camera not detected
**Solution:**
```bash
# Try different camera IDs
python main_enhanced.py --camera --camera-id 0
python main_enhanced.py --camera --camera-id 1
python main_enhanced.py --camera --camera-id 2
```

### Issue: Low FPS / Slow performance
**Solution:**
- Enable GPU: `--gpu` (requires CUDA)
- Skip frames: Set `"frame_skip": 2` in config
- Lower resolution: Reduce video quality
- Use smaller model: Try lighter YOLO variant

### Issue: Too many/few announcements
**Solution:**
- Adjust cooldown times in `config.json`
- Change `max_distance` to limit detection range
- Modify `confidence_threshold` (higher = fewer detections)

### Issue: Model file not found
**Solution:**
```bash
# Specify model path explicitly
python main_enhanced.py --model path/to/your/model.pt
```

---

## 🎯 Use Cases

### 1. **Indoor Navigation**
- Detect people, furniture, obstacles
- Navigate hallways and rooms
- Avoid collisions

### 2. **Outdoor Walking**
- Detect vehicles (cars, bikes, motorcycles)
- Identify traffic signs
- Safe street crossing

### 3. **Public Spaces**
- Crowded area navigation
- Object avoidance in malls, stations
- Real-time spatial awareness

### 4. **Training & Education**
- Teach visually impaired individuals about their environment
- Practice navigation in controlled settings
- Build spatial awareness skills

---

## 🚀 Performance Tips

1. **Use GPU**: Install CUDA and PyTorch with GPU support
   ```bash
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Optimize Frame Rate**: Skip frames for faster processing
   ```json
   "frame_skip": 1  // Process every other frame
   ```

3. **Adjust Detection Range**: Reduce max_distance for faster processing
   ```json
   "max_distance": 8.0  // Only detect objects <8 meters
   ```

4. **Lower Resolution**: Use lower resolution camera/video
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

---

## 📈 Future Enhancements

- [ ] **3D Spatial Audio** - Stereo panning for better directionality
- [ ] **Voice Commands** - Hands-free control ("start", "stop")
- [ ] **Multi-language Support** - Announcements in various languages
- [ ] **Mobile App** - Android/iOS version
- [ ] **Object Tracking** - Track specific objects across frames
- [ ] **Path Recommendation** - Suggest safe walking paths
- [ ] **Haptic Feedback** - Vibration alerts (mobile)
- [ ] **Cloud Integration** - Remote monitoring/assistance
- [ ] **Custom Object Training** - Add specific objects to detect
- [ ] **Depth Camera Support** - Better distance estimation

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest new features
- Improve documentation
- Submit pull requests

---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **YOLOv8** by Ultralytics for object detection
- **OpenCV** for computer vision tools
- **pyttsx3** for text-to-speech
- Visually impaired community for feedback and testing

---

## 📞 Support

For issues, questions, or suggestions:
- Create an issue on GitHub
- Check documentation files (AUDIO_IMPROVEMENTS.md, CONTROLS_GUIDE.md)
- Review troubleshooting section above

---

## 🎓 Educational Purpose

This project was developed to demonstrate:
- Real-time AI for accessibility
- Audio-based user interfaces
- Computer vision for assistive technology
- Inclusive design principles

**Built with ❤️ to make technology accessible to everyone**

---

## 📝 Version History

### v2.0 (Enhanced) - 2025
- ✅ Real-time camera support
- ✅ Configuration file system
- ✅ Command-line interface
- ✅ Statistics dashboard
- ✅ Performance optimization
- ✅ Portable design (no hardcoded paths)
- ✅ GPU acceleration support

### v1.0 (Original) - 2025
- ✅ Basic object detection
- ✅ Audio announcements
- ✅ Spatial beeps
- ✅ Distance calculation
- ✅ Video file processing

---

**Made for the visually impaired community to navigate the world with confidence! 🦯**

