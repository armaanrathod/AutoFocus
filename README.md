# 🚗 Enhanced Driver Monitoring System

A comprehensive AI-powered driver monitoring system using MediaPipe, OpenCV, and modern web technologies to detect drowsiness, distraction, and inattention in real-time.

## ✨ Features

### 🎯 Core Monitoring Capabilities
- **Drowsiness Detection**: Eye Aspect Ratio (EAR) analysis with configurable thresholds
- **Distraction Detection**: Advanced iris position tracking for gaze direction
- **Head Pose Monitoring**: Real-time head orientation tracking (yaw, pitch, roll)
- **Multi-level Alerts**: Progressive alert system with audio feedback

### 🌐 Web Dashboard
- **Real-time Monitoring**: Professional web interface with live statistics
- **Interactive Charts**: Eye tracking and head pose visualization
- **WebSocket Integration**: Real-time updates without page refresh
- **Responsive Design**: Works on desktop, tablet, and mobile devices

### 💾 Data Management
- **SQLite Database**: Automatic session and alert logging
- **Session Tracking**: Complete monitoring session history
- **Alert Analytics**: Detailed alert statistics and trends

### 🔧 Technical Stack
- **Computer Vision**: MediaPipe Face Mesh (468 facial landmarks)
- **Backend**: FastAPI with WebSocket support
- **Frontend**: Modern HTML5/CSS3/JavaScript
- **Database**: SQLite with SQLAlchemy ORM
- **Audio Alerts**: Platform-specific beep functionality

## 🚀 Quick Start

### Prerequisites
- Python 3.8-3.11 (MediaPipe compatibility)
- Webcam/camera device
- Windows/Linux/macOS support

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/armaanrathod/AutoFocus.git
cd AutoFocus
```

2. **Create virtual environment**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS  
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install opencv-python mediapipe fastapi uvicorn numpy sqlite3 sqlalchemy
```

### Running the System

1. **Start the monitoring server**
```bash
python driver_monitor_stable.py
```

2. **Open the dashboard**
Navigate to `http://127.0.0.1:8000` in your web browser

3. **Begin monitoring**
Click "Start Monitoring" in the dashboard to begin camera-based detection

## 📊 Detection Algorithms

### Eye Aspect Ratio (EAR)
- Calculates eye openness using facial landmark geometry
- Threshold: 0.25 (configurable)
- Alert trigger: 2+ seconds below threshold

### Iris Position Tracking
- Tracks iris position relative to eye corners
- Detects left/right gaze direction
- Alert trigger: 3+ seconds of distraction

### Head Pose Estimation
- 6DOF head tracking using PnP algorithm
- Yaw/Pitch thresholds: ±25°/±20°
- Alert trigger: 4+ seconds of extreme pose

## 🎛️ Configuration

### Alert Thresholds
```python
EAR_THRESHOLD = 0.25              # Eye closure threshold
DISTRACTION_THRESHOLD = 0.35       # Iris deviation from center
HEAD_YAW_THRESHOLD = 25           # Head left/right angle (degrees)
HEAD_PITCH_THRESHOLD = 20         # Head up/down angle (degrees)

# Time-based thresholds
DROWSY_TIME_THRESHOLD = 2.0       # Seconds
DISTRACTION_TIME_THRESHOLD = 3.0  # Seconds  
HEAD_TIME_THRESHOLD = 4.0         # Seconds
```

### Camera Settings
```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)
```

## 🎨 Dashboard Features

### Real-time Monitoring
- Live EAR values with visual progress indicators
- Eye state visualization with animated iris tracking  
- Head pose angles with historical charts
- Session duration and alert counters

### Alert Management
- Color-coded alert severity (High/Medium/Low)
- Recent alerts list with timestamps
- Alert statistics and trends
- Audio notification system

### Data Visualization
- EAR trend chart with drowsiness threshold line
- Head pose movement history (yaw/pitch)
- Eye position visualization with animated iris
- Real-time statistics updates

## 🗄️ Database Schema

### Sessions Table
- `id`: Primary key
- `start_time`: Session start timestamp  
- `end_time`: Session end timestamp
- `duration_seconds`: Total session length
- `total_alerts`: Total alert count
- `drowsy_alerts`: Drowsiness alert count
- `distraction_alerts`: Distraction alert count
- `head_alerts`: Head pose alert count

### Alerts Table
- `id`: Primary key
- `session_id`: Foreign key to sessions
- `timestamp`: Alert timestamp
- `alert_type`: drowsy/distraction/head_pose
- `severity`: high/medium/low
- `ear_value`: EAR at alert time
- `eye_state`: Eye position at alert
- `head_yaw/pitch`: Head angles at alert

## 🔌 API Endpoints

### REST API
- `GET /`: Dashboard interface
- `GET /api/stats`: Current monitoring statistics
- `POST /api/start_monitoring`: Begin monitoring session
- `POST /api/stop_monitoring`: End monitoring session

### WebSocket
- `WS /ws`: Real-time updates
  - `stats_update`: Live monitoring data
  - `alert`: Alert notifications
  - `monitoring_started`: Session start confirmation

## 🛠️ Development

### File Structure
```
AutoFocus/
├── driver_monitor_stable.py    # Main monitoring system
├── static/
│   └── dashboard.html          # Web dashboard
├── driver_monitor.db          # SQLite database (created on first run)
└── README.md                  # This file
```

### Extending the System
- **Custom Alert Types**: Add new detection algorithms in `monitoring_loop()`
- **UI Customization**: Modify `static/dashboard.html` for custom styling
- **Database Schema**: Extend tables in `init_database()` function
- **API Integration**: Add new endpoints in the FastAPI app section

## 🐛 Troubleshooting

### Common Issues
1. **MediaPipe compatibility**: Use Python 3.8-3.11
2. **Camera access**: Ensure no other applications are using the camera
3. **WebSocket errors**: Check firewall settings and port 8000 availability
4. **Performance issues**: Reduce camera resolution or frame rate

### Debug Mode
Add debug visualization by uncommenting the OpenCV display section:
```python
cv2.imshow('Driver Monitor', frame)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- MediaPipe team for robust facial landmark detection
- FastAPI developers for the excellent web framework
- OpenCV community for computer vision tools

## 📞 Support

For issues and questions:
- Create an issue in this repository
- Check the troubleshooting section above
- Review MediaPipe documentation for vision-related issues

---

**⚠️ Safety Notice**: This system is designed for research and development purposes. For production vehicle integration, additional safety testing and certification may be required.
