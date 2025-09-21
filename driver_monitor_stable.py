#!/usr/bin/env python3
"""
🚗 Enhanced Driver Monitoring System (Stable Version)
📊 Features: MediaPipe + Database + Web Dashboard + Real-time Alerts

This system provides comprehensive driver monitoring using:
- MediaPipe for face detection and landmark analysis
- Eye Aspect Ratio (EAR) for drowsiness detection
- Iris position tracking for distraction detection
- Head pose estimation for attention monitoring
- FastAPI web server with real-time WebSocket updates
- SQLite database for session logging
- Professional web dashboard with live statistics

Author: Enhanced by AI Assistant
Version: Stable v2.0
Date: September 2025
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import sqlite3
import json
import threading
import queue
import asyncio
import platform
from datetime import datetime
from collections import deque
import winsound  # For Windows beep

# FastAPI and WebSocket imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn

# ---------------------- Database Setup ----------------------
def init_database():
    """Initialize SQLite database with required tables."""
    conn = sqlite3.connect('driver_monitor.db')
    cursor = conn.cursor()
    
    # Create sessions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_time TIMESTAMP,
            duration_seconds INTEGER,
            total_alerts INTEGER DEFAULT 0,
            drowsy_alerts INTEGER DEFAULT 0,
            distraction_alerts INTEGER DEFAULT 0,
            head_alerts INTEGER DEFAULT 0,
            avg_ear REAL,
            notes TEXT
        )
    ''')
    
    # Create alerts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            alert_type TEXT,
            severity TEXT,
            ear_value REAL,
            eye_state TEXT,
            head_yaw REAL,
            head_pitch REAL,
            FOREIGN KEY (session_id) REFERENCES sessions (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database initialized successfully")

# ---------------------- WebSocket Manager ----------------------
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections.copy():  # Use copy to avoid modification during iteration
            try:
                await connection.send_text(message)
            except:
                # Remove disconnected connections
                if connection in self.active_connections:
                    self.active_connections.remove(connection)

manager = ConnectionManager()

# ---------------------- Global Variables ----------------------
current_session_id = None
monitoring_active = False
current_stats = {
    "ear": 0.0,
    "eye_state": "Center",
    "head_pose": {"yaw": 0.0, "pitch": 0.0},
    "alerts_count": 0,
    "session_duration": 0
}

# Message queue for communication between threads
message_queue = queue.Queue()

# ---------------------- Beep Function ----------------------
def beep():
    if platform.system() == "Windows":
        winsound.Beep(1000, 500)  # 1000Hz for 500ms
    else:
        print("\a")  # System beep for non-Windows

# ---------------------- MediaPipe Setup ----------------------
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Eye landmarks (MediaPipe face mesh indices)
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Eye corners for iris position calculation
LEFT_EYE_CORNERS = [33, 133]  # inner and outer corners
RIGHT_EYE_CORNERS = [362, 263]  # inner and outer corners

# Iris landmarks (approximate center points)
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]

def calculate_ear(eye_landmarks, landmarks):
    """Calculate Eye Aspect Ratio for drowsiness detection."""
    try:
        # Get eye landmark coordinates
        eye_points = []
        for idx in eye_landmarks:
            if idx < len(landmarks.landmark):
                x = landmarks.landmark[idx].x
                y = landmarks.landmark[idx].y
                eye_points.append([x, y])
        
        if len(eye_points) < 6:
            return 0.0
            
        eye_points = np.array(eye_points)
        
        # Calculate distances
        # Vertical distances
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal distance
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # EAR calculation
        if C > 0:
            ear = (A + B) / (2.0 * C)
            return ear
        return 0.0
    except:
        return 0.0

def get_iris_position(eye_corners, iris_landmarks, landmarks):
    """Calculate iris position relative to eye corners for distraction detection."""
    try:
        # Get eye corner coordinates
        if eye_corners[0] >= len(landmarks.landmark) or eye_corners[1] >= len(landmarks.landmark):
            return 0.5  # center position as default
            
        inner_corner = landmarks.landmark[eye_corners[0]]
        outer_corner = landmarks.landmark[eye_corners[1]]
        
        # Get iris center (approximate)
        iris_x = 0
        iris_y = 0
        valid_iris_points = 0
        
        for idx in iris_landmarks:
            if idx < len(landmarks.landmark):
                iris_x += landmarks.landmark[idx].x
                iris_y += landmarks.landmark[idx].y
                valid_iris_points += 1
        
        if valid_iris_points == 0:
            return 0.5
            
        iris_x /= valid_iris_points
        iris_y /= valid_iris_points
        
        # Calculate relative position (0 = inner corner, 1 = outer corner)
        eye_width = abs(outer_corner.x - inner_corner.x)
        if eye_width > 0:
            iris_ratio = (iris_x - inner_corner.x) / eye_width
            return max(0, min(1, iris_ratio))  # Clamp between 0 and 1
        
        return 0.5
    except:
        return 0.5

def get_head_pose(landmarks, img_shape):
    """Calculate head pose angles (yaw, pitch, roll)."""
    try:
        h, w = img_shape[:2]
        
        # 3D model points (generic face model)
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)
        
        # 2D image points from landmarks
        image_points = np.array([
            (landmarks.landmark[1].x * w, landmarks.landmark[1].y * h),     # Nose tip
            (landmarks.landmark[152].x * w, landmarks.landmark[152].y * h), # Chin
            (landmarks.landmark[226].x * w, landmarks.landmark[226].y * h), # Left eye left corner
            (landmarks.landmark[446].x * w, landmarks.landmark[446].y * h), # Right eye right corner
            (landmarks.landmark[57].x * w, landmarks.landmark[57].y * h),   # Left mouth corner
            (landmarks.landmark[287].x * w, landmarks.landmark[287].y * h)  # Right mouth corner
        ], dtype=np.float64)
        
        # Camera internals
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.zeros((4,1))
        
        # Solve PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs)
        
        if success:
            # Convert rotation vector to angles
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            angles, _, _, _, _, _ = cv2.RQDecomp3x3(rotation_matrix)
            
            yaw = angles[1]
            pitch = angles[0]
            roll = angles[2]
            
            return yaw, pitch, roll
        
        return 0.0, 0.0, 0.0
    except:
        return 0.0, 0.0, 0.0

# ---------------------- FastAPI App ----------------------
app = FastAPI(title="Driver Monitoring System", version="2.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_dashboard():
    """Serve the main dashboard."""
    try:
        with open("static/dashboard.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Dashboard not found. Please ensure static/dashboard.html exists.</h1>")

@app.get("/api/stats")
async def get_stats():
    """Get current monitoring statistics."""
    global current_stats, current_session_id
    
    # Update session duration if monitoring is active
    if monitoring_active and current_session_id:
        conn = sqlite3.connect('driver_monitor.db')
        cursor = conn.cursor()
        cursor.execute('SELECT start_time FROM sessions WHERE id = ?', (current_session_id,))
        result = cursor.fetchone()
        if result:
            start_time = datetime.fromisoformat(result[0])
            duration = (datetime.now() - start_time).total_seconds()
            current_stats["session_duration"] = int(duration)
        conn.close()
    
    return current_stats

@app.post("/api/start_monitoring")
async def start_monitoring():
    global monitoring_active, current_session_id
    
    if not monitoring_active:
        monitoring_active = True
        
        # Create new session in database
        conn = sqlite3.connect('driver_monitor.db')
        cursor = conn.cursor()
        cursor.execute('INSERT INTO sessions (start_time) VALUES (?)', (datetime.now(),))
        current_session_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Start monitoring thread
        monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        monitoring_thread.start()
        
        # Broadcast status update
        message = json.dumps({"type": "monitoring_started", "session_id": current_session_id})
        message_queue.put(message)
        
    return {"status": "started", "session_id": current_session_id}

@app.post("/api/stop_monitoring")
async def stop_monitoring():
    global monitoring_active
    monitoring_active = False
    return {"status": "stopped"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Check for messages from monitoring thread
            try:
                while True:
                    message = message_queue.get_nowait()
                    await manager.broadcast(message)
            except queue.Empty:
                pass
            
            # Keep connection alive
            try:
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
                
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        manager.disconnect(websocket)

# ---------------------- Monitoring Loop ----------------------
def monitoring_loop():
    global monitoring_active, current_stats
    global total_alerts, drowsy_alerts, distraction_alerts, head_alerts
    global ear_buffer, eye_state_buffer, yaw_vals, pitch_vals
    global drowsy_start, distraction_start, head_start
    
    # Initialize counters
    total_alerts = 0
    drowsy_alerts = 0
    distraction_alerts = 0
    head_alerts = 0
    
    # Buffers for stability
    ear_buffer = deque(maxlen=10)
    eye_state_buffer = deque(maxlen=15)
    yaw_vals = deque(maxlen=10)
    pitch_vals = deque(maxlen=10)
    
    # Alert timing
    drowsy_start = None
    distraction_start = None
    head_start = None
    
    # Thresholds
    EAR_THRESHOLD = 0.25
    DISTRACTION_THRESHOLD = 0.35  # How far from center is considered distracted
    HEAD_YAW_THRESHOLD = 25
    HEAD_PITCH_THRESHOLD = 20
    
    # Time thresholds (seconds)
    DROWSY_TIME_THRESHOLD = 2.0
    DISTRACTION_TIME_THRESHOLD = 3.0
    HEAD_TIME_THRESHOLD = 4.0
    
    print("🎥 Starting camera monitoring...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    if not cap.isOpened():
        print("❌ Error: Could not open camera")
        return
    
    # Initialize MediaPipe
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        print("🚀 Driver monitoring started successfully!")
        print("📋 Monitoring features:")
        print("   • Eye Aspect Ratio (EAR) for drowsiness")
        print("   • Iris position tracking for distraction")
        print("   • Head pose estimation for attention")
        print("   • Real-time alerts and logging")
        
        frame_count = 0
        
        while monitoring_active:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break
                
            frame_count += 1
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate metrics
                    left_ear = calculate_ear(LEFT_EYE[:6], face_landmarks)  # Use first 6 points for EAR
                    right_ear = calculate_ear(RIGHT_EYE[:6], face_landmarks)
                    avg_ear = (left_ear + right_ear) / 2.0
                    
                    # Get iris positions
                    left_iris_pos = get_iris_position(LEFT_EYE_CORNERS, LEFT_IRIS, face_landmarks)
                    right_iris_pos = get_iris_position(RIGHT_EYE_CORNERS, RIGHT_IRIS, face_landmarks)
                    avg_iris_pos = (left_iris_pos + right_iris_pos) / 2.0
                    
                    # Determine eye state based on iris position
                    if avg_iris_pos < 0.3:
                        eye_state = "Looking Left"
                    elif avg_iris_pos > 0.7:
                        eye_state = "Looking Right"
                    else:
                        eye_state = "Center"
                    
                    # Get head pose
                    yaw, pitch, roll = get_head_pose(face_landmarks, frame.shape)
                    
                    # Add to buffers for stability
                    ear_buffer.append(avg_ear)
                    eye_state_buffer.append(avg_iris_pos)
                    yaw_vals.append(yaw)
                    pitch_vals.append(pitch)
                    
                    # Calculate stable values
                    stable_ear = np.mean(ear_buffer) if ear_buffer else avg_ear
                    stable_iris = np.mean(eye_state_buffer) if eye_state_buffer else avg_iris_pos
                    stable_yaw = np.mean(yaw_vals) if yaw_vals else yaw
                    stable_pitch = np.mean(pitch_vals) if pitch_vals else pitch
                    
                    current_time = time.time()
                    
                    # ==================== DROWSINESS DETECTION ====================
                    if stable_ear < EAR_THRESHOLD:
                        if drowsy_start is None:
                            drowsy_start = current_time
                        elif current_time - drowsy_start >= DROWSY_TIME_THRESHOLD:
                            if drowsy_start is not None:  # Only alert once per drowsy period
                                drowsy_alerts += 1
                                total_alerts += 1
                                log_alert("drowsy", "high", stable_ear, eye_state, stable_yaw, stable_pitch)
                                beep()
                                print(f"⚠️  DROWSINESS ALERT! EAR: {stable_ear:.3f}")
                                
                                # Send WebSocket message
                                alert_msg = json.dumps({
                                    "type": "alert",
                                    "alert_type": "drowsy",
                                    "message": f"Drowsiness detected! EAR: {stable_ear:.3f}",
                                    "severity": "high",
                                    "timestamp": datetime.now().isoformat()
                                })
                                message_queue.put(alert_msg)
                                drowsy_start = None  # Reset to prevent continuous alerts
                    else:
                        drowsy_start = None
                    
                    # ==================== DISTRACTION DETECTION ====================
                    iris_deviation = abs(stable_iris - 0.5)  # Distance from center (0.5)
                    if iris_deviation > DISTRACTION_THRESHOLD:
                        if distraction_start is None:
                            distraction_start = current_time
                        elif current_time - distraction_start >= DISTRACTION_TIME_THRESHOLD:
                            if distraction_start is not None:
                                distraction_alerts += 1
                                total_alerts += 1
                                log_alert("distraction", "medium", stable_ear, eye_state, stable_yaw, stable_pitch)
                                print(f"⚠️  DISTRACTION ALERT! Looking away: {eye_state}")
                                
                                # Send WebSocket message
                                alert_msg = json.dumps({
                                    "type": "alert",
                                    "alert_type": "distraction",
                                    "message": f"Distraction detected! {eye_state}",
                                    "severity": "medium",
                                    "timestamp": datetime.now().isoformat()
                                })
                                message_queue.put(alert_msg)
                                distraction_start = None
                    else:
                        distraction_start = None
                    
                    # ==================== HEAD POSE DETECTION ====================
                    if abs(stable_yaw) > HEAD_YAW_THRESHOLD or abs(stable_pitch) > HEAD_PITCH_THRESHOLD:
                        if head_start is None:
                            head_start = current_time
                        elif current_time - head_start >= HEAD_TIME_THRESHOLD:
                            if head_start is not None:
                                head_alerts += 1
                                total_alerts += 1
                                log_alert("head_pose", "low", stable_ear, eye_state, stable_yaw, stable_pitch)
                                print(f"⚠️  HEAD POSE ALERT! Yaw: {stable_yaw:.1f}°, Pitch: {stable_pitch:.1f}°")
                                
                                # Send WebSocket message
                                alert_msg = json.dumps({
                                    "type": "alert",
                                    "alert_type": "head_pose",
                                    "message": f"Head pose alert! Yaw: {stable_yaw:.1f}°",
                                    "severity": "low",
                                    "timestamp": datetime.now().isoformat()
                                })
                                message_queue.put(alert_msg)
                                head_start = None
                    else:
                        head_start = None
                    
                    # Update current stats
                    current_stats.update({
                        "ear": round(stable_ear, 3),
                        "eye_state": eye_state,
                        "head_pose": {"yaw": round(stable_yaw, 1), "pitch": round(stable_pitch, 1)},
                        "alerts_count": total_alerts
                    })
                    
                    # Send periodic stats update via WebSocket
                    if frame_count % 30 == 0:  # Every 30 frames (~1 second at 30fps)
                        stats_msg = json.dumps({
                            "type": "stats_update",
                            "stats": current_stats,
                            "timestamp": datetime.now().isoformat()
                        })
                        message_queue.put(stats_msg)
                    
                    # Draw landmarks and info (for debugging - can be removed)
                    if frame_count % 10 == 0:  # Draw every 10th frame to reduce load
                        # Draw face mesh
                        mp_drawing.draw_landmarks(
                            frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                            None, mp_drawing_styles.get_default_face_mesh_contours_style())
                        
                        # Draw eye landmarks
                        for idx in LEFT_EYE[:6] + RIGHT_EYE[:6]:
                            x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                            y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        
                        # Draw info text
                        cv2.putText(frame, f"EAR: {stable_ear:.3f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Eye: {eye_state}", (10, 60), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Yaw: {stable_yaw:.1f}, Pitch: {stable_pitch:.1f}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Alerts: {total_alerts}", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Show frame (optional - comment out for headless operation)
                        cv2.imshow('Driver Monitor', frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
            
            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Update session end time
    if current_session_id:
        conn = sqlite3.connect('driver_monitor.db')
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE sessions 
            SET end_time = ?, total_alerts = ?, drowsy_alerts = ?, 
                distraction_alerts = ?, head_alerts = ?
            WHERE id = ?
        ''', (datetime.now(), total_alerts, drowsy_alerts, 
              distraction_alerts, head_alerts, current_session_id))
        conn.commit()
        conn.close()
    
    print("🛑 Driver monitoring stopped")

def log_alert(alert_type, severity, ear_value, eye_state, head_yaw, head_pitch):
    """Log alert to database."""
    if current_session_id:
        conn = sqlite3.connect('driver_monitor.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO alerts (session_id, alert_type, severity, ear_value, 
                              eye_state, head_yaw, head_pitch)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (current_session_id, alert_type, severity, ear_value, 
              eye_state, head_yaw, head_pitch))
        conn.commit()
        conn.close()

# ---------------------- Main Execution ----------------------
if __name__ == "__main__":
    print("🚗 Enhanced Driver Monitoring System (Stable Version)")
    print("📊 Features: MediaPipe + Database + Web Dashboard")
    print("🌐 Starting web server...")
    
    # Initialize database
    init_database()
    
    # Run FastAPI server
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")