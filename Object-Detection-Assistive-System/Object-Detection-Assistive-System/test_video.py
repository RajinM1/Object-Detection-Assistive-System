#!/usr/bin/env python3
"""
Simple Video Testing Script
Test object detection on test_video2.mp4 with audio feedback
"""

import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time
import os

# Check if video file exists
VIDEO_PATH = "test_video2.mp4"
MODEL_PATH = "gpModel.pt"

if not os.path.exists(VIDEO_PATH):
    print(f"❌ Video file not found: {VIDEO_PATH}")
    exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model file not found: {MODEL_PATH}")
    exit(1)

# Object sizes for distance calculation
class_avg_sizes = {
    "person": {"width_ratio": 0.6},
    "car": {"width_ratio": 1.8},
    "truck": {"width_ratio": 2.5},
    "cat": {"width_ratio": 0.3},
    "dog": {"width_ratio": 0.4},
}

def calculate_distance(box, frame_width, label):
    """Calculate distance to object"""
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    
    if label in class_avg_sizes:
        real_width = class_avg_sizes[label]["width_ratio"]
        focal_length = frame_width * 0.8
        distance = (real_width * focal_length) / (object_width + 1e-6)
    else:
        distance = (frame_width * 0.3) / (object_width + 1e-6)
    
    distance = max(0.1, min(distance, 50.0))
    return round(distance, 1)

def get_position(frame_width, box):
    """Get object position"""
    center_x = box[0] + (box[2] - box[0]) / 2
    
    if center_x < frame_width * 0.3:
        return "to your left"
    elif center_x < frame_width * 0.7:
        return "ahead"
    else:
        return "to your right"

def speak_distance(label, distance, position):
    """Speak the detection with fresh TTS engine"""
    message = f"{label} {distance:.1f} meters {position}"
    print(f"🎤 SPEAKING: {message}")
    
    try:
        # Create fresh engine each time
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)
        engine.setProperty('volume', 1.0)
        engine.say(message)
        engine.runAndWait()
        engine.stop()
        del engine
        print(f"   ✅ Finished speaking")
    except Exception as e:
        print(f"   ❌ TTS Error: {e}")
        # Simple fallback - just print the message
        print(f"   💬 Fallback: {message}")

# Load model and video
print("🤖 Loading YOLO model...")
model = YOLO(MODEL_PATH)

print(f"🎥 Opening video: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Could not open video file")
    exit(1)

# Get video info
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = total_frames / fps

print(f"📊 Video Info:")
print(f"   Total frames: {total_frames}")
print(f"   FPS: {fps:.1f}")
print(f"   Duration: {duration:.1f} seconds")
print(f"\n🎮 CONTROLS:")
print(f"   [SPACE] - Manual speak (instant)")
print(f"   [Q] - Quit")
print(f"   [P] - Pause/Resume")
print(f"\n🔊 AUTO AUDIO: Enabled (every 2 seconds)")
print(f"\n▶️ Starting video playback...\n")

frame_count = 0
paused = False
last_speak_time = 0
last_spoken_objects = {}  # Track when each object was last announced

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("📹 End of video reached")
            break
        
        frame_count += 1
        
        # Run detection
        results = model.predict(frame, verbose=False, conf=0.5)
        result = results[0]
        
        detections = []
        
        # Process detections
        for box in result.boxes:
            label = result.names[box.cls[0].item()]
            confidence = box.conf[0].item()
            coords = [round(x) for x in box.xyxy[0].tolist()]
            distance = calculate_distance(box, frame.shape[1], label)
            position = get_position(frame.shape[1], coords)
            
            detections.append({
                'label': label,
                'distance': distance,
                'position': position,
                'confidence': confidence,
                'coords': coords
            })
            
            # Draw bounding box
            color = (0, 255, 0) if label == "person" else (0, 255, 255) if label == "car" else (255, 0, 0)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color, 2)
            
            # Draw label with distance
            label_text = f"{label} - {distance:.1f}m ({confidence*100:.0f}%)"
            cv2.putText(frame, label_text, (coords[0], coords[1] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Show frame info
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Automatic audio announcements
        current_time = time.time()
        if detections:
            closest = min(detections, key=lambda x: x['distance'])
            cv2.putText(frame, f"Closest: {closest['label']} at {closest['distance']:.1f}m", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Auto-speak closest object every 2 seconds
            object_key = f"{closest['label']}_{closest['position']}"
            time_since_spoken = current_time - last_spoken_objects.get(object_key, 0)
            
            if time_since_spoken >= 2.0:  # Speak every 2 seconds
                print(f"\n🔊 AUTO ANNOUNCEMENT:")
                speak_distance(closest['label'], closest['distance'], closest['position'])
                last_spoken_objects[object_key] = current_time
        
        # Show controls
        cv2.putText(frame, "AUTO AUDIO ON - [SPACE]Manual [P]Pause [Q]Quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Video Test - Object Detection', frame)
    
    # Handle keyboard input
    key = cv2.waitKey(30) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\n⏹️ Quit requested")
        break
    elif key == ord('p') or key == ord('P'):
        paused = not paused
        print(f"{'⏸️ PAUSED' if paused else '▶️ RESUMED'}")
    elif key == ord(' '):  # Spacebar
        current_time = time.time()
        if current_time - last_speak_time > 0.3:  # Even faster response
            if detections:
                closest = min(detections, key=lambda x: x['distance'])
                print(f"\n🎯 Manual speech triggered:")
                speak_distance(closest['label'], closest['distance'], closest['position'])
                last_speak_time = current_time
            else:
                print("🔍 No objects detected to speak")

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("\n✅ Video test completed!")