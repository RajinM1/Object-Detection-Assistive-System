"""
Enhanced Object Detection Assistive System for Visually Impaired People
Features:
- Real-time camera support
- Configurable settings via config.json
- Performance optimization with GPU support
- Enhanced UI with statistics dashboard
- Portable (no hardcoded paths)
"""

import pyttsx3
from threading import Thread
from queue import PriorityQueue
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import sys
import json
import argparse
import winsound  # For beep sounds on Windows

# Global variables
last_spoken = {}
last_distances = {}
last_beep = {}
audio_queue = PriorityQueue()
config = {}
current_distances = {}  # Real-time distance tracking
sync_mode = True  # Perfect audio-visual sync mode
statistics = {
    'total_detections': 0,
    'closest_encounter': float('inf'),
    'closest_object': None,
    'session_start': time.time(),
    'objects_count': {}
}

# Object real-world widths in meters for accurate distance calculation
class_avg_sizes = {
    "person": {"width_ratio": 0.6},     # Average person shoulder width
    "car": {"width_ratio": 1.8},       # Average car width
    "truck": {"width_ratio": 2.5},     # Average truck width
    "bicycle": {"width_ratio": 0.6},   # Bicycle width
    "motorcycle": {"width_ratio": 0.8}, # Motorcycle width
    "bus": {"width_ratio": 2.5},       # Bus width
    "traffic light": {"width_ratio": 0.3}, # Traffic light width
    "stop sign": {"width_ratio": 0.6}, # Stop sign width
    "bench": {"width_ratio": 1.5},     # Bench width
    "cat": {"width_ratio": 0.3},       # Cat width
    "dog": {"width_ratio": 0.4},       # Dog width
    "chair": {"width_ratio": 0.5},     # Chair width
    "bottle": {"width_ratio": 0.08},   # Bottle width
    "book": {"width_ratio": 0.2},      # Book width
}

def load_config(config_path='config.json'):
    """Load configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠️ Config file not found at {config_path}. Using defaults.")
        return get_default_config()
    except json.JSONDecodeError as e:
        print(f"⚠️ Error parsing config file: {e}. Using defaults.")
        return get_default_config()

def get_default_config():
    """Return default configuration"""
    return {
        "video": {
            "source": "camera",
            "camera_id": 0,
            "video_file": "test_video2.mp4",
            "output_file": "output_with_boxes.avi",
            "display_window": True,
            "save_output": True,
            "fps": 20
        },
        "model": {
            "path": "gpModel.pt",
            "confidence_threshold": 0.5,
            "use_gpu": True
        },
        "audio": {
            "enabled": True,
            "start_on_launch": False,
            "speech_rate": 200,
            "volume": 1.0,
            "language": "en"
        },
        "detection": {
            "max_distance": 12.5,
            "frame_skip": 0,
            "blur_faces": True
        },
        "alerts": {
            "beep_frequencies": {
                "left": 300,
                "center": 500,
                "right": 800
            },
            "cooldown_times": {
                "danger_zone": 0.3,
                "very_close": 0.6,
                "close": 1.0,
                "far": 1.5
            },
            "distance_zones": {
                "danger": 1.5,
                "very_close": 3.0,
                "close": 5.0,
                "far": 8.0
            },
            "object_priorities": {
                "person": 2,
                "car": 2,
                "truck": 12,
                "bus": 12,
                "motorcycle": 3,
                "bicycle": 3,
                "traffic light": 4,
                "stop sign": 4
            }
        },
        "display": {
            "show_distance": True,
            "show_boxes": True,
            "show_statistics": True,
            "show_controls": True,
            "font_scale": 0.6,
            "box_thickness": 2
        }
    }

def play_beep(position, distance):
    """Play directional beep based on position and distance"""
    try:
        frequency = config['alerts']['beep_frequencies'].get(position, 500)
        
        # Duration based on distance (closer = shorter beeps = more urgent)
        zones = config['alerts']['distance_zones']
        if distance <= zones['danger']:
            duration = 50  # Very short beep (urgent)
        elif distance <= zones['very_close']:
            duration = 100
        elif distance <= zones['close']:
            duration = 150
        else:
            duration = 200
        
        winsound.Beep(frequency, duration)
    except Exception as e:
        pass  # Silently fail if beep doesn't work

def speak_thread():
    """Background thread for text-to-speech - CONTINUOUS VOICE FEEDBACK"""
    print("🔊 Audio System Started")
    print("📍 TTS Engine Ready")
    print("🎤 CONTINUOUS VOICE MODE ENABLED")
    print("=" * 60)
    
    while True:
        if not audio_queue.empty():
            priority, timestamp, label, distance, position, motion = audio_queue.get()
            
            # Always use the most current distance for perfect sync
            if label in current_distances:
                current_distance = current_distances[label]
                if abs(current_distance - distance) > 0.1:  # Only update if significantly different
                    print(f"   🔄 SYNC UPDATE: Using current {current_distance:.1f}m instead of queued {distance:.1f}m")
                    distance = current_distance
            
            # Clear, detailed announcement with EXACT distance
            dist_str = f"{distance:.1f}"
            
            # Improved message format - clearer and more natural
            if "DANGER" in motion:
                message = f"DANGER! {label} {dist_str} meters {position}"
            elif motion == "very close":
                message = f"Warning! {label} {dist_str} meters {position}"
            elif motion == "approaching":
                message = f"{label} approaching {dist_str} meters {position}"
            else:
                message = f"{label} {dist_str} meters {position}"
            
            print(f"🎤 QUEUED TO SPEAK: {message}")
            print(f"   📊 DEBUG: Distance={distance:.1f}m, Position={position}, Motion={motion}")
            
            # Create fresh TTS engine for each speech to avoid corruption
            try:
                print(f"   ▶️ NOW SPEAKING...")
                engine = pyttsx3.init()
                engine.setProperty('rate', 200)  # Slower, clearer speech
                engine.setProperty('volume', config['audio']['volume'])
                engine.say(message)
                engine.runAndWait()
                engine.stop()
                del engine
                print(f"   ✅ FINISHED SPEAKING")
            except Exception as e:
                print(f"   ❌ TTS ERROR: {e}")
                print(f"   💡 You should hear BEEPS only")
        else:
            time.sleep(0.05)  # Slightly longer sleep when queue empty

def beep_thread():
    """Background thread for continuous beeping based on proximity"""
    while True:
        current_time = time.time()
        
        if last_distances:
            closest_label = min(last_distances, key=last_distances.get)
            closest_distance = last_distances[closest_label]
            
            zones = config['alerts']['distance_zones']
            cooldowns = config['alerts']['cooldown_times']
            
            # Determine beep interval based on distance
            if closest_distance <= zones['very_close']:
                beep_interval = 0.15  # VERY fast beeping for DANGER!
                should_beep = True
            elif closest_distance <= zones['close']:
                beep_interval = 0.3  # Fast beeping for very close
                should_beep = True
            elif closest_distance <= zones['close']:
                beep_interval = 0.6  # Medium beeping
                should_beep = True
            elif closest_distance <= zones['far']:
                beep_interval = 1.0  # Slow beeping
                should_beep = True
            else:
                should_beep = False
            
            if should_beep:
                time_since_beep = current_time - last_beep.get('continuous', 0)
                if time_since_beep >= beep_interval:
                    play_beep("center", closest_distance)
                    last_beep['continuous'] = current_time
        
        time.sleep(0.05)

def calculate_distance(box, frame_width, label):
    """Calculate distance to object using improved perspective projection"""
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    
    # Improved distance calculation with better scaling
    if label in class_avg_sizes:
        # Use known object sizes for better accuracy
        real_width = class_avg_sizes[label]["width_ratio"]
        focal_length = frame_width * 0.8  # Adjusted focal length
        distance = (real_width * focal_length) / (object_width + 1e-6)
    else:
        # Fallback calculation
        distance = (frame_width * 0.3) / (object_width + 1e-6)
    
    # Clamp distance to reasonable range
    distance = max(0.1, min(distance, 50.0))
    return round(distance, 1)

def get_position(frame_width, box):
    """Determine object position with more natural language"""
    center_x = box[0] + (box[2] - box[0]) / 2  # Object center
    
    if center_x < frame_width * 0.3:
        return "to your left"
    elif center_x < frame_width * 0.7:
        return "ahead"
    else:
        return "to your right"

def blur_person(image, box):
    """Blur face region for privacy"""
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    top_region = image[y:y+int(0.08 * h), x:x+w]
    if top_region.size > 0:
        blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
        image[y:y+int(0.08 * h), x:x+w] = blurred_top_region
    return image

def update_statistics(label, distance):
    """Update session statistics"""
    statistics['total_detections'] += 1
    statistics['objects_count'][label] = statistics['objects_count'].get(label, 0) + 1
    
    if distance < statistics['closest_encounter']:
        statistics['closest_encounter'] = distance
        statistics['closest_object'] = label

def draw_statistics(frame):
    """Draw statistics overlay on frame"""
    if not config['display']['show_statistics']:
        return frame
    
    # Statistics panel (top right)
    panel_x = frame.shape[1] - 350
    panel_y = 10
    panel_height = 150
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (panel_x, panel_y), (frame.shape[1] - 10, panel_y + panel_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Statistics text
    session_time = int(time.time() - statistics['session_start'])
    minutes = session_time // 60
    seconds = session_time % 60
    
    stats_text = [
        f"Session: {minutes:02d}:{seconds:02d}",
        f"Detections: {statistics['total_detections']}",
        f"Closest: {statistics['closest_object'] or 'N/A'}",
        f"  @ {statistics['closest_encounter']:.1f}m" if statistics['closest_encounter'] != float('inf') else "  @ N/A",
        f"FPS: {cv2.getTickFrequency() / (cv2.getTickCount()):.1f}"
    ]
    
    y_offset = panel_y + 25
    for text in stats_text:
        cv2.putText(frame, text, (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25
    
    return frame

def get_video_source(args, config):
    """Determine and open video source (camera or file)"""
    if args.camera:
        source = args.camera_id if args.camera_id is not None else config['video']['camera_id']
        print(f"📷 Using camera: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {source}")
            sys.exit(1)
        return cap, "camera"
    elif args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"❌ Error: Video file not found: {video_path}")
            sys.exit(1)
        print(f"🎥 Using video file: {video_path}")
        cap = cv2.VideoCapture(video_path)
        return cap, "file"
    else:
        # Use config default
        if config['video']['source'] == 'camera':
            source = config['video']['camera_id']
            print(f"📷 Using camera: {source}")
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                print(f"⚠️ Camera {source} not available. Trying video file fallback...")
                if os.path.exists(config['video']['video_file']):
                    cap = cv2.VideoCapture(config['video']['video_file'])
                    return cap, "file"
                else:
                    print(f"❌ Error: Could not open camera or find video file")
                    sys.exit(1)
            return cap, "camera"
        else:
            video_path = config['video']['video_file']
            if not os.path.exists(video_path):
                print(f"❌ Error: Video file not found: {video_path}")
                sys.exit(1)
            print(f"🎥 Using video file: {video_path}")
            cap = cv2.VideoCapture(video_path)
            return cap, "file"

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Enhanced Object Detection Assistive System for Visually Impaired People',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use camera (default)
  python main_enhanced.py --camera
  
  # Use specific camera
  python main_enhanced.py --camera --camera-id 1
  
  # Use video file
  python main_enhanced.py --video path/to/video.mp4
  
  # Custom model and config
  python main_enhanced.py --model custom_model.pt --config custom_config.json
  
  # Disable visual display (audio only)
  python main_enhanced.py --no-display
        """
    )
    
    parser.add_argument('--camera', action='store_true', 
                       help='Use camera as input source')
    parser.add_argument('--camera-id', type=int, default=None,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--video', type=str, 
                       help='Path to video file')
    parser.add_argument('--model', type=str, 
                       help='Path to YOLO model file (default: from config)')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file (default: config.json)')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable visual display window')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save output video')
    parser.add_argument('--gpu', action='store_true',
                       help='Force GPU usage')
    parser.add_argument('--cpu', action='store_true',
                       help='Force CPU usage')
    
    return parser.parse_args()

def main():
    global config, last_spoken, last_distances, last_beep, audio_queue, current_distances, sync_mode
    
    # Parse arguments
    args = parse_arguments()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine device (GPU/CPU)
    import torch
    if args.gpu:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            print("⚠️ GPU requested but CUDA not available. Using CPU.")
    elif args.cpu:
        device = 'cpu'
    else:
        device = 'cuda' if torch.cuda.is_available() and config['model']['use_gpu'] else 'cpu'
    
    print(f"🖥️  Device: {device.upper()}")
    
    # Load model
    model_path = args.model if args.model else config['model']['path']
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    model.to(device)
    
    # Get video source
    cap, source_type = get_video_source(args, config)
    
    # Output video setup
    out = None
    if config['video']['save_output'] and not args.no_save:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = config['video']['fps']
        output_file = config['video']['output_file']
        out = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))
        print(f"💾 Output will be saved to: {output_file}")
    
    # Start audio threads
    if config['audio']['enabled']:
        Thread(target=speak_thread, daemon=True).start()
        Thread(target=beep_thread, daemon=True).start()
    
    # Print system info
    print(f"\n{'='*60}")
    print(f"🎥 Enhanced Object Detection System for Visually Impaired")
    print(f"{'='*60}")
    print(f"📊 Source: {source_type.upper()}")
    print(f"🔊 Audio: {'ENABLED' if config['audio']['enabled'] else 'DISABLED'}")
    print(f"📺 Display: {'ENABLED' if config['video']['display_window'] and not args.no_display else 'DISABLED'}")
    print(f"💾 Save Output: {'YES' if out else 'NO'}")
    print(f"\n🎮 CONTROLS:")
    print(f"   [S] - START audio system (SYNC MODE)")
    print(f"   [E] - STOP/END audio system")
    print(f"   [P] - Pause/Resume video")
    print(f"   [Q] - Quit program")
    print(f"   [R] - Reset statistics")
    print(f"   [T] - Toggle sync mode (current: {'ON' if sync_mode else 'OFF'})")
    print(f"\n🚨 SAFETY FEATURE:")
    print(f"   Emergency Override: ENABLED")
    print(f"   System will AUTO-SPEAK for extreme dangers (≤1m)")
    print(f"   Even when audio is STOPPED!")
    print(f"\n⏸️  System STOPPED - Press 'S' to START\n")
    print(f"{'='*60}\n")
    
    # Main loop variables
    frame_count = 0
    pause = False
    system_active = config['audio']['start_on_launch']
    
    if system_active:
        print("✅ Audio system started automatically")
    
    try:
        while cap.isOpened():
            if not pause:
                ret, frame = cap.read()
                if not ret:
                    if source_type == "camera":
                        print("⚠️ Camera feed interrupted")
                        break
                    else:
                        print("📹 End of video file reached")
                        break
                
                frame_count += 1
                
                # Frame skipping for performance
                if config['detection']['frame_skip'] > 0 and frame_count % (config['detection']['frame_skip'] + 1) != 0:
                    continue
                
                # Object detection
                results = model.predict(frame, verbose=False, conf=config['model']['confidence_threshold'])
                result = results[0]
                objects_detected = []
                
                for box in result.boxes:
                    label = result.names[box.cls[0].item()]
                    confidence = box.conf[0].item()  # Get confidence score
                    cords = [round(x) for x in box.xyxy[0].tolist()]
                    distance = calculate_distance(box, frame.shape[1], label)
                    
                    # Update real-time distance tracking
                    current_distances[label] = distance
                    
                    # Perfect sync mode - speak immediately if distance changed significantly
                    if sync_mode and system_active and label in last_distances:
                        prev_dist = last_distances.get(label, 0)
                        if abs(distance - prev_dist) > 0.3:  # Significant change
                            current_time = time.time()
                            time_since_last = current_time - last_spoken.get(label, 0)
                            if time_since_last >= 0.5:  # Minimum interval
                                position = get_position(frame.shape[1], cords)
                                motion = "approaching" if distance < prev_dist else "moving away" if distance > prev_dist else "ahead"
                                
                                # Clear queue and add immediate announcement
                                while not audio_queue.empty():
                                    try:
                                        audio_queue.get_nowait()
                                    except:
                                        break
                                
                                audio_queue.put((0, current_time, label, distance, position, motion))
                                last_spoken[label] = current_time
                                print(f"   🎯 SYNC SPEAK: {label} {distance:.1f}m {position} (was {prev_dist:.1f}m)")
                    
                    # Update statistics
                    update_statistics(label, distance)
                    
                    # Blur faces if enabled
                    if label == "person" and config['detection']['blur_faces']:
                        frame = blur_person(frame, box)
                        color = (0, 255, 0)  # Green
                    elif label == "car":
                        color = (0, 255, 255)  # Yellow
                    elif label == "truck":
                        color = (255, 165, 0)  # Orange
                    elif label == "bus":
                        color = (255, 0, 255)  # Magenta
                    elif label in class_avg_sizes:
                        color = (255, 0, 0)  # Red
                    else:
                        continue
                    
                    # Draw bounding box
                    if config['display']['show_boxes']:
                        cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), 
                                    color, config['display']['box_thickness'])
                        if config['display']['show_distance']:
                            # Show label with distance and confidence
                            label_text = f"{label} - {distance:.1f}m ({confidence*100:.0f}%)"
                            cv2.putText(frame, label_text, (cords[0], cords[1] - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, config['display']['font_scale'], color, 2)
                    
                    # Collect objects for announcement (filter out very far objects)
                    if distance <= config['detection']['max_distance'] and distance >= 0.2:
                        position = get_position(frame.shape[1], cords)
                        priority = config['alerts']['object_priorities'].get(label, 5)
                        objects_detected.append((priority, distance, label, position, cords))
                        # Only print detection occasionally to reduce spam
                        if frame_count % 30 == 0:
                            print(f"   🔍 DETECTED: {label} at {distance:.1f}m {position} (confidence: {confidence*100:.0f}%)")
                
                # Process detections and queue announcements
                # Check for URGENT situations (bypass system_active for extreme danger only)
                urgent_detected = any(dist <= 1.0 for _, dist, _, _, _ in objects_detected) if objects_detected else False
                
                if objects_detected and config['audio']['enabled'] and (system_active or urgent_detected):
                    objects_detected.sort(key=lambda x: (x[0], x[1]))
                    
                    current_time = time.time()
                    announced_count = 0
                    
                    for priority, distance, label, position, cords in objects_detected:
                        # Determine cooldown based on distance
                        zones = config['alerts']['distance_zones']
                        cooldowns = config['alerts']['cooldown_times']
                        
                        if distance <= zones['danger']:
                            required_cooldown = cooldowns['danger_zone']
                        elif distance <= zones['very_close']:
                            required_cooldown = cooldowns['very_close']
                        elif distance <= zones['close']:
                            required_cooldown = cooldowns['close']
                        else:
                            required_cooldown = cooldowns['far']
                        
                        time_since_last = current_time - last_spoken.get(label, 0)
                        
                        # Check if object is approaching
                        prev_distance = last_distances.get(label, None)
                        is_approaching = False
                        if prev_distance is not None and distance < prev_distance - 0.2:
                            is_approaching = True
                        
                        # Bypass cooldown for critical situations (approaching or first detection)
                        bypass_cooldown = is_approaching or (prev_distance is None)
                        
                        # SAFETY OVERRIDE: Only for extreme danger (≤1m) when system is stopped
                        is_urgent = distance <= 1.0  # More restrictive emergency override
                        # Real-time updates - much faster intervals for perfect sync
                        min_interval = 0.1 if is_urgent else 0.2  # Very fast updates
                        should_announce = (system_active and time_since_last >= min_interval) or (is_urgent and time_since_last >= 0.1)
                        
                        if should_announce:
                            # Determine motion
                            if prev_distance is not None:
                                if distance < prev_distance - 0.2:
                                    motion = "approaching"
                                elif distance > prev_distance + 0.2:
                                    motion = "moving away"
                                else:
                                    motion = "ahead"
                            else:
                                motion = "ahead"
                            
                            # Override for very close objects
                            if distance <= zones['danger']:
                                motion = "DANGER very close"
                            elif distance <= zones['very_close']:
                                motion = "very close"
                            
                            # Update tracking with current distance
                            last_distances[label] = distance
                            last_spoken[label] = current_time
                            
                            # Adjust priority for closer/approaching objects
                            adjusted_priority = priority
                            if distance <= 2:
                                adjusted_priority = 0
                            elif is_approaching:
                                adjusted_priority = 1
                            
                            # Clear ALL old announcements to ensure only current distances
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except:
                                    break
                            
                            # Queue only the current distance - immediate sync
                            print(f"   📤 QUEUEING CURRENT: {label} at {distance:.1f}m {position} [Motion: {motion}]")
                            audio_queue.put((adjusted_priority, current_time, label, distance, position, motion))
                            
                            # Play directional beep (extract direction from position text)
                            beep_direction = "left" if "left" in position else "right" if "right" in position else "center"
                            play_beep(beep_direction, distance)
                            
                            # Show alert with urgency indicator (reduce spam)
                            if time_since_last >= 1.0:  # Only print alert once per second per object
                                alert_type = "🚨 URGENT" if is_urgent and not system_active else "⚠️ DANGER" if distance <= 1.5 else "➕"
                                emergency_msg = " [EMERGENCY OVERRIDE - AUTO SPEAKING!]" if is_urgent and not system_active else ""
                                print(f"{alert_type} Queued: {label} at {distance:.1f}m ({position}) [Motion: {motion}]{emergency_msg}")
                            
                            announced_count += 1
                            max_announcements = 1  # Only one announcement per frame for perfect sync
                            if announced_count >= max_announcements:
                                break
                
                # Draw UI elements
                if config['video']['display_window'] and not args.no_display:
                    # System status with sync mode indicator
                    if system_active:
                        status_color = (0, 255, 0)
                        status_text = f"AUDIO: ACTIVE {'(SYNC)' if sync_mode else '(NORMAL)'}"
                        cv2.rectangle(frame, (10, 10), (300, 55), (0, 180, 0), -1)
                        cv2.rectangle(frame, (10, 10), (300, 55), (0, 255, 0), 3)
                    else:
                        status_color = (0, 0, 255)
                        status_text = "AUDIO: STOPPED"
                        cv2.rectangle(frame, (10, 10), (300, 55), (0, 0, 180), -1)
                        cv2.rectangle(frame, (10, 10), (300, 55), (0, 0, 255), 3)
                    
                    cv2.putText(frame, status_text, (20, 40), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 75), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    if objects_detected:
                        cv2.putText(frame, f"Tracking: {len(objects_detected)} objects", (10, 100), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                        
                        closest = objects_detected[0]
                        cv2.putText(frame, f"Closest: {closest[2]} at {closest[1]:.1f}m ({closest[3]})", 
                                  (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)
                    
                    # Speaking indicator (show even during emergency override)
                    if not audio_queue.empty():
                        # Red for normal, Orange for emergency
                        indicator_color = (0, 140, 255) if urgent_detected and not system_active else (0, 0, 255)
                        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, indicator_color, -1)
                        speaking_text = "🚨 EMERGENCY" if urgent_detected and not system_active else "SPEAKING"
                        cv2.putText(frame, speaking_text, (frame.shape[1] - 150, 40), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, indicator_color, 2)
                    
                    # Statistics
                    frame = draw_statistics(frame)
                    
                    # Control panel
                    if config['display']['show_controls']:
                        control_y = frame.shape[0] - 60
                        cv2.rectangle(frame, (0, control_y - 10), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
                        cv2.putText(frame, "Controls: [S]Start  [E]Stop  [P]Pause  [Q]Quit  [R]Reset  [T]Sync", 
                                  (10, control_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    cv2.imshow('Enhanced Object Detection for Visually Impaired', frame)
                
                # Save output video
                if out:
                    out.write(frame)
            
            # Handle keyboard input
            if config['video']['display_window'] and not args.no_display:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("\n⏹️ Quit requested by user")
                    break
                elif key == ord('s') or key == ord('S'):
                    if not system_active:
                        system_active = True
                        print("\n✅ AUDIO SYSTEM STARTED - Listening for objects...")
                elif key == ord('e') or key == ord('E'):
                    if system_active:
                        system_active = False
                        last_spoken.clear()
                        last_distances.clear()
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except:
                                break
                        print("\n⏹️ AUDIO SYSTEM STOPPED - Press 'S' to restart")
                elif key == ord('p') or key == ord('P'):
                    pause = not pause
                    if pause:
                        print("⏸️ Video PAUSED - Press 'P' to resume")
                    else:
                        print("▶️ Video RESUMED")
                elif key == ord('r') or key == ord('R'):
                    statistics['total_detections'] = 0
                    statistics['closest_encounter'] = float('inf')
                    statistics['closest_object'] = None
                    statistics['session_start'] = time.time()
                    statistics['objects_count'] = {}
                    print("📊 Statistics RESET")
                elif key == ord('t') or key == ord('T'):
                    sync_mode = not sync_mode
                    print(f"\n🔄 SYNC MODE: {'ENABLED' if sync_mode else 'DISABLED'}")
                    print(f"   Audio will {'match screen updates' if sync_mode else 'use normal cooldowns'}")
            else:
                # If no display, add small delay to prevent busy loop
                time.sleep(0.001)
    
    except KeyboardInterrupt:
        print("\n⏹️ Interrupted by user (Ctrl+C)")
    
    finally:
        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        print(f"\n{'='*60}")
        print("✅ Processing complete!")
        if out:
            print(f"📹 Output video saved to: {config['video']['output_file']}")
        print(f"🎬 Total frames processed: {frame_count}")
        print(f"\n📊 Session Statistics:")
        print(f"   Total detections: {statistics['total_detections']}")
        print(f"   Closest encounter: {statistics['closest_object']} at {statistics['closest_encounter']:.1f}m")
        print(f"   Objects detected: {dict(statistics['objects_count'])}")
        print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

