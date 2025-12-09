import pyttsx3
from threading import Thread
from queue import Queue, PriorityQueue
from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import winsound  # For beep sounds on Windows

# Define paths (relative to current directory - now portable!)
MODEL_PATH = "gpModel.pt"
VIDEO_PATH = "test_video2.mp4"

# Audio settings
last_spoken = {}
last_distances = {}
last_beep = {}

# Priority queue for announcements (lower number = higher priority)
audio_queue = PriorityQueue()

# Object priorities (lower = more important)
OBJECT_PRIORITIES = {
    "person": 2,
    "car": 2,
    "truck": 12,
    "bus": 12,
    "motorcycle": 3,
    "bicycle": 3,
    "traffic light": 4,
    "stop sign": 4,
}

# Beep frequencies for spatial audio
BEEP_FREQUENCIES = {
    "left": 300,    # Low pitch
    "center": 500,  # Mid pitch
    "right": 800    # High pitch
}

def play_beep(position, distance):
    """Play directional beep based on position and distance"""
    try:
        frequency = BEEP_FREQUENCIES.get(position, 500)

        # Duration based on distance (closer = shorter beeps = more urgent)
        if distance <= 1:
            duration = 50  # Very short beep (urgent)
        elif distance <= 2:
            duration = 100
        elif distance <= 5:
            duration = 150
        else:
            duration = 200

        winsound.Beep(frequency, duration)
    except Exception as e:
        pass  # Silently fail if beep doesn't work

def speak_thread():
    """Background thread for text-to-speech - CONTINUOUS VOICE FEEDBACK"""
    # Initialize TTS engine ONCE at start
    engine = None

    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 200)  # Slower for better clarity
        engine.setProperty('volume', 1.0)
        print("🔊 Audio System Started")
        print("📍 TTS Engine Ready")
        print("🎤 CONTINUOUS VOICE MODE ENABLED")
        print("=" * 60)
    except Exception as e:
        print(f"❌ CRITICAL: Audio engine failed to initialize: {e}")
        print("You will NOT hear voice - only beeps!")
        print("=" * 60)

    while True:
        if not audio_queue.empty():
            priority, timestamp, label, distance, position, motion = audio_queue.get()

            # Clear, detailed announcement
            dist_str = f"{distance:.1f}" if distance < 10 else f"{int(distance)}"

            # Message format with clear information
            if "DANGER" in motion:
                message = f"DANGER! {label} on your {position}, {dist_str} meters"
            elif motion == "very close":
                message = f"Warning! {label} on your {position}, {dist_str} meters"
            elif motion == "approaching":
                message = f"{label} approaching from {position}, {dist_str} meters"
            else:
                message = f"{label} on your {position}, {dist_str} meters"

            print(f"🎤 QUEUED TO SPEAK: {message}")

            # CRITICAL: Actually speak the message
            if engine is not None:
                try:
                    print(f"   ▶️ NOW SPEAKING...")
                    engine.say(message)
                    engine.runAndWait()
                    print(f"   ✅ FINISHED SPEAKING")
                except Exception as e:
                    print(f"   ❌ TTS ERROR: {e}")
                    print(f"   🔄 Reinitializing engine...")
                    try:
                        engine.stop()
                    except:
                        pass
                    try:
                        engine = pyttsx3.init()
                        engine.setProperty('rate', 200)
                        engine.setProperty('volume', 1.0)
                        print(f"   ✅ Engine reinitialized")
                    except Exception as e2:
                        print(f"   ❌ Reinitialization failed: {e2}")
                        engine = None
            else:
                print(f"   ❌ NO TTS ENGINE - Cannot speak!")
                print(f"   💡 You should hear BEEPS only")
        else:
            time.sleep(0.05)  # Slightly longer sleep when queue empty

def beep_thread():
    """Background thread for continuous beeping based on proximity"""
    while True:
        current_time = time.time()

        # Get closest object info from last_spoken
        if last_distances:
            # Find closest object
            closest_label = min(last_distances, key=last_distances.get)
            closest_distance = last_distances[closest_label]

            # VERY aggressive beep rate for danger zones
            if closest_distance <= 2:
                beep_interval = 0.15  # VERY fast beeping for DANGER!
                should_beep = True
            elif closest_distance <= 3:
                beep_interval = 0.3  # Fast beeping for very close
                should_beep = True
            elif closest_distance <= 5:
                beep_interval = 0.6  # Medium beeping
                should_beep = True
            elif closest_distance <= 8:
                beep_interval = 1.0  # Slow beeping
                should_beep = True
            else:
                should_beep = False

            if should_beep:
                # Check if enough time has passed since last beep
                time_since_beep = current_time - last_beep.get('continuous', 0)
                if time_since_beep >= beep_interval:
                    # Determine position of closest object
                    # (This is simplified - you'd need to track position too)
                    play_beep("center", closest_distance)
                    last_beep['continuous'] = current_time

        time.sleep(0.05)  # Check more frequently

# Start audio threads
Thread(target=speak_thread, daemon=True).start()
Thread(target=beep_thread, daemon=True).start()

# Calculate distance
def calculate_distance(box, frame_width, label):
    object_width = box.xyxy[0, 2].item() - box.xyxy[0, 0].item()
    if label in class_avg_sizes:
        object_width *= class_avg_sizes[label]["width_ratio"]
    distance = (frame_width * 0.5) / np.tan(np.radians(70 / 2)) / (object_width + 1e-6)
    return round(distance, 2)

# Get object position
def get_position(frame_width, box):
    if box[0] < frame_width // 3:
        return "left"
    elif box[0] < 2 * (frame_width // 3):
        return "center"
    else:
        return "right"

# Blur region
def blur_person(image, box):
    x, y, w, h = box.xyxy[0].cpu().numpy().astype(int)
    top_region = image[y:y+int(0.08 * h), x:x+w]
    blurred_top_region = cv2.GaussianBlur(top_region, (15, 15), 0)
    image[y:y+int(0.08 * h), x:x+w] = blurred_top_region
    return image

# Load model and video
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

# Output video setup
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_with_boxes.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Object widths
class_avg_sizes = {
    "person": {"width_ratio": 2.5},
    "car": {"width_ratio": 0.37},
    "truck": {"width_ratio": 0.25},
    "bicycle": {"width_ratio": 2.3},
    "motorcycle": {"width_ratio": 2.4},
    "bus": {"width_ratio": 0.3},
    "traffic light": {"width_ratio": 2.95},
    "stop sign": {"width_ratio": 2.55},
    "bench": {"width_ratio": 1.6},
    "cat": {"width_ratio": 1.9},
    "dog": {"width_ratio": 1.5},
}

frame_count = 0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"\n🎥 Enhanced Object Detection System for Visually Impaired")
print(f"📊 Total frames: {total_frames}")
print("📺 Live video window enabled")
print("🔊 Audio features:")
print("   • Priority-based announcements (closest objects first)")
print("   • Spatial audio beeps (Left=Low, Center=Mid, Right=High pitch)")
print("   • Continuous proximity beeping")
print("   • Fast, clear speech announcements")
print("\n🎮 CONTROLS:")
print("   [S] - START audio system")
print("   [E] - STOP/END audio system")
print("   [P] - Pause/Resume video")
print("   [Q] - Quit program")
print("\n🚨 SAFETY FEATURE:")
print("   Emergency Override: ENABLED")
print("   System will AUTO-SPEAK for urgent dangers (≤2m)")
print("   Even when audio is STOPPED!")
print("\n⏸️  System STOPPED - Press 'S' to START\n")

pause = False
system_active = False  # Audio system starts as OFF
while cap.isOpened():
    if not pause:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing: {frame_count}/{total_frames} frames ({frame_count*100//total_frames}%)")

        results = model.predict(frame, verbose=False)
        result = results[0]
        objects_detected = []

        for box in result.boxes:
            label = result.names[box.cls[0].item()]
            cords = [round(x) for x in box.xyxy[0].tolist()]
            distance = calculate_distance(box, frame.shape[1], label)

            if label == "person":
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

            cv2.rectangle(frame, (cords[0], cords[1]), (cords[2], cords[3]), color, 2)
            cv2.putText(frame, f"{label} - {distance:.1f}m", (cords[0], cords[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Collect objects for announcement
            if distance <= 12.5:
                position = get_position(frame.shape[1], cords)
                priority = OBJECT_PRIORITIES.get(label, 5)
                objects_detected.append((priority, distance, label, position, cords))

        # Process detections and queue announcements
        # SAFETY: Check for urgent situations (≤2m) - speak even if system stopped!
        urgent_detected = any(dist <= 2.0 for _, dist, _, _, _ in objects_detected) if objects_detected else False

        if objects_detected and (system_active or urgent_detected):
            # Sort by priority, then by distance
            objects_detected.sort(key=lambda x: (x[0], x[1]))

            current_time = time.time()
            announced_count = 0

            for priority, distance, label, position, cords in objects_detected:
                # Continuous voice feedback with reasonable cooldowns
                if distance <= 1.5:
                    required_cooldown = 1.0   # VERY CLOSE: every 1s with voice
                elif distance <= 3:
                    required_cooldown = 2.0   # CLOSE: every 2s with voice
                elif distance <= 5:
                    required_cooldown = 3.0   # MEDIUM: every 3s with voice
                else:
                    required_cooldown = 4.0   # FAR: every 4s with voice

                # Check if enough time has passed
                time_since_last = current_time - last_spoken.get(label, 0)

                # Check if object is approaching (getting closer)
                prev_distance = last_distances.get(label, None)
                is_approaching = False
                if prev_distance is not None and distance < prev_distance - 0.2:
                    is_approaching = True

                # BYPASS cooldown if: 1) Approaching, or 2) First time (always speak for these)
                bypass_cooldown = is_approaching or (prev_distance is None)

                # SAFETY OVERRIDE: Always announce urgent dangers even if system stopped
                is_urgent = distance <= 2.0
                should_announce = (system_active and (time_since_last >= required_cooldown or bypass_cooldown)) or is_urgent

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
                    if distance <= 1.5:
                        motion = "DANGER very close"
                    elif distance <= 2.5:
                        motion = "very close"

                    # Update tracking
                    last_distances[label] = distance
                    last_spoken[label] = current_time

                    # Higher priority for closer/approaching objects
                    adjusted_priority = priority
                    if distance <= 2:
                        adjusted_priority = 0  # Highest priority for very close
                    elif is_approaching:
                        adjusted_priority = 1  # High priority for approaching

                    # Queue announcement (adjusted priority, timestamp, data)
                    audio_queue.put((adjusted_priority, current_time, label, distance, position, motion))

                    # Play directional beep (more urgent for closer objects)
                    play_beep(position, distance)

                    # Show what we're announcing
                    alert_type = "🚨 URGENT" if is_urgent and not system_active else "⚠️ DANGER" if distance <= 2 else "➕"
                    emergency_msg = " [EMERGENCY OVERRIDE - AUTO SPEAKING!]" if is_urgent and not system_active else ""
                    print(f"{alert_type} Queued: {label} at {distance:.1f}m ({position}) [Motion: {motion}]{emergency_msg}")

                    announced_count += 1
                    # Allow more announcements for dangerous situations
                    max_announcements = 4 if distance <= 2 else 2
                    if announced_count >= max_announcements:
                        break

        # Add enhanced status overlay
        status_y = 30

        # System status indicator (large and prominent)
        if system_active:
            status_color = (0, 255, 0)  # Green when active
            status_text = "AUDIO: ACTIVE"
            cv2.rectangle(frame, (10, 10), (250, 55), (0, 180, 0), -1)
            cv2.rectangle(frame, (10, 10), (250, 55), (0, 255, 0), 3)
        else:
            status_color = (0, 0, 255)  # Red when stopped
            status_text = "AUDIO: STOPPED"
            cv2.rectangle(frame, (10, 10), (250, 55), (0, 0, 180), -1)
            cv2.rectangle(frame, (10, 10), (250, 55), (0, 0, 255), 3)

        cv2.putText(frame, status_text, (20, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if objects_detected:
            cv2.putText(frame, f"Tracking: {len(objects_detected)} objects", (10, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Show closest object
            closest = objects_detected[0]
            cv2.putText(frame, f"Closest: {closest[2]} at {closest[1]:.1f}m ({closest[3]})",
                       (10, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 1)

        # Audio indicator (speaking) - show even during emergency
        if not audio_queue.empty():
            # Orange for emergency, Red for normal
            indicator_color = (0, 140, 255) if urgent_detected and not system_active else (0, 0, 255)
            cv2.circle(frame, (frame.shape[1] - 30, 30), 15, indicator_color, -1)
            speaking_text = "🚨 EMERGENCY" if urgent_detected and not system_active else "SPEAKING"
            cv2.putText(frame, speaking_text, (frame.shape[1] - 150, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, indicator_color, 2)

        # Control panel at bottom
        control_y = frame.shape[0] - 60
        cv2.rectangle(frame, (0, control_y - 10), (frame.shape[1], frame.shape[0]), (50, 50, 50), -1)
        cv2.putText(frame, "Controls: [S]Start  [E]Stop  [P]Pause  [Q]Quit",
                   (10, control_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        out.write(frame)
        cv2.imshow('Enhanced Object Detection for Visually Impaired', frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n⏹️ Quit requested by user")
        break
    elif key == ord('s') or key == ord('S'):
        if not system_active:
            system_active = True
            print("\n✅ AUDIO SYSTEM STARTED - Listening for objects...")
    elif key == ord('e') or key == ord('E'):
        if system_active:
            system_active = False
            # Clear tracking data
            last_spoken.clear()
            last_distances.clear()
            # Clear audio queue
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

cap.release()
out.release()
cv2.destroyAllWindows()

print("\n✅ Processing complete!")
print(f"📹 Output video saved to: output_with_boxes.avi")
print(f"🎬 Total frames processed: {frame_count}/{total_frames}")

