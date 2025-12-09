 #!/usr/bin/env python3
"""
Simple test to verify TTS is speaking the correct distance values
"""

import pyttsx3
import time

def test_tts():
    """Test TTS with known distance values"""
    
    test_distances = [0.4, 1.2, 2.5, 3.8, 5.1]
    
    for distance in test_distances:
        print(f"\n🎤 Testing distance: {distance:.1f}m")
        
        # Format message exactly like the main system
        dist_str = f"{distance:.1f}"
        message = f"person {dist_str} meters ahead"
        
        print(f"   📝 Message to speak: '{message}'")
        print(f"   🔢 Distance value: {distance:.1f}")
        print(f"   📊 Distance string: '{dist_str}'")
        
        try:
            print(f"   ▶️ NOW SPEAKING...")
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.setProperty('volume', 1.0)
            engine.say(message)
            engine.runAndWait()
            engine.stop()
            del engine
            print(f"   ✅ FINISHED SPEAKING")
        except Exception as e:
            print(f"   ❌ TTS ERROR: {e}")
        
        # Wait between tests
        time.sleep(2)
    
    print("\n✅ TTS test complete!")

if __name__ == "__main__":
    test_tts()