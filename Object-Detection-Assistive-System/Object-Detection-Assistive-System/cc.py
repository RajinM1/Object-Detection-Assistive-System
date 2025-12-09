import pyttsx3
import time

engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
if voices:
    engine.setProperty('voice', voices[0].id)  # Or try voices[1].id

messages = [
    "Test one: This should play first.",
    "Test two: If you hear this, multiple works.",
    "Test three: Continuing.",
    "Test four: Almost done.",
    "Test five: Success if heard!"
]

for msg in messages:
    print(f"Speaking: {msg}")
    engine.say(msg)
    engine.runAndWait()
    engine.stop()  # Clears queue to prevent skips
    time.sleep(1)