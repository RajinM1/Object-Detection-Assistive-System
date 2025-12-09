"""
Setup Verification Script
Run this to check if your system is ready to use the Object Detection Assistive System
"""

import sys
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def print_status(item, status, message=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {item:40s} {message}")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print_status("Python Version", True, f"{version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print_status("Python Version", False, f"{version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    dependencies = {
        'ultralytics': 'YOLO Model',
        'cv2': 'OpenCV',
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pyttsx3': 'Text-to-Speech'
    }
    
    all_installed = True
    for package, name in dependencies.items():
        try:
            if package == 'cv2':
                import cv2
            elif package == 'ultralytics':
                from ultralytics import YOLO
            elif package == 'torch':
                import torch
            elif package == 'torchvision':
                import torchvision
            elif package == 'numpy':
                import numpy
            elif package == 'pyttsx3':
                import pyttsx3
            
            print_status(name, True, "Installed")
        except ImportError:
            print_status(name, False, "NOT installed")
            all_installed = False
    
    return all_installed

def check_files():
    """Check if required files exist"""
    files = {
        'main_enhanced.py': 'Main Enhanced Script',
        'config.json': 'Configuration File',
        'requirements.txt': 'Requirements File',
        'README.md': 'Documentation',
        'gpModel.pt': 'YOLO Model (optional for testing)'
    }
    
    all_exist = True
    for filename, description in files.items():
        exists = os.path.exists(filename)
        if filename == 'gpModel.pt':
            # Model is optional, just warn
            if exists:
                print_status(description, True, "Found")
            else:
                print_status(description, False, "Not found (you'll need a model to run)")
        else:
            print_status(description, exists, "Found" if exists else "MISSING")
            if not exists and filename != 'gpModel.pt':
                all_exist = False
    
    return all_exist

def check_gpu():
    """Check if GPU/CUDA is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print_status("GPU (CUDA)", True, f"{gpu_name}")
            return True
        else:
            print_status("GPU (CUDA)", False, "Not available (CPU will be used)")
            return False
    except:
        print_status("GPU (CUDA)", False, "Unable to check")
        return False

def check_camera():
    """Check if camera is accessible"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print_status("Camera (ID: 0)", True, "Accessible")
                return True
            else:
                print_status("Camera (ID: 0)", False, "Opened but no frame")
                return False
        else:
            print_status("Camera (ID: 0)", False, "Cannot open")
            return False
    except Exception as e:
        print_status("Camera (ID: 0)", False, f"Error: {str(e)}")
        return False

def check_audio():
    """Check if text-to-speech is working"""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        print_status("Text-to-Speech Engine", True, "Initialized successfully")
        return True
    except Exception as e:
        print_status("Text-to-Speech Engine", False, f"Error: {str(e)}")
        return False

def main():
    print_header("Object Detection Assistive System - Setup Verification")
    
    print("📋 Checking Python Environment...")
    python_ok = check_python_version()
    
    print("\n📦 Checking Dependencies...")
    deps_ok = check_dependencies()
    
    print("\n📁 Checking Required Files...")
    files_ok = check_files()
    
    print("\n🖥️  Checking Hardware...")
    gpu_available = check_gpu()
    camera_ok = check_camera()
    
    print("\n🔊 Checking Audio System...")
    audio_ok = check_audio()
    
    # Summary
    print_header("Verification Summary")
    
    if python_ok and deps_ok and files_ok:
        print("🎉 READY TO USE!")
        print("\nYour system is properly configured.")
        print("\n🚀 Next Steps:")
        print("   1. Ensure you have a YOLO model file (gpModel.pt)")
        print("   2. Run: python main_enhanced.py --camera")
        print("   3. Press 'S' to start audio")
        
        if not gpu_available:
            print("\n💡 Tip: No GPU detected. For better performance:")
            print("   - Install CUDA-enabled PyTorch")
            print("   - Or use --cpu flag (slower but works)")
        
        if not camera_ok:
            print("\n⚠️  Camera Warning:")
            print("   - No camera detected at ID 0")
            print("   - Try: python main_enhanced.py --camera --camera-id 1")
            print("   - Or use a video file: python main_enhanced.py --video your_video.mp4")
        
    else:
        print("❌ SETUP INCOMPLETE\n")
        
        if not python_ok:
            print("⚠️  Python version issue:")
            print("   - Please upgrade to Python 3.8 or higher")
            print("   - Download from: https://www.python.org/downloads/\n")
        
        if not deps_ok:
            print("⚠️  Missing dependencies:")
            print("   - Run: pip install -r requirements.txt\n")
        
        if not files_ok:
            print("⚠️  Missing files:")
            print("   - Ensure you have all project files")
            print("   - Re-download or check your installation\n")
    
    print(f"\n{'='*60}")
    print("For help, see README.md or QUICK_START.md")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()

