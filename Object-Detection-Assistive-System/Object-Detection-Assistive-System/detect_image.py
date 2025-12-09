"""
Image Object Detection with Accuracy Levels
Detect objects in static images and show confidence scores

Features:
- Single image or batch processing
- Display confidence/accuracy for each detection
- Save annotated images with results
- Generate detection reports
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import sys
import json
import argparse
from pathlib import Path
import time

def load_model(model_path):
    """Load YOLO model"""
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found: {model_path}")
        sys.exit(1)
    
    print(f"🤖 Loading model: {model_path}")
    model = YOLO(model_path)
    print("✅ Model loaded successfully")
    return model

def detect_objects_in_image(image_path, model, confidence_threshold=0.25, save_output=True, output_dir="output_images"):
    """
    Detect objects in a single image
    
    Args:
        image_path: Path to input image
        model: YOLO model
        confidence_threshold: Minimum confidence to display (0.0 - 1.0)
        save_output: Whether to save annotated image
        output_dir: Directory to save output images
    
    Returns:
        Detection results dictionary
    """
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image not found: {image_path}")
        return None
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Error: Could not read image: {image_path}")
        return None
    
    print(f"\n📸 Processing: {os.path.basename(image_path)}")
    print(f"   Resolution: {image.shape[1]}x{image.shape[0]}")
    
    # Run detection
    results = model.predict(image, verbose=False, conf=confidence_threshold)
    result = results[0]
    
    # Prepare detection data
    detections = []
    annotated_image = image.copy()
    
    # Process each detection
    for idx, box in enumerate(result.boxes):
        # Get detection info
        label = result.names[box.cls[0].item()]
        confidence = box.conf[0].item()
        coords = box.xyxy[0].cpu().numpy().astype(int)
        
        # Store detection
        detection = {
            'label': label,
            'confidence': confidence,
            'confidence_percent': f"{confidence * 100:.2f}%",
            'bbox': coords.tolist(),
            'position': {
                'x1': int(coords[0]),
                'y1': int(coords[1]),
                'x2': int(coords[2]),
                'y2': int(coords[3])
            }
        }
        detections.append(detection)
        
        # Color based on object type
        color_map = {
            'person': (0, 255, 0),      # Green
            'car': (0, 255, 255),       # Yellow
            'truck': (255, 165, 0),     # Orange
            'bus': (255, 0, 255),       # Magenta
            'motorcycle': (255, 255, 0), # Cyan
            'bicycle': (0, 165, 255),   # Orange-Yellow
            'traffic light': (0, 128, 255), # Orange-Red
            'stop sign': (0, 0, 255),   # Red
        }
        color = color_map.get(label, (255, 0, 0))  # Default: Blue
        
        # Color intensity based on confidence
        # High confidence = bright, low confidence = darker
        color = tuple(int(c * (0.5 + confidence * 0.5)) for c in color)
        
        # Draw bounding box
        thickness = 2 if confidence > 0.7 else 1
        cv2.rectangle(annotated_image, 
                     (coords[0], coords[1]), 
                     (coords[2], coords[3]), 
                     color, thickness)
        
        # Prepare label text with confidence
        label_text = f"{label} {confidence*100:.1f}%"
        
        # Calculate text size for background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, font_thickness
        )
        
        # Draw background rectangle for text
        cv2.rectangle(annotated_image,
                     (coords[0], coords[1] - text_height - 10),
                     (coords[0] + text_width, coords[1]),
                     color, -1)
        
        # Draw text
        cv2.putText(annotated_image, label_text,
                   (coords[0], coords[1] - 5),
                   font, font_scale, (255, 255, 255), font_thickness)
        
        # Add confidence bar (visual indicator)
        bar_width = coords[2] - coords[0]
        bar_height = 8
        bar_fill = int(bar_width * confidence)
        
        # Draw confidence bar background
        cv2.rectangle(annotated_image,
                     (coords[0], coords[3] + 2),
                     (coords[2], coords[3] + 2 + bar_height),
                     (100, 100, 100), -1)
        
        # Draw confidence bar fill
        bar_color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255) if confidence > 0.5 else (0, 165, 255)
        cv2.rectangle(annotated_image,
                     (coords[0], coords[3] + 2),
                     (coords[0] + bar_fill, coords[3] + 2 + bar_height),
                     bar_color, -1)
    
    # Add summary overlay
    summary_height = 120
    overlay = annotated_image.copy()
    cv2.rectangle(overlay, (10, 10), (400, summary_height), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, annotated_image, 0.3, 0, annotated_image)
    
    # Summary text
    cv2.putText(annotated_image, f"Objects Detected: {len(detections)}", 
               (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    if detections:
        # Highest confidence detection
        highest = max(detections, key=lambda x: x['confidence'])
        cv2.putText(annotated_image, f"Highest Confidence: {highest['label']}", 
                   (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated_image, f"  @ {highest['confidence_percent']}", 
                   (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Average confidence
        avg_conf = sum(d['confidence'] for d in detections) / len(detections)
        cv2.putText(annotated_image, f"Average Confidence: {avg_conf*100:.1f}%", 
                   (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Save annotated image
    if save_output:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Saved annotated image: {output_path}")
    
    # Print detection summary
    print(f"\n📊 Detection Summary:")
    print(f"   Total objects: {len(detections)}")
    
    if detections:
        print(f"\n   Detected Objects (sorted by confidence):")
        sorted_detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        for i, det in enumerate(sorted_detections, 1):
            print(f"   {i}. {det['label']:15s} - Confidence: {det['confidence_percent']:7s}")
        
        # Object count by type
        object_counts = {}
        for det in detections:
            object_counts[det['label']] = object_counts.get(det['label'], 0) + 1
        
        print(f"\n   Object Breakdown:")
        for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"   - {obj_type}: {count}")
    
    return {
        'image_path': image_path,
        'image_size': {'width': image.shape[1], 'height': image.shape[0]},
        'total_detections': len(detections),
        'detections': detections,
        'annotated_image_path': output_path if save_output else None
    }

def process_batch_images(image_dir, model, confidence_threshold=0.25, output_dir="output_images"):
    """Process multiple images in a directory"""
    
    # Supported image formats
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff']
    
    # Find all images
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(image_dir).glob(f"*{ext}"))
        image_files.extend(Path(image_dir).glob(f"*{ext.upper()}"))
    
    if not image_files:
        print(f"❌ No images found in: {image_dir}")
        return []
    
    print(f"\n📁 Found {len(image_files)} images to process")
    print(f"{'='*60}\n")
    
    # Process each image
    results = []
    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {image_path.name}")
        result = detect_objects_in_image(str(image_path), model, confidence_threshold, True, output_dir)
        if result:
            results.append(result)
        print()
    
    # Generate batch summary
    print(f"\n{'='*60}")
    print(f"📊 BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total images processed: {len(results)}")
    print(f"Total objects detected: {sum(r['total_detections'] for r in results)}")
    
    # Overall statistics
    all_detections = [det for r in results for det in r['detections']]
    if all_detections:
        avg_confidence = sum(d['confidence'] for d in all_detections) / len(all_detections)
        print(f"Average confidence: {avg_confidence*100:.1f}%")
        
        # Most common objects
        object_counts = {}
        for det in all_detections:
            object_counts[det['label']] = object_counts.get(det['label'], 0) + 1
        
        print(f"\nMost detected objects:")
        for obj_type, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  - {obj_type}: {count}")
    
    print(f"{'='*60}\n")
    
    # Save JSON report
    report_path = os.path.join(output_dir, "detection_report.json")
    with open(report_path, 'w') as f:
        json.dump({
            'summary': {
                'total_images': len(results),
                'total_detections': sum(r['total_detections'] for r in results),
                'average_confidence': avg_confidence if all_detections else 0,
            },
            'results': results
        }, f, indent=2)
    print(f"📄 Detailed report saved: {report_path}")
    
    return results

def display_image(image_path, window_name="Detection Result"):
    """Display image in a window"""
    image = cv2.imread(image_path)
    if image is not None:
        # Resize if too large
        max_width = 1200
        max_height = 800
        h, w = image.shape[:2]
        
        if w > max_width or h > max_height:
            scale = min(max_width/w, max_height/h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
        
        cv2.imshow(window_name, image)
        print("\n👁️  Press any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(
        description='Detect objects in images with confidence/accuracy levels',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode (asks for image path)
  python detect_image.py
  
  # Detect objects in single image
  python detect_image.py --image photo.jpg
  
  # Batch process all images in a folder
  python detect_image.py --batch-dir ./photos
  
  # Custom confidence threshold (only show high-confidence detections)
  python detect_image.py --image photo.jpg --confidence 0.7
  
  # Display result after detection
  python detect_image.py --image photo.jpg --display
  
  # Custom model and output directory
  python detect_image.py --image photo.jpg --model custom.pt --output results/
        """
    )
    
    parser.add_argument('--image', type=str, help='Path to single image file')
    parser.add_argument('--batch-dir', type=str, help='Directory containing multiple images')
    parser.add_argument('--model', type=str, default='gpModel.pt', help='Path to YOLO model (default: gpModel.pt)')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold 0.0-1.0 (default: 0.25)')
    parser.add_argument('--output', type=str, default='output_images', help='Output directory (default: output_images)')
    parser.add_argument('--display', action='store_true', help='Display result image after detection')
    parser.add_argument('--no-save', action='store_true', help='Do not save annotated images')
    
    args = parser.parse_args()
    
    # Interactive mode - ask user for input if no arguments provided
    if not args.image and not args.batch_dir:
        print("\n" + "="*60)
        print("  📸 Interactive Image Object Detection")
        print("="*60 + "\n")
        
        # Ask what mode
        print("Choose detection mode:")
        print("  [1] Single image")
        print("  [2] Batch process (multiple images in folder)")
        print("  [Q] Quit\n")
        
        mode = input("Enter your choice (1/2/Q): ").strip()
        
        if mode.upper() == 'Q':
            print("👋 Goodbye!")
            sys.exit(0)
        elif mode == '1':
            # Single image mode
            image_path = input("\n📁 Enter image path (or drag & drop file here): ").strip()
            # Remove quotes if user drags file
            image_path = image_path.strip('"').strip("'")
            args.image = image_path
            
            # Ask if they want to display
            display_choice = input("🖼️  Display result after detection? (y/n, default: y): ").strip().lower()
            if display_choice != 'n':
                args.display = True
                
        elif mode == '2':
            # Batch mode
            folder_path = input("\n📁 Enter folder path containing images: ").strip()
            # Remove quotes
            folder_path = folder_path.strip('"').strip("'")
            args.batch_dir = folder_path
            
            # Ask about confidence
            conf_choice = input("🎯 Confidence threshold (0.0-1.0, default: 0.25): ").strip()
            if conf_choice:
                try:
                    args.confidence = float(conf_choice)
                except:
                    print("⚠️  Invalid confidence, using default 0.25")
        else:
            print("❌ Invalid choice. Exiting.")
            sys.exit(1)
        
        print("\n" + "="*60 + "\n")
    
    # Validate that we now have either image or batch-dir
    if not args.image and not args.batch_dir:
        print("❌ Error: No image or directory provided")
        sys.exit(1)
    
    # Load model
    model = load_model(args.model)
    
    # Process images
    if args.image:
        # Single image
        result = detect_objects_in_image(
            args.image, 
            model, 
            confidence_threshold=args.confidence,
            save_output=not args.no_save,
            output_dir=args.output
        )
        
        if result and args.display and result['annotated_image_path']:
            display_image(result['annotated_image_path'])
    
    elif args.batch_dir:
        # Batch processing
        results = process_batch_images(
            args.batch_dir,
            model,
            confidence_threshold=args.confidence,
            output_dir=args.output
        )
        
        if results and args.display and results[0]['annotated_image_path']:
            print("\nDisplaying first result...")
            display_image(results[0]['annotated_image_path'])

if __name__ == "__main__":
    main()

