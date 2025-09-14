from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
from pathlib import Path
from ultralytics import YOLO
import torch
import logging
from time import time
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = os.path.join('static')
STATIC_IMAGES_FOLDER = os.path.join('static', 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Create all required directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, STATIC_IMAGES_FOLDER]:
    try:
        Path(folder).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {folder}")
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {e}")

# Load YOLO model
try:
    # Try to use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLO('yolov8x.pt')  # Load YOLOv8x model
    model.to(device)
    logger.info(f"Model loaded successfully on {device}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None  # Handle this case in your detection function

# Configuration for detection
CONFIDENCE_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    # Clean up old files in uploads directory
    cleanup_old_files()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Verify the image can be opened
            img = cv2.imread(file_path)
            if img is None:
                raise ValueError("Could not open image file")
            
            # Perform object detection
            output_image_path, explanations = detect_objects(file_path)
            
            # Get just the filename from the path
            output_filename = os.path.basename(output_image_path)

            # Return the detected objects and output image URL
            return jsonify({
                "message": "File processed successfully",
                "output_image": url_for('static', filename=output_filename),
                "detections": explanations
            })
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({"error": str(e)}), 500
        finally:
            # Clean up uploaded file
            try:
                os.remove(file_path)
            except:
                pass

def detect_objects(image_path):
    if model is None:
        raise Exception("Model not properly loaded")
    try:
        # Perform prediction
        results = model(image_path, conf=CONFIDENCE_THRESHOLD, iou=IOU_THRESHOLD)[0]
        
        # Get the original image for drawing
        image = cv2.imread(image_path)
        
        explanations = []
        
        # Process detections
        for detection in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            class_name = results.names[int(class_id)]
            
            color = generate_color(class_name)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_name} {confidence:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - text_size[1] - 10), (x1 + text_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            explanations.append({
                "label": class_name,
                "confidence": confidence
            })

        explanations.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Clean up old output files before saving new one
        cleanup_output_files()
        
        # Generate unique filename with timestamp and random component
        timestamp = int(time() * 1000)  # millisecond precision
        output_filename = f'output_{timestamp}.jpg'
        output_image_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Ensure the file is written
        success = cv2.imwrite(output_image_path, image)
        if not success:
            raise Exception("Failed to save output image")
        
        # Verify the file exists
        if not os.path.exists(output_image_path):
            raise Exception("Output file was not created")
            
        return output_image_path, explanations
    
    except Exception as e:
        logger.error(f"Error in detect_objects: {e}")
        raise

def generate_color(class_name):
    """Generate a consistent color for a class name using a hash function"""
    hash_value = hash(class_name)
    r = (hash_value & 0xFF0000) >> 16
    g = (hash_value & 0x00FF00) >> 8
    b = hash_value & 0x0000FF
    return (b, g, r)  # BGR format for OpenCV

def cleanup_old_files():
    """Clean up old files from uploads directory"""
    try:
        for file in Path(UPLOAD_FOLDER).glob('*'):
            if file.is_file():
                file.unlink()
    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")

def cleanup_output_files():
    """Clean up old output images"""
    try:
        output_pattern = os.path.join(app.config['OUTPUT_FOLDER'], 'output_*.jpg')
        for file in Path(app.config['OUTPUT_FOLDER']).glob('output_*.jpg'):
            try:
                file.unlink()
                logger.info(f"Cleaned up file: {file}")
            except Exception as e:
                logger.warning(f"Failed to delete file {file}: {e}")
    except Exception as e:
        logger.error(f"Error in cleanup_output_files: {e}")

if __name__ == '__main__':
    # Verify CUDA availability
    if torch.cuda.is_available():
        logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA is not available. Using CPU")
    
    app.run(debug=True)