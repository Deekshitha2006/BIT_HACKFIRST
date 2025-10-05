from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import traceback
import cv2
import numpy as np
import os
import tempfile
import uuid
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create a directory for temporary files
TEMP_DIR = tempfile.mkdtemp()
app.config['TEMP_DIR'] = TEMP_DIR

try:
    model = YOLO('yolov8n.pt')  # Load pre-trained model
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

# Helper function to determine crowd level
def get_crowd_level(count):
    if count < 10:
        return 'Low'
    elif count < 30:
        return 'Medium'
    else:
        return 'High'

# Route for the first page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the second page with temple ID parameter
@app.route('/second')
def second_page():
    # Get temple ID from URL parameters, default to 'iskon' if not provided
    temple_id = request.args.get('id', 'iskon')
    return render_template('index1.html', temple_id=temple_id)

@app.route('/detect', methods=['POST'])
def detect_objects():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run inference
        results = model(img)
        
        # Extract detections
        detections = []
        people_count = 0
        
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                detections.append({
                    'class': class_name,
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]
                })
                
                # Count people
                if class_name == 'person':
                    people_count += 1
        
        return jsonify({
            'detections': detections,
            'people_count': people_count,
            'crowd_level': get_crowd_level(people_count)
        })
    except Exception as e:
        print(f"Error during detection: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect_crowd', methods=['POST'])
def detect_crowd():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run inference with crowd detection
        results = model(img)
        
        # Extract detections
        detections = []
        people_count = 0
        
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                detections.append({
                    'class': class_name,
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]
                })
                
                # Count people
                if class_name == 'person':
                    people_count += 1
        
        # Calculate crowd density
        img_width, img_height = img.size
        img_area = img_width * img_height / 1000000  # Area in megapixels
        crowd_density = round(people_count / img_area, 2) if img_area > 0 else 0
        
        return jsonify({
            'detections': detections,
            'people_count': people_count,
            'crowd_density': crowd_density,
            'crowd_level': get_crowd_level(people_count)
        })
    except Exception as e:
        print(f"Error during crowd detection: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/process_video', methods=['POST'])
def process_video():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create a unique filename for the processed video
        unique_id = str(uuid.uuid4())
        input_filename = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_input.mp4")
        output_filename = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_output.mp4")
        
        # Save the uploaded video
        file.save(input_filename)
        
        # Process the video
        cap = cv2.VideoCapture(input_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        frame_count = 0
        total_people_count = 0
        max_people_count = 0
        min_people_count = float('inf')
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLO inference on the frame
            results = model(frame)
            
            # Count people in this frame
            frame_people_count = 0
            for result in results:
                for box in result.boxes:
                    class_name = model.names[int(box.cls)]
                    if class_name == 'person':
                        frame_people_count += 1
                        
                        # Draw bounding box
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"{class_name} {float(box.conf):.2f}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Update statistics
            total_people_count += frame_people_count
            max_people_count = max(max_people_count, frame_people_count)
            min_people_count = min(min_people_count, frame_people_count)
            
            # Write the frame
            out.write(frame)
            frame_count += 1
        
        # Release everything
        cap.release()
        out.release()
        
        # Calculate average people count
        avg_people_count = total_people_count / frame_count if frame_count > 0 else 0
        
        # Return the processed video URL and statistics
        return jsonify({
            'video_url': f"/get_video/{unique_id}",
            'frame_count': frame_count,
            'total_people_count': total_people_count,
            'average_people_count': avg_people_count,
            'max_people_count': max_people_count,
            'min_people_count': min_people_count,
            'people_count': round(avg_people_count),  # Use average as representative count
            'crowd_level': get_crowd_level(round(avg_people_count))
        })
    except Exception as e:
        print(f"Error during video processing: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap():
    try:
        print("Starting heatmap generation...")
        
        if 'file' not in request.files:
            print("No file uploaded")
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("No file selected")
            return jsonify({'error': 'No file selected'}), 400
        
        print(f"Processing file: {file.filename}")
        
        # Read the image
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        print(f"Original image mode: {img.mode}, size: {img.size}")
        
        # Convert to RGB if needed
        if img.mode != 'RGB':
            img = img.convert('RGB')
            print(f"Converted to RGB mode")
        
        img_array = np.array(img)
        print(f"Image array shape: {img_array.shape}")
        
        # Create a unique filename for the heatmap
        unique_id = str(uuid.uuid4())
        heatmap_filename = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_heatmap.jpg")
        print(f"Heatmap filename: {heatmap_filename}")
        
        # Run YOLO inference
        print("Running YOLO inference...")
        results = model(img)
        
        # Create a blank heatmap
        heatmap = np.zeros((img_array.shape[0], img_array.shape[1]), dtype=np.float32)
        
        # Count people and create heatmap
        people_count = 0
        density_zones = {'high': 0, 'medium': 0, 'low': 0}
        
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                if class_name == 'person':
                    people_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Add a Gaussian blob at the center of each person
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Create a small Gaussian kernel
                    kernel_size = 51
                    sigma = 15
                    kernel = np.zeros((kernel_size, kernel_size))
                    for i in range(kernel_size):
                        for j in range(kernel_size):
                            dist = np.sqrt((i - kernel_size//2)**2 + (j - kernel_size//2)**2)
                            kernel[i, j] = np.exp(-dist**2 / (2 * sigma**2))
                    
                    # Normalize the kernel
                    kernel = kernel / np.sum(kernel)
                    
                    # Add the kernel to the heatmap with bounds checking
                    y_start = max(0, center_y - kernel_size//2)
                    y_end = min(heatmap.shape[0], center_y + kernel_size//2 + 1)
                    x_start = max(0, center_x - kernel_size//2)
                    x_end = min(heatmap.shape[1], center_x + kernel_size//2 + 1)
                    
                    # Calculate kernel slices
                    kernel_y_start = max(0, kernel_size//2 - center_y)
                    kernel_y_end = kernel_y_start + (y_end - y_start)
                    kernel_x_start = max(0, kernel_size//2 - center_x)
                    kernel_x_end = kernel_x_start + (x_end - x_start)
                    
                    # Ensure we have valid ranges
                    if (y_start < y_end and x_start < x_end and 
                        kernel_y_start < kernel_y_end and kernel_x_start < kernel_x_end):
                        heatmap[y_start:y_end, x_start:x_end] += kernel[kernel_y_start:kernel_y_end, kernel_x_start:kernel_x_end]
        
        print(f"People detected: {people_count}")
        print(f"Heatmap max value: {np.max(heatmap)}")
        
        # Normalize the heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap) * 255
        else:
            # If no people detected, create a uniform low-value heatmap
            heatmap = np.ones_like(heatmap) * 50
            print("No people detected, creating uniform heatmap")
        
        # Apply a colormap to the heatmap
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend the original image with the heatmap
        alpha = 0.6  # Transparency factor
        result = cv2.addWeighted(img_array, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Count density zones
        for i in range(0, 256, 85):
            mask = (heatmap >= i) & (heatmap < i + 85)
            count = np.sum(mask)
            
            if i == 0:  # Low density
                density_zones['low'] = count
            elif i == 85:  # Medium density
                density_zones['medium'] = count
            else:  # High density
                density_zones['high'] = count
        
        print(f"Density zones: {density_zones}")
        
        # Save the heatmap
        print(f"Saving heatmap to: {heatmap_filename}")
        success = cv2.imwrite(heatmap_filename, result)
        print(f"Save success: {success}")
        
        # Check if file was saved successfully
        if not os.path.exists(heatmap_filename):
            print("File was not saved successfully")
            return jsonify({'error': 'Failed to save heatmap file'}), 500
        
        file_size = os.path.getsize(heatmap_filename)
        print(f"Heatmap file size: {file_size} bytes")
        
        # Return the heatmap URL and statistics
        return jsonify({
            'heatmap_url': f"/get_heatmap/{unique_id}",
            'people_count': people_count,
            'density_zones': density_zones,
            'crowd_level': get_crowd_level(people_count)
        })
    except Exception as e:
        print(f"Error during heatmap generation: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/track_motion', methods=['POST'])
def track_motion():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create a unique filename for the processed video
        unique_id = str(uuid.uuid4())
        input_filename = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_input.mp4")
        output_filename = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_motion.mp4")
        
        # Save the uploaded video
        file.save(input_filename)
        
        # Process the video
        cap = cv2.VideoCapture(input_filename)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        # Initialize background subtractor
        fgbg = cv2.createBackgroundSubtractorMOG2()
        
        # For tracking
        tracks = {}
        track_id = 0
        tracks_count = 0
        speeds = []
        flow_directions = []
        congestion_areas = []
        
        # For optical flow
        prev_frame = None
        prev_gray = None
        
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply background subtraction
            fgmask = fgbg.apply(gray)
            
            # Find contours
            contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Process contours
            for contour in contours:
                if cv2.contourArea(contour) < 500:  # Filter small contours
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Simple tracking (just for demo)
                center = (x + w // 2, y + h // 2)
                
                # Check if this is a new track
                matched = False
                for tid, track in tracks.items():
                    track_center = track['center']
                    distance = np.sqrt((center[0] - track_center[0])**2 + (center[1] - track_center[1])**2)
                    
                    if distance < 50:  # If close to an existing track
                        # Update track
                        track['center'] = center
                        track['positions'].append(center)
                        track['last_seen'] = frame_count
                        
                        # Calculate speed
                        if len(track['positions']) > 1:
                            prev_pos = track['positions'][-2]
                            speed = np.sqrt((center[0] - prev_pos[0])**2 + (center[1] - prev_pos[1])**2)
                            speeds.append(speed)
                            
                            # Calculate direction
                            direction = np.arctan2(center[1] - prev_pos[1], center[0] - prev_pos[0])
                            flow_directions.append(direction)
                        
                        matched = True
                        break
                
                if not matched:
                    # Create new track
                    tracks[track_id] = {
                        'center': center,
                        'positions': [center],
                        'first_seen': frame_count,
                        'last_seen': frame_count
                    }
                    track_id += 1
                    tracks_count += 1
            
            # Clean up old tracks
            tracks_to_remove = []
            for tid, track in tracks.items():
                if frame_count - track['last_seen'] > 30:  # If not seen for 30 frames
                    tracks_to_remove.append(tid)
            
            for tid in tracks_to_remove:
                del tracks[tid]
            
            # Draw tracks
            for tid, track in tracks.items():
                positions = track['positions']
                if len(positions) > 1:
                    for i in range(1, len(positions)):
                        cv2.line(frame, positions[i-1], positions[i], (0, 0, 255), 2)
            
            # Optical flow for congestion detection
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                
                # Calculate magnitude and angle of flow vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Find areas with high magnitude (congestion)
                congestion_mask = magnitude > np.percentile(magnitude, 90)
                if np.any(congestion_mask):
                    congestion_areas.append(frame_count)
            
            # Update previous frame
            prev_gray = gray.copy()
            
            # Write the frame
            out.write(frame)
            frame_count += 1
        
        # Release everything
        cap.release()
        out.release()
        
        # Calculate statistics
        avg_speed = np.mean(speeds) if speeds else 0
        
        # Determine main flow direction
        if flow_directions:
            # Convert angles to degrees
            angles_deg = [np.degrees(angle) % 360 for angle in flow_directions]
            
            # Create histogram of directions
            hist, bins = np.histogram(angles_deg, bins=8, range=(0, 360))
            
            # Find the bin with the highest count
            max_bin = np.argmax(hist)
            direction_angle = (bins[max_bin] + bins[max_bin + 1]) / 2
            
            # Convert to compass direction
            if 337.5 <= direction_angle or direction_angle < 22.5:
                direction_flow = "East"
            elif 22.5 <= direction_angle < 67.5:
                direction_flow = "Northeast"
            elif 67.5 <= direction_angle < 112.5:
                direction_flow = "North"
            elif 112.5 <= direction_angle < 157.5:
                direction_flow = "Northwest"
            elif 157.5 <= direction_angle < 202.5:
                direction_flow = "West"
            elif 202.5 <= direction_angle < 247.5:
                direction_flow = "Southwest"
            elif 247.5 <= direction_angle < 292.5:
                direction_flow = "South"
            else:
                direction_flow = "Southeast"
        else:
            direction_flow = "Unknown"
        
        # Count congestion areas
        congestion_count = len(congestion_areas)
        
        # Return the processed video URL and statistics
        return jsonify({
            'video_url': f"/get_motion_video/{unique_id}",
            'tracks_count': tracks_count,
            'average_speed': avg_speed,
            'direction_flow': direction_flow,
            'congestion_areas': congestion_count,
            'people_count': tracks_count,  # Use track count as people count
            'crowd_level': get_crowd_level(tracks_count)
        })
    except Exception as e:
        print(f"Error during motion tracking: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/check_social_distancing', methods=['POST'])
def check_social_distancing():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        
        # Run inference with crowd detection
        results = model(img)
        
        # Extract detections
        detections = []
        people_positions = []
        people_count = 0
        violations = 0
        
        for result in results:
            for box in result.boxes:
                class_name = model.names[int(box.cls)]
                confidence = float(box.conf)
                bbox = box.xyxy.tolist()[0]
                
                detections.append({
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                # Store people positions
                if class_name == 'person':
                    people_count += 1
                    
                    # Calculate center of bounding box
                    x1, y1, x2, y2 = bbox
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    people_positions.append((center_x, center_y))
        
        # Check social distancing violations
        min_distance = 100  # Minimum distance in pixels
        
        for i in range(len(people_positions)):
            for j in range(i + 1, len(people_positions)):
                # Calculate Euclidean distance
                distance = np.sqrt(
                    (people_positions[i][0] - people_positions[j][0])**2 + 
                    (people_positions[i][1] - people_positions[j][1])**2
                )
                
                if distance < min_distance:
                    violations += 1
                    
                    # Mark violation in detections
                    if i < len(detections):
                        detections[i]['violation'] = True
                    if j < len(detections):
                        detections[j]['violation'] = True
        
        # Calculate compliance rate
        total_pairs = people_count * (people_count - 1) / 2 if people_count > 1 else 1
        compliance_rate = ((total_pairs - violations) / total_pairs) * 100
        
        return jsonify({
            'detections': detections,
            'people_count': people_count,
            'violations': violations,
            'compliance_rate': compliance_rate,
            'crowd_level': get_crowd_level(people_count)
        })
    except Exception as e:
        print(f"Error during social distancing check: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# Routes to serve processed files
@app.route('/get_video/<unique_id>')
def get_video(unique_id):
    try:
        video_path = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_output.mp4")
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
            
        return send_file(video_path, as_attachment=True)
    except Exception as e:
        print(f"Error serving video: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get_heatmap/<unique_id>')
def get_heatmap(unique_id):
    try:
        heatmap_path = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_heatmap.jpg")
        
        if not os.path.exists(heatmap_path):
            print(f"Heatmap file not found: {heatmap_path}")
            return jsonify({'error': 'Heatmap file not found'}), 404
            
        print(f"Serving heatmap file: {heatmap_path}")
        return send_file(heatmap_path, as_attachment=True)
    except Exception as e:
        print(f"Error serving heatmap: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
@app.route('/get_motion_video/<unique_id>')
def get_motion_video(unique_id):
    try:
        video_path = os.path.join(app.config['TEMP_DIR'], f"{unique_id}_motion.mp4")
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Motion video file not found'}), 404
            
        return send_file(video_path, as_attachment=True)
    except Exception as e:
        print(f"Error serving motion video: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'YOLOv8 detection API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)