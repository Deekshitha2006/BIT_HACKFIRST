from flask import Flask, request, jsonify, render_template, redirect, url_for
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import io
import traceback

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

try:
    model = YOLO('yolov8n.pt')  # Load pre-trained model
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    traceback.print_exc()

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
        for result in results:
            for box in result.boxes:
                detections.append({
                    'class': model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy.tolist()[0]
                })
        
        return jsonify({'detections': detections})
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
        
        # Calculate crowd density and level
        img_width, img_height = img.size
        img_area = img_width * img_height / 1000000  # Area in megapixels
        
        crowd_density = round(people_count / img_area, 2) if img_area > 0 else 0
        
        if people_count < 10:
            crowd_level = 'Low'
        elif people_count < 30:
            crowd_level = 'Medium'
        else:
            crowd_level = 'High'
        
        return jsonify({
            'detections': detections,
            'people_count': people_count,
            'crowd_density': crowd_density,
            'crowd_level': crowd_level
        })
    except Exception as e:
        print(f"Error during crowd detection: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'YOLOv8 detection API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)