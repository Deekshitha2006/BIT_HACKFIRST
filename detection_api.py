from flask import Flask, request, jsonify
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

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'message': 'YOLOv8 detection API is running'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
