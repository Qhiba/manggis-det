import os
import cv2
import base64
import json
import numpy as np
from io import BytesIO
from ultralytics import YOLO
from flask import Flask, Response, jsonify, request, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = YOLO('./weights/manggisv8n.pt')
detection_enabled = False


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_enabled
    detection_enabled = not detection_enabled
    return jsonify({'detection_enabled': detection_enabled})


@app.route('/detect', methods=['POST'])
def detect():
    try:
        data = json.loads(request.data)
        image_data = data['image'].split(',')[1]

        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        results = model(frame)

        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]

                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': conf,
                    'class': name
                })
        
        return jsonify({
            'success': True,
            'detections': detections
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500  


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)