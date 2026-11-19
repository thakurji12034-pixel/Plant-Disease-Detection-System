import os
import sys
import subprocess

# Auto-use project virtual environment if invoked with global python
if sys.prefix == sys.base_prefix:
    _venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
    if os.path.exists(_venv_python) and os.path.normcase(sys.executable) != os.path.normcase(_venv_python):
        sys.exit(subprocess.call([_venv_python] + sys.argv))

os.environ["KERAS_BACKEND"] = "torch"
import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import keras 

app = Flask(__name__)
CORS(app)

print("Loading model...")
model = keras.models.load_model('model.keras')
print("Model loaded successfully.")

input_shape = model.input_shape
if input_shape is None or len(input_shape) < 3 or input_shape[1] is None:
    img_size = (224, 224)
else:
    img_size = (input_shape[1], input_shape[2])

CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Late Blight",
    "Leaf Mold",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Healthy Leaves"
]

@app.route('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fsd project.html'))

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), filename)

@app.route('/health')
def health():
    return jsonify({"status": "ok", "model_ready": model is not None})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"prediction": "No file uploaded", "confidence": "0%"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"prediction": "No file selected", "confidence": "0%"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        img = img.resize(img_size)
        
        img_array = np.array(img, dtype=np.float32)
        
        # MobileNetV2 preprocessing: inputs are scaled between -1 and 1
        img_array = (img_array / 127.5) - 1.0
        
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array)
        score = predictions[0] 
        
        class_idx = np.argmax(score)
        
        # Convert logits to probabilities if necessary
        if np.sum(score) < 0.9 or np.sum(score) > 1.1:
            exp_score = np.exp(score - np.max(score))
            score = exp_score / sum(exp_score)
            
        confidence = 100 * score[class_idx]

        if class_idx < len(CLASS_NAMES):
            predicted_class = CLASS_NAMES[class_idx]
        else:
            predicted_class = f"Class {class_idx}"

        return jsonify({
            "prediction": predicted_class,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"prediction": "Error processing image", "confidence": "0%"}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=port)
