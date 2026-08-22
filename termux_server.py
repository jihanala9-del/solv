import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

try:
    import onnxruntime as ort
except ImportError:
    print("Silakan jalankan: pip install onnxruntime")
    exit(1)

app = Flask(__name__)
CORS(app)

MODEL_FILE = "captcha_v4_native.onnx"

PERMUTATIONS = [
    (0, 1, 2), (0, 2, 1), (1, 0, 2),
    (1, 2, 0), (2, 0, 1), (2, 1, 0)
]

PALETTE_V4 = {
    'RED': (239, 68, 68),
    'GREEN': (34, 197, 94),
    'BLUE': (59, 130, 246),
    'YELLOW': (234, 179, 8),
    'ORANGE': (249, 115, 22),
    'PURPLE': (168, 85, 247)
}

TARGET_COORDS = [
    {'x': 400, 'y': 50},   # Top (0)
    {'x': 400, 'y': 140},  # Middle (1)
    {'x': 400, 'y': 230}   # Bottom (2)
]

print("⚡ Memuat model ONNX di Termux...")
session = ort.InferenceSession(MODEL_FILE, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("✅ Server Termux Siap & Online di port 5000!")

def detect_node_color(img, x, y):
    pixel = img.getpixel((x, y))[:3]
    best_color = None
    min_dist = float('inf')
    for name, rgb in PALETTE_V4.items():
        dist = sum((p - r) ** 2 for p, r in zip(pixel, rgb))
        if dist < min_dist:
            min_dist = dist
            best_color = name
    return best_color

def preprocess(img):
    img = img.convert("RGB")
    if img.size != (450, 280):
        img = img.resize((450, 280), Image.BILINEAR)
    arr = np.array(img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)) # HWC to CHW
    return np.expand_dims(arr, axis=0) # Add batch

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "mode": "Termux Localhost"})

@app.route("/solve", methods=["POST"])
def solve_api():
    try:
        data = request.json
        image_b64 = data.get("image")
        prompt_text = data.get("prompt", "")

        if not image_b64:
            return jsonify({"status": "error", "message": "No image data"}), 400

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        image_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        if img.size != (450, 280):
            img = img.resize((450, 280), Image.BILINEAR)

        # 1. Deteksi warna kiri & kanan
        left_nodes = [
            detect_node_color(img, 50, 50),
            detect_node_color(img, 50, 140),
            detect_node_color(img, 50, 230)
        ]
        right_nodes = [
            detect_node_color(img, 400, 50),
            detect_node_color(img, 400, 140),
            detect_node_color(img, 400, 230)
        ]

        # 2. Inferensi ONNX (~10-20ms di HP)
        input_tensor = preprocess(img)
        outputs = session.run(None, {input_name: input_tensor})
        pred_class = int(np.argmax(outputs[0][0]))

        mapping_tuple = PERMUTATIONS[pred_class]
        color_to_right_idx = {
            left_nodes[0]: mapping_tuple[0],
            left_nodes[1]: mapping_tuple[1],
            left_nodes[2]: mapping_tuple[2],
        }

        # 3. Tentukan target warna
        target_color = None
        for c in PALETTE_V4.keys():
            if c in prompt_text.upper():
                target_color = c
                break

        if not target_color:
            target_color = left_nodes[0]

        target_right_idx = color_to_right_idx.get(target_color, mapping_tuple[0])
        click_coord = TARGET_COORDS[target_right_idx]

        return jsonify({
            "status": "success",
            "target_color": target_color,
            "target_idx": target_right_idx,
            "destination_color": right_nodes[target_right_idx],
            "click": click_coord,
            "mapping": color_to_right_idx
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
