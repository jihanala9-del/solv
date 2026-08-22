import os
import io
import base64
import urllib.request
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

ONNX_MODEL_FILE = "captcha_v4_native.onnx"
ONNX_DOWNLOAD_URL = "https://github.com/jihanala9-del/solv/releases/download/v1.0/captcha_v4_native.onnx"

# 1. Download model ONNX ringan (hanya 200 KB, download cuma 0.2 detik)
if not os.path.exists(ONNX_MODEL_FILE):
    print("Downloading lightweight ONNX model...")
    req = urllib.request.Request(ONNX_DOWNLOAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(ONNX_MODEL_FILE, 'wb') as out_file:
        out_file.write(response.read())
    print("ONNX Model downloaded successfully!")

# 2. Inisialisasi ONNX Runtime (CPU super cepat, <5ms inferensi)
opts = ort.SessionOptions()
opts.intra_op_num_threads = 2
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(ONNX_MODEL_FILE, sess_options=opts, providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
print("ONNX Runtime initialized and ready!")

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

def preprocess_image(img):
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1)) # HWC to CHW
    return np.expand_dims(arr, axis=0) # [1, 3, 280, 450]

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "engine": "ONNX Runtime (Ultra Fast)"})

@app.route("/solve", methods=["POST"])
def solve_api():
    try:
        data = request.json
        image_b64 = data.get("image")
        prompt_text = data.get("prompt", "")
        sequence = data.get("sequence", [])

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

        # 2. ONNX Inference (< 5 milidetik)
        input_data = preprocess_image(img)
        outputs = session.run(None, {input_name: input_data})
        pred_class = int(np.argmax(outputs[0][0]))

        mapping_tuple = PERMUTATIONS[pred_class]

        color_to_right_idx = {
            left_nodes[0]: mapping_tuple[0],
            left_nodes[1]: mapping_tuple[1],
            left_nodes[2]: mapping_tuple[2],
        }

        target_color = None
        for c in PALETTE_V4.keys():
            if c in prompt_text.upper():
                target_color = c
                break

        if not target_color and sequence and len(sequence) > 0:
            target_color = sequence[0].upper()

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
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=False)
