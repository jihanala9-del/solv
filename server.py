import os
import io
import base64
import urllib.request
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

torch.set_num_threads(2)
torch.set_grad_enabled(False)

MODEL_FILE = "captcha_v4_native.pth"
MODEL_DOWNLOAD_URL = "https://github.com/jihanala9-del/solv/releases/download/v1.0/captcha_v4_native.pth"

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

device = torch.device("cpu")

# Download & Load Model saat startup server
if not os.path.exists(MODEL_FILE):
    print(f"Downloading {MODEL_FILE}...")
    try:
        req = urllib.request.Request(MODEL_DOWNLOAD_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as resp, open(MODEL_FILE, 'wb') as f:
            f.write(resp.read())
        print("Download completed!")
    except Exception as e:
        print("Download error:", e)

model = models.resnet34()
model.fc = nn.Linear(model.fc.in_features, 6)
if os.path.exists(MODEL_FILE):
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device))
    print("Model ResNet34 loaded successfully in memory!")
model.eval()

# Warm-up
dummy = torch.randn(1, 3, 280, 450)
_ = model(dummy)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "online", "service": "Captcha V4 PyTorch API (Fast)"})

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

        tensor = transform(img).unsqueeze(0)
        output = model(tensor)
        pred_class = int(output.argmax(dim=1).item())

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
