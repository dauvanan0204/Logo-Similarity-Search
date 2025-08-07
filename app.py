from flask import Flask, request, jsonify
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import io

# ==== Khởi tạo Flask ====
app = Flask(__name__)

# ==== Load model ResNet50 để trích đặc trưng ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights="IMAGENET1K_V1").to(device)
model = torch.nn.Sequential(*list(resnet.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(image: Image.Image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model(image).squeeze().cpu().numpy()
    return emb / np.linalg.norm(emb)

# ==== Load metadata đã xử lý trước ====
try:
    with open("embeddings_metadata.json", "r") as f:
        database = json.load(f)
    print(f"📦 Loaded metadata with {len(database)} entries.")
except Exception as e:
    print(f"❌ Lỗi khi đọc metadata: {e}")
    database = []

# ==== API nhận logo và tìm ảnh khớp ====
@app.route("/match_logo", methods=["POST"])
def match_logo():
    try:
        print("📥 Đã nhận request từ Streamlit")

        if "logo" not in request.files:
            print("❌ Không có file 'logo'")
            return jsonify({"error": "Missing logo file"}), 400

        logo_file = request.files["logo"]
        print("📁 Tên file nhận:", logo_file.filename)

        try:
            logo_image = Image.open(logo_file.stream).convert("RGB")
        except Exception as e:
            print(f"❌ Không đọc được ảnh: {e}")
            return jsonify({"error": "Invalid image format"}), 400

        # Tính embedding
        logo_embedding = get_embedding(logo_image)
        print("✅ Đã tính xong embedding")

        SIMILARITY_THRESHOLD = 0.75
        matched_images = {}

        for entry in database:
            emb = np.array(entry["embedding"])
            sim = cosine_similarity([logo_embedding], [emb])[0][0]
            if sim > SIMILARITY_THRESHOLD:
                img = entry["image"]
                if img not in matched_images:
                    matched_images[img] = []
                matched_images[img].append({
                    "bbox": entry["bbox"],
                    "similarity": round(sim, 3)
                })

        print(f"✅ Có {len(matched_images)} ảnh khớp")
        return jsonify({
            "matched_images": matched_images,
            "total": len(matched_images)
        })

    except Exception as e:
        print("🔥 Lỗi không xác định trong API:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=False)
