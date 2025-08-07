import os
import json
from PIL import Image
from dotenv import load_dotenv
from minio import Minio
from inference_sdk import InferenceHTTPClient
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

# ======= CẤU HÌNH =======
load_dotenv()
API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "logo_detection-ydqxq/5"  # Hoặc thử logo_detection-ydqxq/2 nếu model này không detect được
SIMILARITY_THRESHOLD = 0.75
TEMP_FOLDER = "minio_images"
os.makedirs(TEMP_FOLDER, exist_ok=True)

# ======= MINIO CONFIG =======
client = Minio(
    endpoint="localhost:9000",
    access_key="minioadmin",
    secret_key="minioadmin",
    secure=False
)
bucket_name = "logo-matching"
prefix = "test/"

# ======= ROBOFLOW DETECTION CLIENT =======
detector = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=API_KEY
)

# ======= RESNET50 EMBEDDING MODEL =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = models.resnet50(weights="IMAGENET1K_V1").to(device)
embedding_model = torch.nn.Sequential(*list(resnet.children())[:-1])
embedding_model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def get_embedding(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = embedding_model(image).squeeze().cpu().numpy()
    return embedding / np.linalg.norm(embedding)

# ======= BẮT ĐẦU XỬ LÝ =======
metadata = []
objects = client.list_objects(bucket_name, prefix=prefix, recursive=True)

for obj in objects:
    if not obj.object_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    file_name = os.path.basename(obj.object_name)
    local_path = os.path.join(TEMP_FOLDER, file_name)

    # Tải từ MinIO
    try:
        client.fget_object(bucket_name, obj.object_name, local_path)
        print(f"⬇️  Đã tải: {obj.object_name}")
    except Exception as e:
        print(f"[❌] Không tải được ảnh từ MinIO: {obj.object_name} → {e}")
        continue

    # Mở ảnh
    try:
        image = Image.open(local_path).convert("RGB")
    except Exception as e:
        print(f"[❌] Không mở được ảnh: {file_name} → {e}")
        continue

    img_width, img_height = image.size

    # Gửi ảnh đến Roboflow để detect logo
    try:
        result = detector.infer(local_path, model_id=MODEL_ID)
        print(f"📤 Đã gửi {file_name} đến Roboflow.")
        print("📩 Kết quả Roboflow:", json.dumps(result, indent=2))

        predictions = result.get("predictions", [])
        print(f"🔍 {file_name}: phát hiện {len(predictions)} vùng logo")

        if not predictions:
            continue

        for pred in predictions:
            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
            left = max(0, x - w / 2)
            top = max(0, y - h / 2)
            right = min(img_width, x + w / 2)
            bottom = min(img_height, y + h / 2)

            cropped = image.crop((left, top, right, bottom))
            embedding = get_embedding(cropped)

            metadata.append({
                "image": file_name,
                "bbox": [left, top, right, bottom],
                "embedding": embedding.tolist()
            })

    except Exception as e:
        print(f"[❌] Lỗi xử lý {file_name}: {e}")

# ======= LƯU METADATA =======
print(f"\n📦 Tổng số vùng logo được trích xuất: {len(metadata)}")
with open("test_embeddings_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
print("✅ Đã lưu embeddings_metadata.json thành công.")
