import streamlit as st
import requests
from PIL import Image
import io

# ===== CẤU HÌNH =====
API_ENDPOINT = "http://127.0.0.1:5000/match_logo"  # Flask API endpoint
# MINIO_HOST = "http://localhost:9000" 
MINIO_HOST = "https://311c8c5d06d1.ngrok-free.app"  # Địa chỉ public MinIO
BUCKET_NAME = "logo-matching"
IMAGE_PREFIX = "images_to_check"

# ===== GIAO DIỆN =====
st.set_page_config(page_title="Logo Matching", layout="centered")
st.title("🔍 Logo Matching với MinIO + AI")
st.write("Upload một ảnh logo và hệ thống sẽ tìm ảnh tương ứng trong kho MinIO.")

# ===== FORM UPLOAD LOGO =====
uploaded_file = st.file_uploader("📁 Chọn ảnh logo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    logo = Image.open(uploaded_file)
    st.image(logo, caption="Ảnh logo", use_container_width=True)

    if st.button("🚀 Tìm ảnh khớp"):
        with st.spinner("Đang xử lý..."):
            files = {"logo": (uploaded_file.name, uploaded_file.getvalue(), "image/jpeg")}
            response = requests.post(API_ENDPOINT, files=files)

        if response.status_code != 200:
            st.error(f"❌ Lỗi từ server Flask: {response.status_code} - {response.text}")
        else:
            result = response.json()
            matched = result.get("matched_images", {})
            total = result.get("total", 0)

            if total == 0:
                st.warning("Không tìm thấy ảnh nào khớp.")
            else:
                st.success(f"✅ Tìm thấy {total} ảnh khớp.")
                for img_name, matches in matched.items():
                    minio_url = f"{MINIO_HOST}/{BUCKET_NAME}/{IMAGE_PREFIX}/{img_name}"
                    st.markdown(f"🖼️ **{img_name}** - `{len(matches)} vùng khớp`")
                    st.markdown(f"[📥 Mở ảnh trên MinIO]({minio_url})")
                    st.write("---")
