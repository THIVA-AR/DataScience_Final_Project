import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# -----------------------------
# 1. Page Configuration
# -----------------------------
st.set_page_config(page_title="🎖️ Military Object Detection", layout="wide")

st.title("🎖️ Military Object Detection System")
st.markdown("Upload **images or videos** to detect **soldiers, weapons, vehicles, and trenches**.")

# -----------------------------
# 2. Load YOLO Model
# -----------------------------
# 👉 Change this path to your trained model
MODEL_PATH = r"C:\Users\user\OneDrive\Documents\guvi\guvi project 1\final project for data science\yolov8n.pt"

if not os.path.exists(MODEL_PATH):
    st.error(f"❌ Model not found at `{MODEL_PATH}`. Please update MODEL_PATH.")
    st.stop()

model = YOLO(MODEL_PATH)

# -----------------------------
# 3. File Uploader
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an Image or Video",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mpeg4"]
)

# -----------------------------
# 4. Handle Uploaded File
# -----------------------------
if uploaded_file is not None:
    file_type = uploaded_file.type

    # Save file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
        tmp.write(uploaded_file.read())
        file_path = tmp.name

    # -----------------------------
    # 5. If Image
    # -----------------------------
    if file_type.startswith("image/"):
        img = Image.open(file_path)
        st.image(img, caption="📷 Uploaded Image", use_container_width=True)

        # Run YOLOv8 inference
        results = model.predict(source=file_path, save=False, conf=0.5)

        # Display results
        st.subheader("Detected Objects")
        for r in results:
            st.image(r.plot(), caption="Detections", use_container_width=True)

    # -----------------------------
    # 6. If Video
    # -----------------------------
    elif file_type.startswith("video/"):
        st.video(file_path)

        st.info("⏳ Running YOLOv8 inference on video... please wait.")
        results = model.predict(source=file_path, save=True, conf=0.5)

        # YOLO saves processed video in runs/detect/ directory
        output_dir = results[0].save_dir
        detected_video = os.path.join(output_dir, uploaded_file.name)

        if os.path.exists(detected_video):
            st.success("✅ Detection complete. Processed video below:")
            st.video(detected_video)
        else:
            st.error("⚠️ Processed video not found.")

    else:
        st.warning("Unsupported file type. Please upload an image or video.")
