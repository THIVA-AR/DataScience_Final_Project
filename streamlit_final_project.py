import streamlit as st
import tempfile
import cv2
import numpy as np
from ultralytics import YOLO

# --------- CONFIG: UPDATE MODEL PATHS ----------
MAIN_MODEL_PATH   = r"C:\Users\user\Documents\guvi\guvi project 1\final_project\yolo_runs\military_3class_model\weights\best.pt"      # 3-class model
WEAPON_MODEL_PATH = r"C:\Users\user\Documents\guvi\guvi project 1\final_project\yolo_runs\weapon_model\weights\best.pt"       # 1-class weapon model

@st.cache_resource
def load_models():
    main_model   = YOLO(MAIN_MODEL_PATH)
    weapon_model = YOLO(WEAPON_MODEL_PATH)
    return main_model, weapon_model

main_model, weapon_model = load_models()


# --------- THREAT CLASSIFICATION ----------
def classify_threat(soldiers, weapons):
    n_soldiers = len(soldiers)
    n_weapons  = len(weapons)
    if n_weapons > 0:
        return f"HIGH THREAT: {n_weapons} weapon(s)", "red"
    if n_soldiers > 0:
        return f"MEDIUM: {n_soldiers} soldier(s)", "orange"
    return "LOW / NON-THREAT", "green"


def process_frame(frame_bgr):
    """Run both models on one frame and draw boxes + threat label."""
    frame = frame_bgr.copy()

    # Main multi-class model
    res_main = main_model(frame, conf=0.35, iou=0.5, verbose=False)[0]
    soldiers, vehicles, trenches = [], [], []

    if res_main.boxes is not None:
        for b in res_main.boxes:
            cls_id = int(b.cls)
            conf   = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            if cls_id == 0:        # soldier
                soldiers.append((x1, y1, x2, y2, conf))
            elif cls_id == 2:      # vehicle
                vehicles.append((x1, y1, x2, y2, conf))
            elif cls_id == 3:      # trench (if present)
                trenches.append((x1, y1, x2, y2, conf))

    # Weapon specialist model
    res_wp = weapon_model(frame, conf=0.25, iou=0.45, verbose=False)[0]
    weapons = []
    if res_wp.boxes is not None:
        for b in res_wp.boxes:
            conf   = float(b.conf)
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            weapons.append((x1, y1, x2, y2, conf))

    # Threat level
    threat_text, threat_color = classify_threat(soldiers, weapons)

    # Draw boxes
    for x1, y1, x2, y2, conf in weapons:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"weapon {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for x1, y1, x2, y2, conf in soldiers:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"soldier {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for x1, y1, x2, y2, conf in vehicles:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"vehicle {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for x1, y1, x2, y2, conf in trenches:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"trench {conf:.2f}", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Threat banner
    # cv2.rectangle(frame, (0, 0), (360, 35), (0, 0, 0), -1)
    # cv2.putText(frame, threat_text, (10, 25),
               # cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame[:, :, ::-1], threat_text  # return RGB for Streamlit


# --------- STREAMLIT UI ----------
st.title("Military Object Detection & Threat Classification")

st.markdown(
    "Upload an **image** or **video** to detect soldiers, weapons, and vehicles, "
    "and classify the scene as threat / non-threat."
)

mode = st.radio("Select input type:", ["Image", "Video"])

uploaded_file = st.file_uploader(
    "Upload file",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
    accept_multiple_files=False
)

if uploaded_file is not None:
    if mode == "Image" and uploaded_file.type.startswith("image"):    
        # ---------- IMAGE PIPELINE ----------
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        processed_rgb, threat_text = process_frame(img_bgr)

        st.subheader("Threat classification")
        st.markdown(f"**{threat_text}**")   # text ABOVE image

        st.subheader("Detection result")
        st.image(processed_rgb, caption="Detected objects", use_container_width=True)


        # Download button
        _, tmp_path = tempfile.mkstemp(suffix=".png")
        cv2.imwrite(tmp_path, processed_rgb[:, :, ::-1])
        with open(tmp_path, "rb") as f:
            st.download_button(
                label="Download result image",
                data=f,
                file_name="detection_result.png",
                mime="image/png"
            )

    elif mode == "Video" and uploaded_file.type.startswith("video"):
        # ---------- VIDEO PIPELINE ----------
        # Save uploaded video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.flush()

        cap = cv2.VideoCapture(tfile.name)

        st_frame = st.empty()
        last_threat = "Processing..."

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        out_writer = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed_rgb, threat_text = process_frame(frame)
            last_threat = threat_text

            # Show in app
            st_frame.image(processed_rgb, channels="RGB")

            # Init writer lazily
            if out_writer is None:
                h, w, _ = processed_rgb.shape
                out_writer = cv2.VideoWriter(out_temp.name, fourcc, 20, (w, h))

            out_writer.write(processed_rgb[:, :, ::-1])  # back to BGR for saving

        cap.release()
        if out_writer is not None:
            out_writer.release()

        st.subheader("Final threat assessment")
        st.write(last_threat)

        # Download processed video
        with open(out_temp.name, "rb") as f:
            st.download_button(
                label="Download result video",
                data=f,
                file_name="detection_result.mp4",
                mime="video/mp4"
            )
    else:
        st.warning(f"Uploaded file type does not match selected mode '{mode}'.")
