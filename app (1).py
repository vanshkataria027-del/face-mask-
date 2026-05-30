import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

model = tf.keras.models.load_model("mask_detector.h5")

st.title("Face Mask Detector")

mode = st.radio("Choose Mode", ["Upload Image", "Live Camera"])

def predict(image):
    img = image.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0][0]
    return prediction

# ── Upload Image Mode ─────────────────────────────────────────────────────────
if mode == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        prediction = predict(image)
        if prediction > 0.5:
            st.error("❌ No Mask Detected!")
        else:
            st.success("✅ Mask Detected!")

# ── Live Camera Mode ──────────────────────────────────────────────────────────
elif mode == "Live Camera":
    st.write("Webcam se live detection ho rahi hai. **Stop** karne ke liye button dabao.")

    run = st.checkbox("Start Camera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Webcam nahi mila!")
            break

        frame = cv2.flip(frame, 1)

        # Predict on current frame
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        prediction = predict(pil_img)

        # Draw label on frame
        if prediction > 0.5:
            label = "No Mask"
            color = (0, 0, 255)   # Red
        else:
            label = "Mask"
            color = (0, 255, 0)   # Green

        cv2.putText(frame, label, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.rectangle(frame, (10, 10), (300, 70), color, 2)

        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
