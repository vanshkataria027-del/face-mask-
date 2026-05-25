import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("mask_detector.h5")

st.title("Face Mask Detector")
st.write("Upload a face image to check if mask is worn or not.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128, 128))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    if prediction > 0.5:
        st.error("No Mask Detected!")
    else:
        st.success("Mask Detected!")