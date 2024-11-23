import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(
    page_title="Breast Cancer Classification", 
    page_icon="🧬", 
    layout="centered", 
    initial_sidebar_state="collapsed",
)

model = load_model('modelo.h5')

def preprocess_image(image):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)  
    image = cv2.resize(image, (128, 128))  
    image = np.expand_dims(image, axis=-1)  
    image = image / 255.0  
    image = np.expand_dims(image, axis=0)  
    return image

def predict_image(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    label = np.argmax(prediction, axis=1)  
    return label[0], prediction

st.title("Algorithm for detecting tumors")
st.write("Upload an image to find out if the tumor is benign or malignant.")
st.write("This algorithm is experimental, do not diagnose yourself under any circumstances and go see a doctor.")


uploaded_file = st.file_uploader("choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='image uploaded', use_container_width=True)

    if st.button("make predictionn"):
        label, prediction = predict_image(image)
        if label == 0:
            st.write("The tumor is **Benign**")
        else:
            st.write("the tumor is **Malignant**")
        
        st.write(f"Probability of Benign: {prediction[0][0]:.2f}")
        st.write(f"Probability of Malignant: {prediction[0][1]:.2f}")

