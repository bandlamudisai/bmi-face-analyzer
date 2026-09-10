import numpy as np
import joblib
import streamlit as st
from PIL import Image
import face_recognition

st.set_page_config(page_title="Face BMI Estimator", page_icon="🧑", layout="centered")


@st.cache_resource
def load_models():
    height_model = joblib.load("height_predictor.model")
    weight_model = joblib.load("weight_predictor.model")
    bmi_model = joblib.load("bmi_predictor.model")
    return height_model, weight_model, bmi_model


def bmi_category(bmi):
    if bmi < 18.5:
        return "Skinny / Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def predict(image, height_model, weight_model, bmi_model):
    image_array = np.array(image.convert("RGB"))
    encodings = face_recognition.face_encodings(image_array)
    if len(encodings) == 0:
        return None
    X = np.expand_dims(encodings[0], axis=0)
    height = float(np.exp(height_model.predict(X)[0]))
    weight = float(np.exp(weight_model.predict(X)[0]))
    bmi = float(np.exp(bmi_model.predict(X)[0]))
    return {"height_m": round(height, 2), "weight_kg": round(weight, 1), "bmi": round(bmi, 1)}


height_model, weight_model, bmi_model = load_models()

st.title("Face BMI Estimator")
uploaded_file = st.file_uploader("Upload a face photo", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Predict"):
        result = predict(image, height_model, weight_model, bmi_model)
        if result is None:
            st.error("No face detected. Try a clearer, front-facing photo.")
        else:
            category = bmi_category(result["bmi"])
            col1, col2, col3 = st.columns(3)
            col1.metric("Height", f"{result['height_m']} m")
            col2.metric("Weight", f"{result['weight_kg']} kg")
            col3.metric("BMI", result["bmi"])
            st.success(f"Category: {category}")
