import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import joblib
from PIL import Image

# ---------------------------
# 1. Load MobileNetV2 feature extractor
# ---------------------------
base_model = keras.applications.MobileNetV2(
    input_shape=(224,224,3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

feature_extractor = keras.Sequential([
    keras.layers.Rescaling(1./255),
    base_model,
    keras.layers.GlobalAveragePooling2D()
])

# ---------------------------
# 2. Load Random Forest model (.pkl)
# ---------------------------
rf_model = joblib.load("dog_skin_rf_model.pkl")

# ---------------------------
# 3. Define class names
# ---------------------------
class_names = ['demodicosis','Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity','ringworm']  

# ---------------------------
# 4. Add Disease Information Dictionary
# ---------------------------
disease_info = {
    "demodicosis": "Demodicosis (mange) is caused by Demodex mites. Symptoms include hair loss, redness, and skin scaling. Treatment often involves medicated baths and oral medications.",
    "Dermatitis": "Dermatitis is skin inflammation. Symptoms include itching, redness, and rashes. Treatment may include antihistamines, medicated shampoos, or antibiotics if infected.",
    "Fungal_infections": "Fungal infections often cause circular patches of hair loss, itching, and scaling. Treatment includes antifungal creams, shampoos, and oral medications.",
    "Healthy": "The dog's skin looks healthy with no visible signs of infection or irritation. Maintain proper hygiene and nutrition to prevent diseases.",
    "Hypersensitivity": "Hypersensitivity (allergic reaction) can cause itching, redness, and swelling. Common causes are food, fleas, or environmental allergens. Treatment includes antihistamines and avoidance of allergens.",
    "ringworm": "Ringworm is a contagious fungal infection that causes circular patches of hair loss and scaly skin. Treatment includes antifungal creams, shampoos, and environmental cleaning."
}

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("🐶 Dog Skin Disease Classifier (CNN + Random Forest)")
st.write("Upload an image of a dog's skin to predict the disease type.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Load image
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    img_resized = img.resize((224, 224))
    img_array = image.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0)  # shape (1, 224, 224, 3)
    
    # Extract features using CNN
    features = feature_extractor(img_array).numpy()
    
    # Predict using Random Forest
    try:
        pred = rf_model.predict(features)
        if len(pred) > 0 and pred[0] < len(class_names):
            predicted_class = class_names[pred[0]]
            st.success(f"Prediction: **{predicted_class}**")

            # Show disease information
            if predicted_class in disease_info:
                st.subheader("📖 Disease Information")
                st.write(disease_info[predicted_class])
        else:
            st.error("Prediction failed. Please check your model and class names.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
