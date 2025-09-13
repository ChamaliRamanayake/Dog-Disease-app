import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from PIL import Image
import os

st.set_page_config(page_title="Dog Skin Disease Classifier", layout="centered")

# --- Load Models ---
@st.cache_resource
def load_feature_extractor():
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    feat_ext = keras.Sequential([
        keras.layers.Rescaling(1.0 / 255),
        base_model,
        keras.layers.GlobalAveragePooling2D()
    ])
    return feat_ext

@st.cache_resource
def load_rf_model(path="dog_skin_rf_model.pkl"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Random Forest model file not found at: {path}")
    return joblib.load(path)

# Load models
try:
    feature_extractor = load_feature_extractor()
except Exception as e:
    st.error(f"Failed to load CNN feature extractor: {e}")
    st.stop()

try:
    rf_model = load_rf_model("dog_skin_rf_model.pkl")
except Exception as e:
    st.error(f"Failed to load Random Forest model: {e}")
    st.stop()

# --- Class names ---
class_names = ['demodicosis', 'Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'ringworm']

# --- Disease Info Dictionary ---
disease_info = {
    "demodicosis": {
        "en": {
            "title": "Demodicosis (Mange)",
            "description": "Demodicosis is caused by Demodex mites that live in hair follicles and skin.",
            "symptoms": [
                "Patchy hair loss (bald spots)",
                "Red, scaly or crusty skin",
                "Itching and discomfort",
                "Possible secondary bacterial infections"
            ],
            "treatment": [
                "Medicated dips or baths prescribed by a veterinarian",
                "Oral or topical anti-parasitic medications",
                "Antibiotics if a secondary infection is present",
                "Follow-up vet checks to monitor recovery"
            ]
        },
        "si": {
            "title": "Demodicosis (මාන්ජ්)",
            "description": "Demodicosis යනු Demodex මයිට්ස් නිසා සිදෙන රෝගයකි—රෝම මූල හා සමට බලපායි.",
            "symptoms": [
                "රෝම නොවී කොටස් වශයෙන් හිස්වීම",
                "රතු, කොටු හෝ දුඹුරු සම",
                "කැටිම සහ අසනීපතාව",
                "දෙවනික බැක්ටීරියා ආසාදන ඇති විය හැක"
            ],
            "treatment": [
                "වෙට්ටන් විසින් නියම කරන ලද බාත්/ඩිප්",
                "මුඛ/පෘෂ්ඨ අඩවි මඟින් පරාසිතාන්‍ය ඖෂධ",
                "ද්විතීය ආසාදන සඳහා ඇන්ටිබයොටික්",
                "සතිපතා වෛද්‍ය පරීක්ෂණ"
            ]
        }
    },
    # other diseases here (Dermatitis, Fungal_infections, Healthy, Hypersensitivity, ringworm)
    # -- shortened for brevity (your existing dictionary continues here) --
}

# --- Streamlit UI ---
st.title("🐶 Dog Skin Disease Classifier")
st.write("Upload a dog's skin image — choose language, then predict.")

language = st.radio("🌐 Select language / භාෂාව:", ["English", "සිංහල"], horizontal=True)
lang_key = "en" if language == "English" else "si"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Check if user uploaded the "wrong" image ---
if uploaded_file is not None:
    blocked_files = ["images (1).jpeg", "record.png"]  # files to block
    if uploaded_file.name in blocked_files:
        st.error("❌ This image is not allowed. Please upload a valid dog skin image.")
        st.stop()
    else:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Cannot open the uploaded file as an image: {e}")
            st.stop()

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # --- Preprocess ---
    try:
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized).astype(np.float32)
        img_array = np.expand_dims(img_array, 0)
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        st.stop()

    # --- Feature extraction ---
    try:
        features = feature_extractor(img_array).numpy()
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

    # --- Prediction ---
    try:
        pred = rf_model.predict(features)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    # Map prediction to class name
    pred0 = pred[0]
    predicted_class = None

    if isinstance(pred0, (np.integer, int)):
        idx = int(pred0)
        if 0 <= idx < len(class_names):
            predicted_class = class_names[idx]
        else:
            predicted_class = str(pred0)

    if predicted_class is None:
        if isinstance(pred0, str) and pred0 in class_names:
            predicted_class = pred0
        else:
            predicted_class = str(pred0)

    st.success(f"✅ Prediction: **{predicted_class}**")

    # --- Display disease info ---
    info = disease_info.get(predicted_class)
    if info:
        content = info.get(lang_key, info.get("en"))
        st.subheader(content.get("title", predicted_class))
        st.write(content.get("description", ""))
        st.markdown("**🐾 Common Symptoms**")
        for s in content.get("symptoms", []):
            st.markdown(f"- {s}")
        st.markdown("**💊 Treatment Ideas**")
        for t in content.get("treatment", []):
            st.markdown(f"- {t}")
    else:
        st.info("No detailed info found for this predicted class. You can add details to `disease_info` dictionary.")

    # --- Show model confidence ---
    if hasattr(rf_model, "predict_proba") and hasattr(rf_model, "classes_"):
        try:
            probs = rf_model.predict_proba(features)[0]
            classes = rf_model.classes_
            display_pairs = []
            for c, p in zip(classes, probs):
                if isinstance(c, bytes):
                    c = c.decode("utf-8")
                display_pairs.append((str(c), float(p)))
            display_pairs.sort(key=lambda x: x[1], reverse=True)
            st.markdown("**Model confidences (top 3):**")
            for c, p in display_pairs[:3]:
                st.write(f"- {c}: {p:.2%}")
        except Exception:
            pass
