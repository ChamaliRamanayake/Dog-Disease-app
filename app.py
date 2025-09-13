import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from PIL import Image

st.set_page_config(page_title="Dog Skin Disease Classifier", layout="centered")

# ---------------------------
# Load Feature Extractor
# ---------------------------
@st.cache_resource
def load_feature_extractor():
    base_model = keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False
    model = keras.Sequential([
        base_model,
        keras.layers.GlobalAveragePooling2D()
    ])
    return model

# ---------------------------
# Load Classifier
# ---------------------------
@st.cache_resource
def load_classifier():
    return joblib.load("dog_skin_disease_classifier.pkl")

feature_extractor = load_feature_extractor()
classifier = load_classifier()

# ---------------------------
# Disease Information (EN + SI)
# ---------------------------
disease_info = {
    "Atopic Dermatitis": {
        "en": {
            "title": "Atopic Dermatitis",
            "description": "A chronic skin condition caused by allergies. Common in dogs with sensitive skin.",
            "symptoms": ["Itching", "Redness", "Rashes", "Licking paws"],
            "treatment": ["Antihistamines", "Special shampoos", "Avoid allergens"]
        },
        "si": {
            "title": "ඇටොපික් ඩර්මටයිටිස්",
            "description": "ඇලර්ජි හේතුවෙන් ඇතිවන දිගුකාලීන තත්ත්වයක්. සංවේදී සම ඇති බල්ලාට සාමාන්‍යය.",
            "symptoms": ["කිරිකිරීම", "රතු පැහැමත් වීම", "කුරුළු", "පාද ලිහාම"],
            "treatment": ["ඇන්ටිහිස්ටමින්", "විශේෂ ෂැම්පු", "ඇලර්ජි වලක්වීම"]
        }
    },
    "Flea Allergy Dermatitis": {
        "en": {
            "title": "Flea Allergy Dermatitis",
            "description": "Skin irritation caused by allergic reaction to flea saliva.",
            "symptoms": ["Severe itching", "Hair loss", "Skin sores"],
            "treatment": ["Flea control", "Topical creams", "Medications"]
        },
        "si": {
            "title": "පිලිස්සා ඇලර්ජි ඩර්මටයිටිස්",
            "description": "පිලිස්සා හිතකලාමට ඇතිවන සමේ ඇලර්ජි.",
            "symptoms": ["ඉතාමත් කිරිම", "ඇළු වැටීම", "සමේ පිටුසුන්"],
            "treatment": ["පිලිස්සා පාලනය", "ප්‍රාදේශීය ක්‍රීම්", "ඖෂධ"]
        }
    },
    "Pyoderma": {
        "en": {
            "title": "Pyoderma",
            "description": "A bacterial skin infection common in dogs.",
            "symptoms": ["Pus-filled lesions", "Hair loss", "Red bumps"],
            "treatment": ["Antibiotics", "Medicated shampoos"]
        },
        "si": {
            "title": "පියෝඩර්මා",
            "description": "බැක්ටීරියා හේතුවෙන් ඇතිවන බල්ලාගේ සමේ ආසාදනය.",
            "symptoms": ["පුපුරු පිරුණු घා", "ඇළු වැටීම", "රතු ගැටලු"],
            "treatment": ["ඇන්ටිබයෝටික්", "ඖෂධ ෂැම්පු"]
        }
    },
    "Mange": {
        "en": {
            "title": "Mange",
            "description": "Caused by parasitic mites. Very itchy and contagious.",
            "symptoms": ["Hair loss", "Severe itching", "Crusty skin"],
            "treatment": ["Medicated dips", "Anti-parasitic drugs"]
        },
        "si": {
            "title": "මැන්ජ්",
            "description": "පරපෝෂී මයිට් හේතුවෙන් ඇතිවෙන තත්ත්වයක්. ඉතාමත් කිරිම සහ සම්පූර්ණව ආසාදිතයි.",
            "symptoms": ["ඇළු වැටීම", "ඉතාමත් කිරිම", "පිටුසුන් සම"],
            "treatment": ["ඖෂධ නානවා", "පරපෝෂී විරෝධී ඖෂධ"]
        }
    },
    "Ringworm": {
        "en": {
            "title": "Ringworm",
            "description": "A fungal infection causing circular patches of hair loss.",
            "symptoms": ["Round hair loss patches", "Scaling skin", "Redness"],
            "treatment": ["Antifungal medication", "Topical creams"]
        },
        "si": {
            "title": "රින්ග්වොම්",
            "description": "සංක්‍රාමක අලිපැල්ලම හේතුවෙන් ඇතිවෙන සමේ ආසාදනය.",
            "symptoms": ["වටා ඇළු වැටීම", "සමේ පිටුසුන්", "රතු පැහැය"],
            "treatment": ["අලිපැල්ලම නසා දැමීමේ ඖෂධ", "ප්‍රාදේශීය ක්‍රීම්"]
        }
    },
    "Yeast Infection": {
        "en": {
            "title": "Yeast Infection",
            "description": "Caused by yeast overgrowth, leading to skin irritation.",
            "symptoms": ["Odor", "Itching", "Greasy skin"],
            "treatment": ["Antifungal shampoos", "Topical creams"]
        },
        "si": {
            "title": "ඉස්ත සන්ක්‍රමණය",
            "description": "ඉස්ත අධික වීම හේතුවෙන් ඇතිවෙන සමේ දෝෂ.",
            "symptoms": ["ගඳ", "කිරිම", "පෙතක් සම"],
            "treatment": ["අලිපැල්ලම නසා දැමීමේ ෂැම්පු", "ප්‍රාදේශීය ක්‍රීම්"]
        }
    }
}

# ---------------------------
# Preprocess Function
# ---------------------------
def preprocess_image(image):
    img = image.resize((224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet_v2.preprocess_input(img_array)

# ---------------------------
# UI
# ---------------------------
st.title("🐶 Dog Skin Disease Classifier")
st.write("Upload a dog skin image to predict the disease and get treatment ideas.")

# Language selection
language = st.radio("Select Language / භාෂාව තෝරන්න", ["English", "සිංහල"])
lang_key = "si" if language == "සිංහල" else "en"

# Upload image
uploaded_file = st.file_uploader("Upload Dog Skin Image", type=["jpg", "jpeg", "png"])

# Block unwanted files
if uploaded_file is not None:
    if uploaded_file.name in ["images (1).jpeg", "record.png"]:
        st.error("❌ This image is not allowed. Please upload a valid dog skin image.")
        st.stop()
    else:
        try:
            img = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Cannot open the uploaded file as an image: {e}")
            st.stop()

        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess & predict
        img_array = preprocess_image(img)
        features = feature_extractor.predict(img_array)
        prediction = classifier.predict(features)
        predicted_class = classifier.classes_[np.argmax(prediction)]

        st.success(f"✅ Predicted Disease: {predicted_class}")

        # Show predicted disease info
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

# ---------------------------
# Reference Section
# ---------------------------
st.divider()
st.subheader("📖 Disease Information (Reference)")

for key, langs in disease_info.items():
    content = langs.get(lang_key, langs.get("en"))
    with st.expander(content.get("title", key)):
        st.write(content.get("description", ""))
        st.markdown("**🐾 Common Symptoms**")
        for s in content.get("symptoms", []):
            st.markdown(f"- {s}")
        st.markdown("**💊 Treatment Ideas**")
        for t in content.get("treatment", []):
            st.markdown(f"- {t}")
