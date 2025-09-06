import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import joblib
from PIL import Image
import os

st.set_page_config(page_title="Dog Skin Disease Classifier", layout="centered")

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


class_names = ['demodicosis', 'Dermatitis', 'Fungal_infections', 'Healthy', 'Hypersensitivity', 'ringworm']


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
    "Dermatitis": {
        "en": {
            "title": "Dermatitis",
            "description": "Dermatitis is skin inflammation caused by allergies, irritants, or infection.",
            "symptoms": [
                "Itching and scratching",
                "Redness and swelling",
                "Dry or flaky patches",
                "Open sores from intense scratching"
            ],
            "treatment": [
                "Medicated shampoos to soothe skin",
                "Antihistamines or corticosteroids (vet prescribed)",
                "Antibiotics if bacterial infection is present",
                "Identify and remove allergens (food/fleas/environmental)"
            ]
        },
        "si": {
            "title": "Dermatitis (ස්කින් දුෂ්ඨතාවය)",
            "description": "Dermatitis යනු ඇලර්ජි, ආශිලක හෝ ආසාදන වැනි හේතු මත සිදුවන සමේ දායමකි.",
            "symptoms": [
                "කැටිම හා පිරිම්පීම",
                "රතු වීම සහ ද්‍රවත්වීම",
                "කැකුළු හෝ ගැඹුරු තැන්",
                "ගැටිම නිසා ඇතිවන තුවාල"
            ],
            "treatment": [
                "සම සන්සුන් කරන ශැම්පු හා බාත්",
                "ඇන්ටිහිස්ටමිනයන් හෝ කොටිසොයිඩ් (වෙට් නියමිත)",
                "බැක්ටීරියා ආසාදන ඇත්නම් ඇන්ටිබයොටික්",
                "ඇලර්ජි හඳුනාගෙන ඉවත් කිරීම"
            ]
        }
    },
    "Fungal_infections": {
        "en": {
            "title": "Fungal Infections",
            "description": "Skin fungal infections (e.g. ringworm-like infections) cause patches of hair loss and scaling.",
            "symptoms": [
                "Circular or irregular patches of hair loss",
                "Itching and redness",
                "Scaly or flaky skin",
                "Sometimes unpleasant odor"
            ],
            "treatment": [
                "Topical antifungal creams or medicated shampoos",
                "Oral antifungal medication for widespread cases",
                "Clean and disinfect bedding and environment",
                "Keep the pet dry and well-groomed"
            ]
        },
        "si": {
            "title": "බීජාණු ආසාදන",
            "description": "බීජාණු ආසාදන (දෘශ්‍ය වශයෙන් රවුම් වර්ගයේ) සමේ රෝම නැතිවීම් සහ පසුබැසීම සිදු කරයි.",
            "symptoms": [
                "රවුම් හෝ අනියමිත රෝම නැතිකිරීම්",
                "කැටිම සහ රතු වීම",
                "කැකිළි හෝ උඩිමැටි සම",
                "නියම නොවන ගන්දරයක් ඇති විය හැක"
            ],
            "treatment": [
                "පෘෂ්ඨ/කැලෑම antifungal කිරිම්",
                "වයාපෘති අවස්ථාවන්හි මුඛ ඖෂධ",
                "බැඩින් සහ පරිසරය පිරිසිදු කිරීම",
                "වැසි නැති හා සුදුසු ගෘහ පරිපාලනය"
            ]
        }
    },
    "Healthy": {
        "en": {
            "title": "Healthy Skin",
            "description": "No visible signs of disease. Skin and coat appear normal.",
            "symptoms": [
                "Full, glossy coat",
                "No redness, sores or scaling",
                "No persistent itching"
            ],
            "treatment": [
                "Balanced diet and hydration",
                "Regular grooming and flea/tick prevention",
                "Routine vet check-ups"
            ]
        },
        "si": {
            "title": "සෞඛ්‍ය සම්පුර්ණ සම",
            "description": "කිසිදු පෙනෙන රෝග ලක්ෂණ නොමැති අතර සම හා රෝම සාමාන්‍ය ලෙස පෙනේ.",
            "symptoms": [
                "සම්පූර්ණ සහ මිහිරි රෝම",
                "රතුකිරීම, තුවාල හෝ කැකිළි නොමැති වීම",
                "පාහේ කැටිමක් නොමැති වීම"
            ],
            "treatment": [
                "සමබැඳි ආහාර හා ජලය",
                "නිතිපතා සොබාදහමින් සෝදන හා පිරිසිදු කිරීම",
                "නිති වෛද්‍ය පරීක්ෂණ"
            ]
        }
    },
    "Hypersensitivity": {
        "en": {
            "title": "Hypersensitivity (Allergy)",
            "description": "Allergic reactions to fleas, food, or environmental allergens causing skin problems.",
            "symptoms": [
                "Severe itching and scratching",
                "Redness, rashes or hives",
                "Hair loss in irritated areas",
                "Secondary infections may occur"
            ],
            "treatment": [
                "Antihistamines or steroids prescribed by a vet",
                "Flea control if fleas are the cause",
                "Elimination diet to identify food allergies",
                "Medicated shampoos and topical care"
            ]
        },
        "si": {
            "title": "හයිපර් සේන්සිටිවිටි (ඇලර්ජි)",
            "description": "පිළිකිලි, ආහාර හෝ පරිසරික ඇලර්ජි මඟින් ඇතිවන සම ප්‍රතිචාර.",
            "symptoms": [
                "තිව්‍ර කැටිම හා පිරිම්පීම",
                "රතුකිරීම, රැස් වීම හෝ හයිව්ස්",
                "බලාගන hair නැතිවීම",
                "දෙවනික ආසාදන ඇති විය හැක"
            ],
            "treatment": [
                "වෙට් නියමිත ඇන්ටිහිස්ටමින් හෝ ස්ටෙරොයිඩ්",
                "පිළිකිලි පාලනය (නිසා නම්)",
                "ආහාර හඳුනාගැනීම සඳහා අහාර හැරීමේ පරීක්ෂණ",
                "ශැම්පු සහ තවත් topical ප්‍රතිකාර"
            ]
        }
    },
    "ringworm": {
        "en": {
            "title": "Ringworm",
            "description": "Ringworm is a contagious fungal infection that affects skin and hair.",
            "symptoms": [
                "Circular bald patches",
                "Scaly, crusty skin",
                "Itching and possible spread to other animals/humans"
            ],
            "treatment": [
                "Topical antifungal creams/shampoos",
                "Oral antifungals for extensive cases",
                "Disinfect environment and isolate infected pets",
                "Wash bedding and toys frequently"
            ]
        },
        "si": {
            "title": "Ringworm (චක්රාරූපී ආසාදනය)",
            "description": "Ringworm යනු සන්ක්‍රමණීය බීජාණු ආසාදනයකි, සම සහ රෝම ක්ෂේම කරයි.",
            "symptoms": [
                "රවුම් හිස් තැන්",
                "කැකිළි සහ දුඹුරු සම",
                "කැටිම සහ අන් සතුන්/මිනිසුන් වෙත පැතිරීම"
            ],
            "treatment": [
                "පෘෂ්ඨ antifungal කිරිම/ශැම්පු",
                "විශාල අවශ්‍යතාවයකදී මුඛ antifungal ඖෂධ",
                "පරිසරය පිරිසිදු කිරීම හා ආසාදිත සතුන් වෙන් කිරීම",
                "බැදි, ක්‍රීඩා ද්‍රව්‍ය නිතර සෝදන්න"
            ]
        }
    }
}


st.title("🐶 Dog Skin Disease Classifier")
st.write("Upload a dog's skin image — choose language, then predict.")

language = st.radio("🌐 Select language / භාෂාව:", ["English", "සිංහල"], horizontal=True)
lang_key = "en" if language == "English" else "si"

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Cannot open the uploaded file as an image: {e}")
        st.stop()

    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized).astype(np.float32)
    img_array = np.expand_dims(img_array, 0)

    
    try:
        features = feature_extractor(img_array).numpy()
    except Exception as e:
        st.error(f"Feature extraction failed: {e}")
        st.stop()

    
    try:
        pred = rf_model.predict(features)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pred0 = pred[0]
    predicted_class = None

   
    if isinstance(pred0, (np.integer, int)):
        idx = int(pred0)
        if 0 <= idx < len(class_names):
            predicted_class = class_names[idx]
        else:
            predicted_class = str(pred0)


    if predicted_class is None:
        if isinstance(pred0, (bytes, np.bytes_)):
            try:
                pred0 = pred0.decode("utf-8")
            except Exception:
                pass
        if isinstance(pred0, str) and pred0 in class_names:
            predicted_class = pred0
        else:
            if isinstance(pred0, str):
                normalized = pred0.strip()
                for key in disease_info.keys():
                    if normalized.lower() == key.lower():
                        predicted_class = key
                        break
    if predicted_class is None:
        predicted_class = str(pred0)

    st.success(f"✅ Prediction: **{predicted_class}**")


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

