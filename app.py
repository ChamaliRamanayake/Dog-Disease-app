import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
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
# 4. Disease Information Dictionary (English + Sinhala)
# ---------------------------
disease_info = {
    "demodicosis": {
        "en": {
            "description": "Demodicosis (mange) is caused by Demodex mites that live in the hair follicles and skin.",
            "symptoms": [
                "Hair loss (patchy bald spots)",
                "Red, scaly, or crusty skin",
                "Itching and discomfort",
                "Secondary bacterial infections"
            ],
            "treatment": [
                "Medicated baths or dips prescribed by a vet",
                "Oral or topical anti-parasitic medications",
                "Antibiotics if secondary infection is present",
                "Regular vet check-ups to monitor progress"
            ]
        },
        "si": {
            "description": "Demodicosis (මාන්ජ්) යනු රෝම මූල හා සම තුළ ජීවත් වන Demodex මයිට්ස් නිසා සිදුවන රෝගයකි.",
            "symptoms": [
                "රෝම නැතිවීම (කොටස්වලින් ගිය බොල් තැන්)",
                "රතු, කොටු, හෝ අත්පත් සම",
                "ඇඟේ කැටකිරීම හා නොසන්සුන් වීම",
                "ද්විතීය බැක්ටීරියා ආසාදන"
            ],
            "treatment": [
                "වෙටින්ගෙන් ලැබෙන විශේෂ බාත් හෝ සෝකන ද්‍රව්‍ය",
                "මුඛ/පෘෂ්ඨ මගින් ලබාදෙන පරාසිතාහාරක",
                "ද්විතීය ආසාදන සඳහා ඇන්ටිබයෝටික්",
                "නිත්‍ය වෛද්‍ය පරීක්ෂණ"
            ]
        }
    },
    "Dermatitis": {
        "en": {
            "description": "Dermatitis is inflammation of the skin, often caused by allergies, irritants, or infections.",
            "symptoms": ["Itching and scratching","Redness or swelling","Dry or flaky patches","Open sores from scratching"],
            "treatment": ["Medicated shampoos to soothe the skin","Antihistamines or corticosteroids","Antibiotics if bacterial infection occurs","Identify and remove allergens"]
        },
        "si": {
            "description": "Dermatitis යනු සමේ ආසාදනයකි. බොහෝ විට ඇලර්ජි, රසායන හෝ ආසාදන නිසා සිදුවේ.",
            "symptoms": ["ඇඟේ කැටකිරීම හා කැටිම","රතු වීම හෝ පොටෝවීම","කැටි ගිය තැන්","ගාල් වල තුවාල"],
            "treatment": ["සම සන්සුන් කරන විශේෂ සබන්/ශැම්පු","ඇන්ටිහිස්ටමින් හෝ කෝටිසොයිඩ්","බැක්ටීරියා ආසාදන සඳහා ඇන්ටිබයෝටික්","ඇලර්ජි හඳුනාගැනීම හා ඉවත්කිරීම"]
        }
    },
    # 🐾 Add Sinhala & English for other diseases here in the same format...
}

# ---------------------------
# 5. Streamlit UI
# ---------------------------
st.title("🐶 Dog Skin Disease Classifier (CNN + Random Forest)")
st.write("Upload an image of a dog's skin to predict the disease type.")

# Language selection button
language = st.radio("🌐 Select Language / භාෂාව තෝරන්න:", ["English", "සිංහල"])
lang_key = "en" if language == "English" else "si"

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
            st.success(f"✅ Prediction: **{predicted_class}**")

            # Show disease information in selected language
            if predicted_class in disease_info:
                info = disease_info[predicted_class][lang_key]
                st.subheader("📖 Disease Information / රෝග තොරතුරු")
                st.write(info["description"])
                
                st.subheader("🐾 Common Symptoms / රෝග ලක්ෂණ")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
                
                st.subheader("💊 Treatment Ideas / ප්‍රතිකාර")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")
        else:
            st.error("Prediction failed. Please check your model and class names.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
