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
# 4. Disease Information Dictionary
# ---------------------------
disease_info = {
    "demodicosis": {
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
    "Dermatitis": {
        "description": "Dermatitis is inflammation of the skin, often caused by allergies, irritants, or infections.",
        "symptoms": [
            "Itching and scratching",
            "Redness or swelling",
            "Dry or flaky patches",
            "Open sores from scratching"
        ],
        "treatment": [
            "Medicated shampoos to soothe the skin",
            "Antihistamines or corticosteroids (vet prescribed)",
            "Antibiotics if bacterial infection occurs",
            "Identify and remove allergens (e.g., food, fleas)"
        ]
    },
    "Fungal_infections": {
        "description": "Fungal infections are caused by fungi affecting the skin and hair.",
        "symptoms": [
            "Circular patches of hair loss",
            "Itching and redness",
            "Scaly or flaky skin",
            "Possible odor from infected areas"
        ],
        "treatment": [
            "Topical antifungal creams or shampoos",
            "Oral antifungal medications for severe cases",
            "Maintain good hygiene and a clean environment",
            "Regular grooming to prevent reinfection"
        ]
    },
    "Healthy": {
        "description": "The dog's skin appears healthy with no visible signs of disease or irritation.",
        "symptoms": [
            "Smooth coat with no hair loss",
            "No redness, swelling, or itching",
            "No unusual odor"
        ],
        "treatment": [
            "Maintain balanced nutrition and hydration",
            "Regular grooming and bathing",
            "Preventative care such as flea/tick control",
            "Routine vet check-ups"
        ]
    },
    "Hypersensitivity": {
        "description": "Hypersensitivity is an allergic reaction caused by fleas, food, or environmental allergens.",
        "symptoms": [
            "Intense itching and scratching",
            "Redness and swelling",
            "Skin rashes or hives",
            "Hair loss in affected areas"
        ],
        "treatment": [
            "Antihistamines or corticosteroids (vet prescribed)",
            "Identify and avoid allergens (food, fleas, pollen)",
            "Medicated shampoos to relieve itching",
            "Nutritional supplements to boost immunity"
        ]
    },
    "ringworm": {
        "description": "Ringworm is a contagious fungal infection that affects the skin, hair, and nails.",
        "symptoms": [
            "Circular bald patches",
            "Crusty or scaly skin",
            "Itching and discomfort",
            "Possible spreading to humans and other pets"
        ],
        "treatment": [
            "Topical antifungal creams or shampoos",
            "Oral antifungal medications if severe",
            "Clean and disinfect environment to prevent reinfection",
            "Limit contact with other pets until healed"
        ]
    }
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
            st.success(f"✅ Prediction: **{predicted_class}**")

            # Show disease information
            if predicted_class in disease_info:
                info = disease_info[predicted_class]
                st.subheader("📖 Disease Information")
                st.write(info["description"])
                
                st.subheader("🐾 Common Symptoms")
                for s in info["symptoms"]:
                    st.markdown(f"- {s}")
                
                st.subheader("💊 Treatment Ideas")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")
        else:
            st.error("Prediction failed. Please check your model and class names.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
