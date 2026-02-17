import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model  # TF2-kompatibel
from PIL import Image, ImageOps
import os

st.set_page_config(page_title="Tier-Erkennung", page_icon="🐴")

# Modell laden (nur einmal)
@st.cache_resource
def load_my_model():
    # Lade das konvertierte TF2-Modell (SavedModel-Format)
    model_path = "keras_Model_TF2"
    if not os.path.exists(model_path):
        st.error("Modellordner 'keras_Model_TF2' nicht gefunden!")
        return None
    model = load_model(model_path, compile=False)
    return model

model = load_my_model()

# Falls das Modell nicht geladen werden konnte, abbrechen
if model is None:
    st.stop()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = f.readlines()

st.title("🐴 Pferd, Zebra oder Esel Erkennung")

uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Bild öffnen und in RGB konvertieren
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bildgröße anpassen
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)

    # Normalisieren (-1 bis 1)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)

    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]

    st.subheader("🔎 Ergebnis")
    st.write(f"**Tier:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")

    # Optional: Balkendiagramm aller Wahrscheinlichkeiten
    st.subheader("📊 Wahrscheinlichkeiten")
    probs = {class_names[i][2:].strip(): float(prediction[0][i]) for i in range(len(class_names))}
    st.bar_chart(probs)
