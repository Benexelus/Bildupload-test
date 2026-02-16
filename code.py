import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Modell laden
@st.cache_resource
def load_my_model():
    model = tf.keras.models.load_model("keras_Model.h5", compile=False)
    return model

model = load_my_model()

# Labels laden
with open("labels.txt", "r") as f:
    class_names = f.readlines()

st.title("🐴 Zebra, Esel oder Pferd Erkennung")

uploaded_file = st.file_uploader("Lade ein Bild hoch", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Hochgeladenes Bild", use_column_width=True)

    # Bild vorbereiten
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Vorhersage
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index][2:].strip()
    confidence_score = prediction[0][index]

    st.subheader("Ergebnis:")
    st.write(f"**Tier:** {class_name}")
    st.write(f"**Sicherheit:** {confidence_score * 100:.2f}%")
