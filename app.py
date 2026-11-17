import io
import json
import numpy as np
import streamlit as st
import tensorflow as tf


from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "dree25/indian-bird-classifier-efficientnetv2m"
IMG_SIZE = (384, 384)
TOP_K = 3

st.title("Indian Bird Classifier")
st.write("Identifikasi spesies burung India menggunakan EfficientNetV2M.")

# Memuat model hanya sekali selama resource cache masih berlaku
@st.cache_resource
def load_model():
    hf_token = st.secrets["HF_TOKEN"]

    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_base_model.keras", token=hf_token)
    class_path = hf_hub_download(repo_id=REPO_ID, filename="class_names.json", token=hf_token)

    model = tf.keras.models.load_model(model_path, compile=False)

    with open(class_path, "r") as f:
        class_names = json.load(f)
    
    return model, class_names

def preprocess_uploaded(image_bytes):
    try:
        image = tf.io.decode_image(
            image_bytes,
            channels=3,
            expand_animations=False
        )

    except Exception as e:
        # Fallback jika format gambar tidak dapat dibaca TensorFlow
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            image = tf.convert_to_tensor(
                np.array(pil_image.convert("RGB"))
            )

    return tf.image.resize(image, IMG_SIZE)

# Memuat model
try:
    with st.spinner("Memuat model..."):
        model, class_names = load_model()
    
    st.success("Model berhasil dimuat!")

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# Upload gambar
uploaded_file = st.file_uploader(
    "Upload gambar burung",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    try:
        image_bytes = uploaded_file.getvalue()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        st.image(image, caption="Gambar input", use_container_width=True)

    except Exception as e:
        st.error(f"Gagal membaca gambar: {e}")
        st.stop()

    if st.button("Klasifikasi Jenis Burung"):
        try:
            with st.spinner("Melakukan klasifikasi"):
                # Preprocessing
                image_tensor = preprocess_uploaded(image_bytes)

                # Inference
                predictions = model.predict(
                    image_tensor[tf.newaxis, ...],
                    verbose=0
                )[0]

                if len(class_names) != len(predictions):
                    raise ValueError("Jumlah class names tidak sesuai dengan jumlah output model.")

                # Ranking prediksi
                top_indices = np.argsort(
                    predictions
                )[::-1][:TOP_K]
            
            # Menampilkan hasil
            st.subheader("Hasil Klasifikasi")

            for idx in top_indices:
                probability = float(predictions[idx])

                st.write(f"**{class_names[idx]}**")
                st.progress(probability)
                st.caption(f"Confidence: {probability:.2%}")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat inference: {e}")