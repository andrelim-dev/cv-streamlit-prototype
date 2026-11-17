import io
import json
import html
import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "dree25/indian-bird-classifier-efficientnetv2m"
IMG_SIZE = (384, 384)
TOP_K = 3

# ---------------------------------------------------------------------------
# Konfigurasi halaman (harus menjadi pemanggilan Streamlit pertama)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Indian Bird Classifier",
    page_icon="🐦",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --paper: #F7F5EF;
        --ink: #1F2A24;
        --muted: #6B7A71;
        --forest: #2F6B4F;
        --forest-dark: #1F3D2F;
        --amber: #D98E2B;
        --line: #E2DED2;
        --card: #FFFFFF;
    }

    .stApp {
        background: var(--paper);
        font-family: 'Inter', sans-serif;
    }

    .block-container {
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }

    /* ---------- Hero ---------- */
    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.1rem 2.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, var(--forest-dark) 0%, var(--forest) 62%, #3E7F5E 100%);
        color: var(--paper);
        margin-bottom: 0.9rem;
        box-shadow: 0 12px 30px -18px rgba(31, 61, 47, 0.6);
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -70px;
        top: -70px;
        width: 240px;
        height: 240px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(217, 142, 43, 0.45), transparent 70%);
    }

    .hero-eyebrow {
        font-size: 0.74rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        font-weight: 600;
        color: #F2C98A;
    }

    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.6rem;
        line-height: 1.1;
        margin: 0.4rem 0 0.45rem;
        color: #FFFFFF;
    }

    .hero-sub {
        margin: 0;
        font-size: 1.02rem;
        color: rgba(247, 245, 239, 0.85);
    }

    /* ---------- Status pill ---------- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.32rem 0.8rem;
        border-radius: 999px;
        background: #E4EFE8;
        color: var(--forest);
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 1.6rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #3FA36B;
        box-shadow: 0 0 0 3px rgba(63, 163, 107, 0.2);
    }

    /* ---------- Section label ---------- */
    .section-label {
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--forest);
        margin: 0.2rem 0 0.7rem;
    }

    /* ---------- File uploader ---------- */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background: var(--card);
        border: 1.5px dashed #B9C7BD;
        border-radius: 16px;
    }

    [data-testid="stFileUploader"] section:hover,
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--forest);
    }

    /* ---------- Image preview ---------- */
    [data-testid="stImage"] img {
        border-radius: 16px;
        border: 1px solid var(--line);
    }

    /* ---------- Button ---------- */
    .stButton > button {
        width: 100%;
        padding: 0.75rem 1rem;
        border-radius: 12px;
        border: none;
        background: var(--forest);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1rem;
        transition: background 0.15s ease, transform 0.15s ease;
    }

    .stButton > button:hover {
        background: #24563F;
        color: #FFFFFF;
        transform: translateY(-1px);
    }

    .stButton > button:focus:not(:active) {
        color: #FFFFFF;
        border: none;
    }

    /* ---------- Result: top prediction card ---------- */
    .top-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 5px solid var(--amber);
        border-radius: 16px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.4rem;
    }

    .top-label {
        font-size: 0.74rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: var(--amber);
    }

    .top-name {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 1.9rem;
        line-height: 1.2;
        color: var(--ink);
        margin: 0.3rem 0 0.2rem;
    }

    .top-conf {
        font-size: 1.05rem;
        font-weight: 600;
        color: var(--forest);
    }

    .top-conf span {
        font-weight: 400;
        color: var(--muted);
    }

    /* ---------- Result: ranking rows ---------- */
    .rank-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: var(--muted);
        margin-bottom: 0.6rem;
    }

    .pred-row {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 0.85rem 1rem;
        margin-bottom: 0.6rem;
    }

    .pred-head {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin-bottom: 0.55rem;
    }

    .pred-rank {
        flex: none;
        width: 26px;
        height: 26px;
        border-radius: 8px;
        background: #EDEAE0;
        color: var(--ink);
        font-size: 0.8rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .pred-row.first .pred-rank {
        background: var(--amber);
        color: #FFFFFF;
    }

    .pred-name {
        flex: 1;
        font-weight: 600;
        color: var(--ink);
    }

    .pred-pct {
        font-variant-numeric: tabular-nums;
        font-weight: 600;
        color: var(--forest);
    }

    .bar {
        height: 8px;
        border-radius: 999px;
        background: #E7E3D7;
        overflow: hidden;
    }

    .fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--forest), #5FA37E);
    }

    .pred-row.first .fill {
        background: linear-gradient(90deg, var(--amber), #F2B866);
    }

    /* ---------- Empty state ---------- */
    .empty-state {
        border: 1.5px dashed var(--line);
        border-radius: 16px;
        padding: 2.6rem 1.5rem;
        text-align: center;
        color: var(--muted);
        background: rgba(255, 255, 255, 0.5);
    }

    .empty-icon {
        font-size: 2.2rem;
        margin-bottom: 0.4rem;
    }

    .empty-title {
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.2rem;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: #EFECE3;
        border-right: 1px solid var(--line);
    }

    .side-item {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid var(--line);
        font-size: 0.9rem;
    }

    .side-item span:first-child { 
        color: var(--muted); 
    }

    .side-item span:last-child { 
        font-weight: 600; 
        color: var(--ink); 
    }

    /* ---------- Footer ---------- */
    .app-footer {
        margin-top: 3rem;
        text-align: center;
        font-size: 0.8rem;
        color: var(--muted);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Fungsi inti (tidak diubah)
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# Helper tampilan
# ---------------------------------------------------------------------------
def render_top_card(name, probability):
    st.markdown(
        '<div class="top-card">'
        '<div class="top-label">Prediksi teratas</div>'
        f'<div class="top-name">{html.escape(str(name))}</div>'
        f'<div class="top-conf">{probability:.2%} <span>confidence</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )

def render_prediction_row(rank, name, probability):
    width = max(0.0, min(probability, 1.0)) * 100
    row_class = "pred-row first" if rank == 1 else "pred-row"
    st.markdown(
        f'<div class="{row_class}">'
        '<div class="pred-head">'
        f'<div class="pred-rank">{rank}</div>'
        f'<div class="pred-name">{html.escape(str(name))}</div>'
        f'<div class="pred-pct">{probability:.2%}</div>'
        '</div>'
        f'<div class="bar"><div class="fill" style="width: {width:.2f}%"></div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

def render_empty_state(icon, title, subtitle):
    st.markdown(
        '<div class="empty-state">'
        f'<div class="empty-icon">{icon}</div>'
        f'<div class="empty-title">{title}</div>'
        f'<div>{subtitle}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🐦 Tentang Model")
    st.markdown(
        '<div class="side-item"><span>Arsitektur</span><span>EfficientNetV2M</span></div>'
        f'<div class="side-item"><span>Ukuran input</span><span>{IMG_SIZE[0]} × {IMG_SIZE[1]}</span></div>'
        f'<div class="side-item"><span>Top-K</span><span>{TOP_K}</span></div>'
        '<div class="side-item"><span>Format</span><span>JPG · JPEG · PNG</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown("### 💡 Tips")
    st.caption(
        "Gunakan foto dengan burung terlihat jelas, tidak terlalu jauh, "
        "dan pencahayaan cukup untuk hasil yang lebih akurat."
    )
    st.markdown("")
    st.caption(f"Model: [{REPO_ID}](https://huggingface.co/{REPO_ID})")

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">EfficientNetV2M · Image Classification</div>'
    '<div class="hero-title">Indian Bird Classifier</div>'
    '<p class="hero-sub">Identifikasi spesies burung India menggunakan EfficientNetV2M.</p>'
    '</div>',
    unsafe_allow_html=True,
)

# Memuat model
try:
    with st.spinner("Memuat model..."):
        model, class_names = load_model()

    st.markdown(
        '<div class="status-pill"><span class="status-dot"></span>Model berhasil dimuat!</div>',
        unsafe_allow_html=True,
    )

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ---------------------------------------------------------------------------
# Layout utama
# ---------------------------------------------------------------------------
col_input, col_result = st.columns([1, 1], gap="large")

classify_clicked = False

with col_input:
    st.markdown('<div class="section-label">01 · Upload gambar</div>', unsafe_allow_html=True)

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

        classify_clicked = st.button("Klasifikasi Jenis Burung")

with col_result:
    st.markdown('<div class="section-label">02 · Hasil klasifikasi</div>', unsafe_allow_html=True)

    if uploaded_file and classify_clicked:
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
            best_idx = top_indices[0]
            render_top_card(class_names[best_idx], float(predictions[best_idx]))

            st.markdown(f'<div class="rank-title">Top-{TOP_K} prediksi</div>', unsafe_allow_html=True)

            for rank, idx in enumerate(top_indices, start=1):
                probability = float(predictions[idx])
                render_prediction_row(rank, class_names[idx], probability)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat inference: {e}")

    elif uploaded_file:
        render_empty_state(
            "🔍",
            "Gambar siap diklasifikasi",
            "Klik tombol <b>Klasifikasi Jenis Burung</b> untuk melihat hasil.",
        )

    else:
        render_empty_state(
            "🪶",
            "Belum ada gambar",
            "Upload foto burung di panel kiri untuk memulai.",
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-footer">Dibuat dengan Streamlit · TensorFlow · Hugging Face Hub</div>',
    unsafe_allow_html=True,
)