import asyncio
import base64
import json
import os
import re
import tempfile
import time

import nest_asyncio
import numpy as np
import streamlit as st
import tensorflow as tf
from groq import Groq
import edge_tts

nest_asyncio.apply()

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  — put your Groq key here
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "best_efficientnet.keras")
KB_PATH    = os.path.join(BASE_DIR, "monuments_knowledge_base.json")
CONFIG_PATH    = os.path.join(BASE_DIR, "config.json")
IMG_SIZE   = (224, 224)
TTS_VOICE  = "en-GB-SoniaNeural"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    GROQ = f.read()
    GROQ = json.loads(GROQ)
GROQ_API_KEY = GROQ["GROQ_API_KEY"]

CLASS_NAMES = [
    'Ahmose I', 'Akhenaten', 'Al-Azhar Mosque', 'Al-Muizz Street',
    'Alexandria Library', 'Amenemhat I', 'Amenhotep III', 'Babylon Fortress',
    'Bent pyramid for senefru', 'Cairo Citadel', 'Cairo Tower',
    'Colossoi of Memnon', 'Dahshur Pyramids', 'Deir el-Bahari',
    'Dendera Temple Complex', 'Djoser', 'Egyptian Museum, Cairo',
    'Goddess Isis with her child',
    'Hanging Church (St. Virgin Mary Coptic Orthodox Church)',
    'Hatshepsut face', 'Horemheb', 'Isis Temple at Behbeit El Hagar',
    'Karnak Temple', 'Khafre', 'Khafre Pyramid', 'Merneptah',
    'Montaza Palace', 'Narmer (Menes)', 'Philae Temple',
    'Pyramid_of_Djoser', 'Pyramids of Giza', 'Qaitbay Citadel',
    'Ramesses III', 'Ramessum', 'Ramses II (Ramses the Great)',
    'Ras Muhammad National Park', 'Saqqara Pyramid',
    'Serapeum of Saqqara', 'Sesostris III', 'Sobekneferu', 'Sphinx',
    'St. Catherines Monastery', 'Statue of King Zoser',
    'Statue of Tutankhamun with Ankhesenamun', 'Temple of Habu',
    'Temple of Hathor', 'Temple of Hatshepsut',
    'Temple of Horus at Edfu', 'Temple of Seti I at Abydos',
    'Temple of the Oracle of Amun at Siwa', 'Temple_of_Isis_in_Philae',
    'Temple_of_Kom_Ombo', 'The Great Temple of Ramesses II',
    'Tutankhamun (King Tut)', 'Wadi Natrun Monasteries', 'White Desert',
    'amenhotep iii and tiye', 'khufu statue', 'menkaure pyramid',
]

# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    try:
        raw = json.loads(content)
        return {k.lower(): v for k, v in raw.items()}
    except json.JSONDecodeError:
        pass
    kb   = {}
    keys = re.findall(r'^\s*"([^"]+)"\s*:\s*\{', content, re.MULTILINE)
    for key in keys:
        pattern = re.escape(f'"{key}"') + r'\s*:\s*(\{.*?\n\s*\})'
        match   = re.search(pattern, content, re.DOTALL)
        if not match:
            continue
        block = match.group(1)
        def extract(field, text=block):
            m = re.search(rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"', text, re.DOTALL)
            return m.group(1) if m else ""
        kb[key.lower()] = {
            "class_name":         extract("class_name")         or key,
            "official_name":      extract("official_name")      or key,
            "location":           extract("location"),
            "detailed_knowledge": extract("detailed_knowledge"),
        }
    return kb


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def predict_monument(image_bytes: bytes, model):
    img   = tf.image.decode_image(image_bytes, channels=3, expand_animations=False)
    img   = tf.image.resize(img, IMG_SIZE)
    batch = tf.expand_dims(img, axis=0)
    probs = model.predict(batch, verbose=0)[0]
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx]


def get_monument_info(predicted_class: str, kb: dict) -> dict:
    key = predicted_class.lower().strip()
    if key in kb:
        return kb[key]
    for kb_key, val in kb.items():
        if key in kb_key or kb_key in key:
            return val
    return {
        "class_name":         predicted_class,
        "official_name":      predicted_class,
        "location":           "Egypt",
        "detailed_knowledge": f"{predicted_class} is a remarkable Egyptian monument.",
    }


def generate_narration(predicted_class: str, info: dict) -> str:
    client  = Groq(api_key=GROQ_API_KEY)
    context = (
        f"Name: {info['official_name']}\n"
        f"Location: {info['location']}\n"
        f"Details: {info['detailed_knowledge'][:1500]}"
    )
    resp = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an enthusiastic museum tour guide. "
                    "Generate exactly ~100 words of engaging spoken narration about the given "
                    "Egyptian monument. No bullet points, no headers — just natural speech "
                    "a visitor would love to hear."
                ),
            },
            {
                "role": "user",
                "content": f"Tell me about {predicted_class}.\n\n{context}",
            },
        ],
    )
    return resp.choices[0].message.content


async def _tts_to_bytes(text: str, voice: str) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name
    comm = edge_tts.Communicate(text, voice)
    await comm.save(tmp_path)
    with open(tmp_path, "rb") as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def text_to_speech(text: str, voice: str = TTS_VOICE) -> bytes:
    return asyncio.get_event_loop().run_until_complete(_tts_to_bytes(text, voice))


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EgyptTour — AI Monument Guide",
    page_icon="𓂀",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# FULL CSS — EgyptTour inspired
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@400;600;700&family=Lato:ital,wght@0,300;0,400;0,700;1,300&display=swap');

/* ── Reset & body ── */
html, body, [class*="css"] {
    font-family: 'Lato', sans-serif;
    background-color: #f5efe6;
    color: #2c1a0e;
}
[data-testid="stAppViewContainer"] {
    background: #f5efe6;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Top navigation bar ── */
.navbar {
    background: #fff;
    border-bottom: 2px solid #c8960c;
    padding: 0 40px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    height: 68px;
    margin: -1rem -1rem 0 -1rem;
    box-shadow: 0 2px 12px rgba(200,150,12,0.12);
}
.navbar-brand {
    display: flex;
    align-items: center;
    gap: 10px;
}
.navbar-brand-name {
    font-family: 'Cinzel', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: #1a4731;
    line-height: 1.1;
}
.navbar-brand-sub {
    font-size: 0.7rem;
    color: #c8960c;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.navbar-ankh {
    font-size: 2rem;
    color: #c8960c;
}
.navbar-links {
    display: flex;
    gap: 28px;
    font-size: 0.88rem;
    color: #5a3e28;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.navbar-active { color: #1a4731; border-bottom: 2px solid #1a4731; padding-bottom: 2px; }

/* ── Hero section ── */
.hero {
    background: linear-gradient(rgba(10,6,2,0.48), rgba(10,6,2,0.58)),
                url('https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Kheops-Pyramid.jpg/1280px-Kheops-Pyramid.jpg');
    background-size: cover;
    background-position: center 30%;
    border-radius: 14px;
    padding: 72px 48px;
    margin: 24px 0 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 14px;
    border: 2px solid rgba(200,150,12,0.5);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Cinzel', serif;
    font-size: 0.78rem;
    letter-spacing: 4px;
    color: #c8960c;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero h1 {
    font-family: 'Cinzel', serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #fff;
    line-height: 1.2;
    margin: 0 0 10px;
    text-shadow: 0 2px 16px rgba(0,0,0,0.7);
}
.hero h1 span { color: #f5d76e; }
.hero-sub {
    font-size: 1.05rem;
    color: #d4c09a;
    font-style: italic;
    margin-bottom: 0;
    font-weight: 300;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Cinzel', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1a4731;
    text-align: center;
    margin: 36px 0 6px;
    position: relative;
}
.section-heading::after {
    content: '';
    display: block;
    width: 56px;
    height: 3px;
    background: linear-gradient(90deg, #c8960c, #f5d76e);
    border-radius: 2px;
    margin: 8px auto 0;
}
.section-sub {
    text-align: center;
    color: #7a5c3a;
    font-size: 0.9rem;
    margin-bottom: 28px;
    font-style: italic;
}

/* ── Upload card ── */
.upload-card {
    background: #fff;
    border: 2px dashed #c8960c;
    border-radius: 14px;
    padding: 36px 32px;
    text-align: center;
    transition: border-color 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 18px rgba(200,150,12,0.08);
    margin-bottom: 12px;
}
.upload-card:hover { border-color: #1a4731; box-shadow: 0 6px 24px rgba(26,71,49,0.12); }
.upload-icon { font-size: 2.8rem; margin-bottom: 10px; }
.upload-title { font-family: 'Cinzel', serif; color: #1a4731; font-size: 1rem; margin-bottom: 4px; }
.upload-hint { color: #9a7a5a; font-size: 0.82rem; }

/* ── Image preview ── */
.preview-wrap {
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid #c8960c;
    box-shadow: 0 8px 32px rgba(200,150,12,0.18);
    margin: 18px 0;
}

/* ── Identify button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a4731 0%, #2d7a54 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 40px !important;
    padding: 14px 42px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    letter-spacing: 1px !important;
    width: 100% !important;
    transition: all 0.25s !important;
    box-shadow: 0 4px 18px rgba(26,71,49,0.28) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #c8960c 0%, #f5d76e 100%) !important;
    color: #1a0a00 !important;
    box-shadow: 0 6px 24px rgba(200,150,12,0.38) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: #d8ccbe !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Step progress ── */
.progress-card {
    background: #fff;
    border: 1px solid #e8d5a3;
    border-radius: 12px;
    padding: 20px 28px;
    margin: 18px 0;
    box-shadow: 0 2px 12px rgba(200,150,12,0.08);
}
.step-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 7px 0;
    font-size: 0.9rem;
    color: #5a3e28;
}
.step-dot {
    width: 12px; height: 12px;
    border-radius: 50%;
    flex-shrink: 0;
}
.step-dot.done    { background: #1a4731; }
.step-dot.active  { background: #c8960c; animation: pulse 1s infinite; }
.step-dot.pending { background: #e8d5a3; border: 2px solid #c8960c; }
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.85)} }

/* ── Result hero card ── */
.result-hero {
    background: linear-gradient(135deg, #0e2a1e 0%, #1a4731 60%, #0e2a1e 100%);
    border-radius: 16px;
    padding: 38px 40px 32px;
    text-align: center;
    margin: 28px 0 18px;
    border: 1px solid rgba(200,150,12,0.4);
    box-shadow: 0 8px 32px rgba(14,42,30,0.28);
    position: relative;
    overflow: hidden;
}
.result-hero::before {
    content: '𓂀';
    position: absolute;
    top: -10px; right: 20px;
    font-size: 6rem;
    opacity: 0.06;
    color: #f5d76e;
}
.result-eyebrow {
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #c8960c;
    text-transform: uppercase;
    margin-bottom: 10px;
    font-family: 'Cinzel', serif;
}
.result-name {
    font-family: 'Cinzel', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f5d76e;
    margin-bottom: 10px;
    line-height: 1.2;
}
.result-location {
    font-size: 0.9rem;
    color: #a8c9b8;
    font-style: italic;
    margin-bottom: 0;
}
.result-divider {
    width: 60px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #c8960c, transparent);
    margin: 18px auto;
}

/* ── Narration card ── */
.narration-card {
    background: #fff;
    border-radius: 14px;
    padding: 30px 34px;
    margin: 0 0 20px;
    box-shadow: 0 4px 20px rgba(200,150,12,0.1);
    border: 1px solid #e8d5a3;
    position: relative;
}
.narration-card::before {
    content: '"';
    position: absolute;
    top: 10px; left: 20px;
    font-size: 5rem;
    color: #f5d76e;
    font-family: 'Cinzel', serif;
    line-height: 1;
    opacity: 0.5;
}
.narration-label {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #c8960c;
    text-transform: uppercase;
    margin-bottom: 14px;
}
.narration-text {
    font-size: 1.02rem;
    line-height: 1.85;
    color: #3a2410;
    font-style: italic;
    font-weight: 300;
    padding-left: 8px;
}

/* ── Audio section ── */
.audio-card {
    background: linear-gradient(135deg, #fff9f0, #fff);
    border-radius: 14px;
    padding: 24px 28px;
    border: 1px solid #e8d5a3;
    box-shadow: 0 4px 16px rgba(200,150,12,0.08);
    margin-bottom: 20px;
}
.audio-label {
    font-family: 'Cinzel', serif;
    font-size: 0.75rem;
    letter-spacing: 3px;
    color: #c8960c;
    text-transform: uppercase;
    margin-bottom: 14px;
}
audio {
    width: 100%;
    border-radius: 8px;
    outline: none;
}
audio::-webkit-media-controls-panel { background: #fdf6ea; }

/* ── Download button ── */
.stDownloadButton > button {
    background: transparent !important;
    color: #1a4731 !important;
    border: 2px solid #1a4731 !important;
    border-radius: 40px !important;
    padding: 10px 30px !important;
    font-family: 'Cinzel', serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
.stDownloadButton > button:hover {
    background: #1a4731 !important;
    color: #fff !important;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: transparent;
    border-bottom: 2px solid #e8d5a3;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Cinzel', serif;
    font-size: 0.85rem;
    letter-spacing: 0.5px;
    color: #7a5c3a !important;
    background: transparent !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 20px !important;
    border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #1a4731 !important;
    color: #fff !important;
}

/* ── Footer ── */
.site-footer {
    background: #0e2a1e;
    color: #a8c9b8;
    text-align: center;
    padding: 28px 20px;
    border-radius: 14px;
    margin-top: 52px;
    font-size: 0.82rem;
    letter-spacing: 0.5px;
}
.site-footer .footer-brand {
    font-family: 'Cinzel', serif;
    color: #f5d76e;
    font-size: 1rem;
    margin-bottom: 6px;
}
.site-footer .footer-ankh { font-size: 1.5rem; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-ankh">𓋹</div>
        <div>
            <div class="navbar-brand-name">MonuVision</div>
            <div class="navbar-brand-sub">AI Monument Guide</div>
        </div>
    </div>
    <div class="navbar-links">
        <span class="navbar-active">Home</span>
        <span>Explore</span>
        <span>Monuments</span>
        <span>Interactive Map</span>
        <span>About</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">𓂀 &nbsp; Discover Ancient Egypt &nbsp; 𓂀</div>
    <h1>Explore the Wonders of<br><span>Ancient Egyptian Civilization</span></h1>
    <p class="hero-sub">Scan the Past 📸. Hear the Story 🎧</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("Loading AI model…"):
    model = load_model()
kb = load_knowledge_base()

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE INPUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-heading">Identify a Monument</div>
<div class="section-sub">Upload a photo or use your camera to get an AI-powered audio tour</div>
""", unsafe_allow_html=True)

input_tab1, input_tab2 = st.tabs(["📁  Upload Photo", "📷  Camera"])

image_bytes = None

with input_tab1:
    uploaded = st.file_uploader(
        "Drop an image here",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed",
    )
    if uploaded:
        image_bytes = uploaded.read()

with input_tab2:
    cam_shot = st.camera_input("Take a photo", label_visibility="collapsed")
    if cam_shot:
        image_bytes = cam_shot.read()

if image_bytes:
    st.markdown('<div class="preview-wrap">', unsafe_allow_html=True)
    st.image(image_bytes, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────────────────────────────────────
run_btn = st.button("𓂀  Identify & Start Audio Tour", disabled=(image_bytes is None))

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
if run_btn and image_bytes:

    # Step 1 — Classify
    step_ph = st.empty()
    step_ph.markdown("""
    <div class="progress-card">
        <div class="step-row"><div class="step-dot active"></div> Analysing image with EfficientNetB0…</div>
        <div class="step-row"><div class="step-dot pending"></div> Generating tour guide narration</div>
        <div class="step-row"><div class="step-dot pending"></div> Converting narration to audio</div>
    </div>""", unsafe_allow_html=True)

    pred_class = predict_monument(image_bytes, model)

    # Step 2 — Narration
    step_ph.markdown("""
    <div class="progress-card">
        <div class="step-row"><div class="step-dot done"></div> Monument identified ✓</div>
        <div class="step-row"><div class="step-dot active"></div> Generating tour guide narration…</div>
        <div class="step-row"><div class="step-dot pending"></div> Converting narration to audio</div>
    </div>""", unsafe_allow_html=True)

    info      = get_monument_info(pred_class, kb)
    narration = generate_narration(pred_class, info)

    # Step 3 — TTS
    step_ph.markdown("""
    <div class="progress-card">
        <div class="step-row"><div class="step-dot done"></div> Monument identified ✓</div>
        <div class="step-row"><div class="step-dot done"></div> Narration ready ✓</div>
        <div class="step-row"><div class="step-dot active"></div> Converting to audio…</div>
    </div>""", unsafe_allow_html=True)

    audio_b = text_to_speech(narration, TTS_VOICE)
    step_ph.empty()

    # ── Result hero card
    loc = info.get("location", "Egypt")
    st.markdown(f"""
    <div class="result-hero">
        <div class="result-eyebrow">𓂀 &nbsp; Monument Identified &nbsp; 𓂀</div>
        <div class="result-name">{info.get('official_name', pred_class)}</div>
        <div class="result-location">📍 &nbsp; {loc}</div>
        <div class="result-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    # ── Narration card
    st.markdown(f"""
    <div class="narration-card">
        <div class="narration-label">🎙 &nbsp; Tour Guide Narration</div>
        <div class="narration-text">{narration}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Audio card
    audio_b64 = base64.b64encode(audio_b).decode()
    st.markdown(f"""
    <div class="audio-card">
        <div class="audio-label">🔊 &nbsp; Listen to Your Tour</div>
        <audio controls autoplay>
            <source src="data:audio/mpeg;base64,{audio_b64}" type="audio/mpeg">
        </audio>
    </div>
    """, unsafe_allow_html=True)

    # Download
    st.download_button(
        label="⬇️  Download Audio Tour (MP3)",
        data=audio_b,
        file_name=f"{pred_class.replace(' ', '_')}_tour.mp3",
        mime="audio/mpeg",
    )

elif not image_bytes:
    st.info("📸  Upload a photo or use the camera above to begin your audio tour.")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">
    <div class="footer-ankh">𓋹</div>
    <div class="footer-brand">MonuVision — AI Monument Guide</div>
    <div>Powered by EfficientNetB0 · Groq LLaMA · Edge-TTS &nbsp;|&nbsp; © 2026 MonuVision</div>
</div>
""", unsafe_allow_html=True)
