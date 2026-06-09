# 🏛️ Egyptian Artifact Tour Guide — Streamlit App

## Files needed in the same folder as `app.py`
```
egypt_tour_app/
├── app.py
├── best_efficientnet.keras          ← your trained model
├── monuments_knowledge_base.json    ← knowledge base
├── requirements.txt
└── README.md
```

## Installation
```bash
pip install -r requirements.txt
```

## Running
```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Usage
1. Enter your **Groq API key** in the left sidebar (free at console.groq.com)
2. Choose a **narrator voice**
3. **Upload** an image or **take a photo** with your webcam/phone
4. Click **"Identify & Narrate"**
5. The app will:
   - Classify the monument with EfficientNetB0
   - Generate ~100-word narration via Groq (Llama 3.1 8B)
   - Convert narration to MP3 via Edge-TTS
   - Auto-play the audio + offer a download button
