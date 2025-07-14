# File: translator.py
import streamlit as st
import requests
from googletrans import Translator
from rl_feedback import get_best_model
import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# 20+ Indian languages with their ISO codes for googletrans
INDIAN_LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "Nepali": "ne",
    "Sindhi": "sd",
    "Konkani": "kok",
    "Dogri": "doi",
    "Maithili": "mai",
    "Bhojpuri": "bho",
    "Sanskrit": "sa",
    "Manipuri": "mni",
    "Kashmiri": "ks",
    "Santhali": "sat",
    "Rajasthani": "raj",
}

last_input_text = ""
last_src_code = ""
last_dest_code = ""

def translator_interface():
    global last_input_text, last_src_code, last_dest_code

    translator = Translator()

    st.title("🌐 Indian Languages Translator")

    input_text = st.text_area("Enter text to translate:")

    source_language = st.selectbox("From Language:", list(INDIAN_LANGUAGES.keys()), index=list(INDIAN_LANGUAGES.keys()).index("English"))
    target_language = st.selectbox("To Language:", list(INDIAN_LANGUAGES.keys()), index=list(INDIAN_LANGUAGES.keys()).index("Hindi"))

    src_code = INDIAN_LANGUAGES[source_language]
    dest_code = INDIAN_LANGUAGES[target_language]

    if st.button("Translate"):
        model = get_best_model(auto_select=True)

        if model == "llama":
            try:
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {GROQ_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "llama3-70b-8192",
                        "messages": [
                            {"role": "system", "content": "You are a helpful assistant that translates languages."},
                            {"role": "user", "content": f"Translate from {source_language} to {target_language}: {input_text}"}
                        ],
                        "temperature": 0.7
                    }
                )
                response.raise_for_status()
                data = response.json()
                translation = data["choices"][0]["message"]["content"].strip()
            except Exception as e:
                st.error("Groq API Error: " + str(e))
                try:
                    st.json(response.json())
                except:
                    pass
                translation = "Error: Unable to translate with Groq API"
        else:
            try:
                translation = translator.translate(input_text, src=src_code, dest=dest_code).text
            except Exception as e:
                st.error("Google Translate Error: " + str(e))
                translation = "Error: Unable to translate with Google API"

        last_input_text = input_text
        last_src_code = src_code
        last_dest_code = dest_code

        st.success(f"Translation: {translation}")

   