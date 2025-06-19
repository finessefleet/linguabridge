# sections/abhisek_section.py

import streamlit as st
from utils import translation_models

def show():
    st.title("Abhisek's Code-Mixed Translator")

    lang_option = st.selectbox("Choose Translation Type", [
        "Hinglish to Hindi",
        "Bengalish to Bengali",
        "Bengali to English",
        "Tanglish to Tamil",
        "Telugulish to Telugu",
        "Odia (Code-mix) to Odia",
        "Urdlish (Code-mix) to Urdu"
    ])

    input_text = st.text_area("Enter code-mixed sentence:")

    if st.button("Translate"):
        if not input_text.strip():
            st.warning("Please enter some text.")
            return

        with st.spinner("Translating..."):
            if lang_option == "Hinglish to Hindi":
                output = translation_models.translate_hinglish_to_hindi(input_text)
            elif lang_option == "Bengalish to Bengali":
                output = translation_models.transliterate_bengalish_to_bengali_script(input_text)

            elif lang_option == "Bengali to English":
                output = translation_models.translate_bengali_to_english(input_text)
            elif lang_option == "Tanglish to Tamil":
                output = translation_models.translate_tanglish_to_tamil(input_text)
            elif lang_option == "Telugulish to Telugu":
                output = translation_models.translate_telugulish_to_telugu(input_text)

            elif lang_option == "Odia (Code-mix) to Odia":
                output = translation_models.translate_odia_to_odia(input_text)

            elif lang_option == "Urdlish (Code-mix) to Urdu":
                output = translation_models.translate_urdlish_to_urdu(input_text)

            st.success("Translated Output:")
            st.write(output)
