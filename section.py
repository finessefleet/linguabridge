# sections/abhisek_section.py

import streamlit as st
import translation_module
def show():
    st.title("Code-Mixed Translator")

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
                output = translation_module.translate_hinglish_to_hindi(input_text)
            elif lang_option == "Bengalish to Bengali":
                output = translation_module.transliterate_bengalish_to_bengali_script(input_text)

            elif lang_option == "Bengali to English":
                output = translation_module.translate_bengali_to_english(input_text)
            elif lang_option == "Tanglish to Tamil":
                output = translation_module.translate_tanglish_to_tamil(input_text)
            elif lang_option == "Telugulish to Telugu":
                output = translation_module.translate_telugulish_to_telugu(input_text)

            elif lang_option == "Odia (Code-mix) to Odia":
                output = translation_module.translate_odia_to_odia(input_text)

            elif lang_option == "Urdlish (Code-mix) to Urdu":
                output = translation_module.translate_urdlish_to_urdu(input_text)

            st.success("Translated Output:")
            st.write(output)