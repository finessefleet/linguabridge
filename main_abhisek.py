# main_abhisek.py

import streamlit as st
from sections import abhisek_section

st.set_page_config(page_title="Abhisek's Translator", layout="wide")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", ["Abhisek's Translator"])

    if page == "Abhisek's Translator":
        abhisek_section.show()

if __name__ == "__main__":
    main()
