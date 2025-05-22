# LinguaBridge

## Overview
This is a **Streamlit-based chatbot application** that performs:
- **Translation**: Converts text into Indian languages using `deep_translator`.
- **Sentiment Analysis**: Analyzes text sentiment using `TextBlob`.
- **Chatbot Interface**: Engages the user step-by-step for a natural conversation experience.

---

## Features
### 🎯 Chatbot-Based Translator
- Users enter a sentence.
- The chatbot asks for the target language.
- The system provides a translated response in the chosen language.

### 📊 Sentiment Analysis
- Users input text for sentiment evaluation.
- Detects **positive, negative, or neutral** sentiment.

### 🔤 Supports Multiple Indian Languages
- Assamese, Bengali, Gujarati, Hindi, Kannada, Konkani, Maithili, Malayalam, Marathi, Nepali, Odia, Punjabi, Sanskrit, Sindhi, Tamil, Telugu, Urdu.

---

## Installation
### 🛠 Setup Virtual Environment
```bash
python -m venv env
source env/bin/activate  # For macOS/Linux
env\Scripts\activate    # For Windows
```

### 📦 Install Dependencies
```bash
pip install -r requirement.txt
```

---

## Usage
### 🚀 Run the Application
```bash
streamlit run stream.py
```

### 🏗 Chatbot Flow
1. **User:** Enters a sentence.
2. **System:** Asks for the target language.
3. **User:** Selects the language.
4. **System:** Provides the translated output.

---

## Dependencies
- `streamlit`
- `deep_translator`
- `textblob`

Install them using:
```bash
pip install streamlit deep_translator textblob
```

---

## Future Enhancements
- **Speech-to-Text Support** 🎙
- **Multilingual Sentiment Analysis** 📖
- **Chatbot Personality Customization** 🤖

---

## Author
**Saumik Chakraborty** 🚀

# linguabridge
# linguabridge
