import re
import nltk
import streamlit as st
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn
from nrclex import NRCLex

# Download resources
nltk.download('punkt')
nltk.download('vader_lexicon')

# Initialize tools
vader = SentimentIntensityAnalyzer()
afinn = Afinn()

# === Core Functions ===

def tokenize_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return word_tokenize(text)

def analyze_textblob(text):
    return TextBlob(text).sentiment.polarity

def analyze_vader(text):
    return vader.polarity_scores(text)['compound']

def analyze_afinn(text):
    score = afinn.score(text)
    return max(min(score / 10, 1.0), -1.0)

def average_sentiment(text):
    tb = analyze_textblob(text)
    vd = analyze_vader(text)
    af = analyze_afinn(text)
    return (tb + vd + af) / 3

def get_sentiment_label(score):
    if score > 0.2:
        return "Positive", "😊"
    elif score < -0.2:
        return "Negative", "😟"
    else:
        return "Neutral", "😐"

def detect_emotions(text):
    emotion_obj = NRCLex(text)
    emotions = emotion_obj.raw_emotion_scores
    total = sum(emotions.values())
    if total > 0:
        normalized = {k: round(v / total, 2) for k, v in emotions.items()}
        top3 = dict(sorted(normalized.items(), key=lambda x: x[1], reverse=True)[:3])
        return top3
    return {}

# === Streamlit UI ===

def sentiment_analyzer_interface():
    st.set_page_config(page_title="🧠 Sentiment + Emotion Analyzer", layout="wide")
    st.title("🧠 Sentiment & Emotion Analyzer (Offline)")
    st.markdown("""
Analyze your text using **TextBlob**, **VADER**, and **AFINN** for sentiment, and **NRCLex** for emotion detection.
*No internet or cloud models needed.*
""")

    text = st.text_area("📝 Enter your text here:", height=150)

    if st.button("Analyze"):
        if not text.strip():
            st.warning("⚠️ Please enter some text.")
            return

        with st.spinner("Analyzing sentiment and emotions..."):
            tokens = tokenize_text(text)
            tb_score = analyze_textblob(text)
            vader_score = analyze_vader(text)
            afinn_score = analyze_afinn(text)
            avg_score = average_sentiment(text)
            sentiment, emoji = get_sentiment_label(avg_score)
            emotions = detect_emotions(text)



            # Individual Scores
            st.subheader("📊 Sentiment Scores")
            col1, col2, col3 = st.columns(3)
            col1.metric("TextBlob", f"{tb_score:.3f}")
            col2.metric("VADER", f"{vader_score:.3f}")
            col3.metric("AFINN", f"{afinn_score:.3f}")

            st.markdown("---")
            st.header("📈 Final Sentiment")
            st.metric("Average Score", f"{avg_score:.3f}")
            st.success(f"**Sentiment:** {sentiment} {emoji}")

            # Emotion Chart
            if emotions:
                st.markdown("### ❤️ Emotion Detection (NRCLex)")
                st.bar_chart(emotions)
            else:
                st.info("No strong emotions detected.")

# === Run App ===
if __name__ == "__main__":
    sentiment_analyzer_interface()
