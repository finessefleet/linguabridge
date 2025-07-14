import streamlit as st
from textblob import TextBlob, Word
from textblob.sentiments import NaiveBayesAnalyzer, PatternAnalyzer
from afinn import Afinn
import re
from collections import defaultdict, Counter
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize, sent_tokenize, RegexpTokenizer
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from langdetect import detect, DetectorFactory
from googletrans import Translator
import pandas as pd
import emoji
from spellchecker import SpellChecker as PySpellChecker
import random
from typing import Dict, List, Tuple, Optional

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize utilities
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Initialize spell checker with English language
spell = PySpellChecker()
DetectorFactory.seed = 0  # For consistent language detection

def detect_language(text):
    """Detect the language of the given text."""
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def translate_hindi_to_english(text):
    """Translate Hindi text to English using Google Translate."""
    try:
        translator = Translator()
        translated = translator.translate(text, src='hi', dest='en')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original text if translation fails

def preprocess_bilingual_text(text):
    """Preprocess code-mixed Hindi-English text."""
    # Remove URLs, mentions, and special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    
    # Convert emojis to text
    text = emoji.demojize(text, delimiters=(" ", " "))
    
    # Tokenize while preserving words with apostrophes
    tokenizer = RegexpTokenizer(r"\w+(?:'\w+)?")
    tokens = tokenizer.tokenize(text.lower())
    
    # Translate Hindi words to English
    processed_tokens = []
    for token in tokens:
        if len(token) > 1:  # Ignore single characters
            try:
                # Detect language of the token
                lang = detect(token)
                if lang == 'hi':  # If token is Hindi
                    translated = translate_hindi_to_english(token)
                    if translated != token:  # If translation was successful
                        processed_tokens.extend(translated.lower().split())
                    else:
                        processed_tokens.append(token)
                else:
                    # Correct English spelling
                    corrected = spell.correction(token)
                    if corrected:
                        processed_tokens.append(corrected)
                    else:
                        processed_tokens.append(token)
            except:
                processed_tokens.append(token)
    
    # Remove stopwords and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word, get_wordnet_pos('v')) 
                       for word in processed_tokens 
                       if word not in stop_words and len(word) > 2]
    
    return ' '.join(processed_tokens)

# Distress-related phrases and responses
DISTRESS_PHRASES = [
    "i am willing to die", "i don't want to live anymore", "life is meaningless",
    "i feel like giving up", "i wish it would all end", "death seems like the only escape",
    "i want it all to stop", "i can't take it anymore", "living hurts too much",
    "nothing matters anymore", "i feel completely hopeless", "the pain is unbearable",
    "there is no reason to go on", "i just want peace, even if it means death",
    "everything is dark", "i am tired of being alive", "i want to disappear forever",
    "no one would miss me if i was gone", "i'm drowning in sorrow", "i feel dead inside",
    "i'm broken beyond repair", "i have no more strength to continue",
    "the world would be better without me", "every day feels like torture",
    "i'm just waiting for it to end", "why am i still here?", "my life is a burden",
    "i can't face another day", "i don't care if i die", "i'm emotionally exhausted",
    "i've lost all hope", "i'm constantly hurting inside", "the emptiness never leaves me",
    "i want to sleep and never wake up", "i feel like ending it all",
    "i wish i could stop existing", "no one understands my pain", "i'm screaming inside",
    "i feel invisible and unwanted", "my heart feels shattered",
    "everything is falling apart", "there's nothing left for me"
]

# Supportive responses for distress signals
SUPPORT_RESPONSES = ["Negative response"]
    

# Mental health resources (India specific)
MENTAL_HEALTH_RESOURCES = ["Negative response"]

def detect_distress(text):
    """Detect if the text contains distress signals."""
    text_lower = text.lower()
    # Check for exact matches
    if any(phrase in text_lower for phrase in DISTRESS_PHRASES):
        return True
    # Check for key terms
    distress_terms = ['die', 'death', 'suicid', 'end it all', 'no reason to live', 'want to die']
    if any(term in text_lower for term in distress_terms):
        return True
    return False

def get_supportive_response():
    """Return a supportive response with resources."""
    return {
        'response': random.choice(SUPPORT_RESPONSES),
        'resources': MENTAL_HEALTH_RESOURCES,
        'is_distress': True
    }

# Emotion-related word sets
EMOTION_WORDS = {
    # Basic Emotions
    'joy': {'happy', 'joy', 'delight', 'bliss', 'cheer', 'ecstasy', 'elation', 'euphoria', 'glee', 'jubilation'},
    'sadness': {'sad', 'unhappy', 'sorrow', 'grief', 'melancholy', 'despair', 'gloom', 'misery', 'woe'},
    'anger': {'angry', 'mad', 'fury', 'rage', 'outrage', 'wrath', 'irritation', 'annoyance', 'frustration'},
    'fear': {'afraid', 'scared', 'fear', 'terror', 'dread', 'panic', 'anxiety', 'apprehension', 'nervous'},
    'disgust': {'disgust', 'revulsion', 'repulsion', 'aversion', 'loathing', 'abhorrence'},
    'surprise': {'surprise', 'amazement', 'astonishment', 'shock', 'wonder'},
    'trust': {'trust', 'faith', 'confidence', 'reliance', 'assurance'},
    'anticipation': {'anticipation', 'expectation', 'hope', 'excitement', 'eagerness'},
    
    # Complex Emotions
    'confusion': {'confuse', 'puzzle', 'baffle', 'bewilder', 'perplex'},
    'disappointment': {'disappoint', 'let down', 'dishearten', 'discourage'},
    'gratitude': {'grateful', 'thankful', 'appreciation', 'thanks'},
    'hope': {'hope', 'optimism', 'expectation', 'aspiration'},
    'anxiety': {'anxious', 'worry', 'unease', 'apprehension', 'nervousness'},
    'frustration': {'frustrate', 'exasperate', 'irritate', 'annoy'},
    'sarcasm': {'sarcasm', 'sarcastic', 'irony', 'ironic', 'sardonic'},
    'ambivalence': {'ambivalent', 'mixed feelings', 'uncertain', 'unsure'},
    'contempt': {'contempt', 'scorn', 'disdain', 'disrespect'},
    'resentment': {'resent', 'bitterness', 'indignation', 'rancor'},
    'embarrassment': {'embarrass', 'shame', 'self-conscious', 'mortification'},
    'shame': {'ashamed', 'humiliation', 'disgrace', 'dishonor'},
    'guilt': {'guilty', 'remorse', 'regret', 'contrition'},
    'pride': {'proud', 'pride', 'dignity', 'self-respect'},
    'loneliness': {'lonely', 'alone', 'isolated', 'deserted'},
    'boredom': {'bored', 'uninterested', 'tired', 'weary'},
    'curiosity': {'curious', 'inquisitive', 'interested', 'nosy'},
    'excitement': {'excited', 'thrilled', 'eager', 'enthusiastic'},
    'love': {'love', 'affection', 'fondness', 'adoration'},
    'hate': {'hate', 'loathe', 'despise', 'abhor'},
    'envy': {'envy', 'jealous', 'covet', 'resentment'}
}

# Emotion intensity modifiers
INTENSITY_MODIFIERS = {
    'slightly': 0.5, 'somewhat': 0.7, 'quite': 1.0, 'very': 1.3, 'extremely': 1.5,
    'really': 1.3, 'terribly': 1.4, 'incredibly': 1.5, 'absolutely': 1.6, 'completely': 1.6
}

# Sentiment intensifiers and negation words
INTENSIFIERS = {
    'absolutely': 1.5, 'completely': 1.5, 'extremely': 1.5, 'highly': 1.3,
    'really': 1.3, 'remarkably': 1.4, 'so': 1.2, 'too': 1.2, 'totally': 1.3,
    'utterly': 1.4, 'very': 1.3, 'exceptionally': 1.4, 'incredibly': 1.4,
    'intensely': 1.4, 'terribly': 1.4, 'awfully': 1.4, 'hugely': 1.3
}

NEGATIONS = {
    'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nor',
    'nowhere', 'hardly', 'scarcely', 'barely', 'doesnt', 'isnt', 'wasnt',
    'shouldnt', 'wouldnt', 'couldnt', 'wont', 'cant', 'dont'
}

def get_wordnet_pos(tag):
    """Map POS tag to first character for WordNet lemmatizer."""
    tag = tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text, lemmatize=True, remove_stopwords=True):
    """Enhanced text preprocessing with lemmatization and cleaning."""
    if not isinstance(text, str):
        return ""
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs, mentions, and special characters
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # URLs
    text = re.sub(r'@\w+', '', text)  # Mentions
    text = re.sub(r'#\w+', '', text)  # Hashtags
    text = re.sub(r'<.*?>', '', text)  # HTML tags
    text = re.sub(r'[^\w\s]', ' ', text)  # Punctuation
    text = re.sub(r'\d+', '', text)  # Numbers
    text = re.sub(r'\s+', ' ', text).strip()  # Extra whitespace
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Lemmatization and stopword removal
    if lemmatize or remove_stopwords:
        pos_tags = nltk.pos_tag(tokens)
        processed_tokens = []
        
        for word, tag in pos_tags:
            if remove_stopwords and word in stop_words:
                continue
                
            if lemmatize:
                word = lemmatizer.lemmatize(word, get_wordnet_pos(tag))
                
            if word:  # Ensure word is not empty after processing
                processed_tokens.append(word)
                
        tokens = processed_tokens
    
    return ' '.join(tokens)

def analyze_text_sentiment(text, analyzer_type='pattern'):
    """Analyze text sentiment using specified analyzer."""
    if analyzer_type == 'pattern':
        return TextBlob(text).sentiment
    elif analyzer_type == 'naive_bayes':
        return TextBlob(text, analyzer=NaiveBayesAnalyzer()).sentiment
    return None

def get_sentiment_metrics(polarity, confidence, analyzer_type):
    """Determine sentiment category and emoji based on polarity and confidence."""
    if analyzer_type == 'naive_bayes':
        is_positive = confidence.p_pos > confidence.p_neg
        confidence_score = max(confidence.p_pos, confidence.p_neg)
        
        if is_positive:
            if confidence_score > 0.8:
                return "Extremely Positive", "😍", confidence_score * 100
            elif confidence_score > 0.6:
                return "Very Positive", "😊", confidence_score * 100
            else:
                return "Positive", "🙂", confidence_score * 100
        else:
            if confidence_score > 0.8:
                return "Extremely Negative", "😡", confidence_score * 100
            elif confidence_score > 0.6:
                return "Very Negative", "😠", confidence_score * 100
            else:
                return "Negative", "🙁", confidence_score * 100
    else:  # pattern analyzer
        if polarity > 0.5:
            return "Very Positive", "😊", abs(polarity) * 100
        elif polarity > 0.1:
            return "Positive", "🙂", abs(polarity) * 100
        elif polarity < -0.5:
            return "Very Negative", "😠", abs(polarity) * 100
        elif polarity < -0.1:
            return "Negative", "🙁", abs(polarity) * 100
        else:
            return "Neutral", "😐", 0.0

def analyze_bilingual_text(text):
    """Analyze sentiment of code-mixed Hindi-English text."""
    try:
        if not text or not isinstance(text, str):
            return {"sentiment": "Error", "error": "Invalid input text", "analysis_method": "Bilingual Analysis"}
        
        # Preprocess and translate code-mixed text
        processed_text = preprocess_bilingual_text(text)
        
        # Get language distribution
        tokens = word_tokenize(processed_text)
        if not tokens:
            return {"sentiment": "Neutral", "confidence": 0, "analysis_method": "Bilingual Analysis"}
        
        # Analyze sentiment using TextBlob on processed text
        return analyze_textblob(processed_text, is_bilingual=True)
        
    except Exception as e:
        return {
            "sentiment": "Error",
            "error": str(e),
            "analysis_method": "Bilingual Analysis"
        }

def detect_emotions(text: str) -> Dict[str, float]:
    """Detect emotions in the text and return a dictionary of emotions with their intensities."""
    # Initialize emotion scores
    emotion_scores = {emotion: 0.0 for emotion in EMOTION_WORDS.keys()}
    
    # Tokenize and preprocess the text
    tokens = word_tokenize(text.lower())
    
    # Track intensity modifiers
    current_intensity = 1.0
    
    for i, token in enumerate(tokens):
        # Check for intensity modifiers
        if token in INTENSITY_MODIFIERS:
            current_intensity = INTENSITY_MODIFIERS[token]
            continue
            
        # Check for emotion words
        for emotion, words in EMOTION_WORDS.items():
            if any(word in token for word in words):
                emotion_scores[emotion] += current_intensity
                current_intensity = 1.0  # Reset intensity after use
    
    # Normalize scores
    max_score = max(emotion_scores.values()) if any(emotion_scores.values()) else 1
    if max_score > 0:
        emotion_scores = {k: v/max_score for k, v in emotion_scores.items()}
    
    # Get top 3 emotions
    top_emotions = dict(sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)[:3])
    return {k: v for k, v in top_emotions.items() if v > 0.1}  # Only return emotions with score > 0.1

def analyze_textblob(text, is_bilingual=False):
    """Advanced TextBlob sentiment analysis with enhanced features."""
    if not text.strip():
        return {
            'polarity': 0, 
            'subjectivity': 0, 
            'sentiment': 'neutral', 
            'confidence': 0, 
            'analysis': 'No text provided',
            'emotions': {},
            'dominant_emotion': None
        }
    
    # Check for distress signals first
    if detect_distress(text):
        support = get_supportive_response()
        return {
            'polarity': -1.0,
            'subjectivity': 0.8,
            'sentiment': 'very negative',
            'confidence': 0.95,
            'analysis': support['response'],
            'resources': support['resources'],
            'is_distress': True,
            'emotions': {'sadness': 1.0, 'fear': 0.8, 'hopelessness': 1.0},
            'dominant_emotion': 'sadness'
        }
    
    # Preprocess text
    if is_bilingual:
        cleaned_text = text  # Already preprocessed in analyze_bilingual_text
    else:
        cleaned_text = preprocess_text(text, lemmatize=True, remove_stopwords=True)
        
    # Initialize analyzers
    pattern_analyzer = PatternAnalyzer()
    naive_bayes_analyzer = NaiveBayesAnalyzer()
    
    # Get sentiment from both analyzers
    pattern_sentiment = pattern_analyzer.analyze(cleaned_text)
    nb_sentiment = naive_bayes_analyzer.analyze(cleaned_text)
    
    # Detect emotions
    emotions = detect_emotions(text)
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
    
    # Combine results for better accuracy
    combined_polarity = (pattern_sentiment.polarity + 
                       (0.5 if nb_sentiment.classification == 'pos' else -0.5) * 
                       max(nb_sentiment.p_pos, nb_sentiment.p_neg)) / 2
    
    # Get sentiment metrics
    sentiment, emoji, confidence = get_sentiment_metrics(
        combined_polarity, 
        nb_sentiment,
        'naive_bayes' if max(nb_sentiment.p_pos, nb_sentiment.p_neg) > 0.6 else 'pattern'
    )
    
    # Advanced text analysis
    blob = TextBlob(cleaned_text)
    words = word_tokenize(cleaned_text)
    sentences = sent_tokenize(text)  # Use original text for sentence count
    
    # Enhanced POS tagging and key phrase extraction
    pos_tags = nltk.pos_tag(words)
    noun_phrases = []
    adj_phrases = []
    
    # Extract noun and adjective phrases
    for word, tag in pos_tags:
        if tag.startswith('NN'):
            noun_phrases.append(word)
        elif tag.startswith('JJ'):
            adj_phrases.append(word)
    
    # Calculate reading level (Flesch-Kincaid Grade Level)
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = sum([len(list(filter(lambda c: c in 'aeiouy', word))) for word in words])
    
    if word_count > 0 and sentence_count > 0:
        avg_sentence_length = word_count / sentence_count
        avg_syllables_per_word = syllable_count / word_count
        
        # Flesch-Kincaid Grade Level
        fk_grade = 0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59
        reading_level = max(1, min(12, round(fk_grade)))
    else:
        avg_sentence_length = 0
        reading_level = 0
    
    # Sentiment distribution analysis
    sentence_sentiments = []
    for sent in sentences:
        sent_blob = TextBlob(sent)
        sentence_sentiments.append({
            'text': sent,
            'polarity': sent_blob.sentiment.polarity,
            'subjectivity': sent_blob.sentiment.subjectivity
        })
    
    # Calculate sentiment consistency
    if len(sentence_sentiments) > 1:
        polarities = [s['polarity'] for s in sentence_sentiments]
        sentiment_std = np.std(polarities)
        sentiment_consistency = max(0, 1 - (sentiment_std / 2))  # 0-1 scale, higher is more consistent
    else:
        sentiment_consistency = 1.0
    
    # Prepare response
    result = {
        "sentiment": sentiment,
        "sentiment_emoji": emoji,
        "polarity": round(combined_polarity, 4),
        "subjectivity": round(pattern_sentiment.subjectivity, 4),
        "confidence": round(confidence, 2),
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": round(avg_sentence_length, 2),
        "reading_level": reading_level,
        "sentiment_consistency": round(sentiment_consistency, 2),
        "key_nouns": list(set(noun_phrases))[:8],
        "key_adjectives": list(set(adj_phrases))[:5],
        "sentence_analysis": sentence_sentiments,
        "analysis_method": "Enhanced TextBlob with Hybrid Analysis"
    }
    
    # Add TF-IDF based key terms if text is long enough
    if word_count >= 10:
        try:
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top TF-IDF terms
            tfidf_scores = tfidf_matrix.toarray().flatten()
            top_indices = tfidf_scores.argsort()[-5:][::-1]
            top_terms = [(feature_names[i], round(tfidf_scores[i], 4)) 
                        for i in top_indices if tfidf_scores[i] > 0]
            
            result["key_terms"] = top_terms
        except Exception as e:
            result["key_terms"] = []
    
    return result

def contains_distress_phrase(text):
    """Check if text contains any distress phrases (case-insensitive)."""
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in DISTRESS_PHRASES)

def analyze_afinn(text, is_bilingual=False):
    """Analyze text sentiment using AFINN lexicon with enhanced distress detection."""
    if not text.strip():
        return {'score': 0, 'sentiment': 'neutral', 'analysis': 'No text provided'}
    
    # Check for distress signals first - both exact phrases and general distress terms
    if detect_distress(text) or contains_distress_phrase(text):
        support = get_supportive_response()
        return {
            'score': -5,  # Most negative score possible
            'polarity': -1.0,  # Maximum negative polarity
            'sentiment': 'very negative',
            'sentiment_emoji': '😢',
            'analysis': support['response'],
            'resources': support['resources'],
            'is_distress': True,
            'analysis_method': 'AFINN Lexicon (Distress Detection)'
        }
    
    if is_bilingual:
        processed_text = text  # Already preprocessed
    else:
        processed_text = preprocess_text(text, lemmatize=True, remove_stopwords=True)
        
    af = Afinn()
    score = af.score(processed_text)
    
    # Enhanced sentiment classification for AFINN with more granular scoring
    if score >= 3.0:
        sentiment = "Very Positive"
        emoji = "😊"
    elif score >= 1.0:
        sentiment = "Positive"
        emoji = "🙂"
    elif score <= -3.0:
        sentiment = "Very Negative"
        emoji = "😢"
    elif score <= -1.0:
        sentiment = "Negative"
        emoji = "🙁"
    else:
        # For scores between -1.0 and 1.0
        if score > 0:
            sentiment = "Slightly Positive"
            emoji = "🙂"
        elif score < 0:
            sentiment = "Slightly Negative"
            emoji = "🙁"
        else:
            sentiment = "Neutral"
            emoji = "😐"
    
    # Detect emotions
    emotions = detect_emotions(text)
    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else None
    
    return {
        "sentiment": sentiment,
        "sentiment_emoji": emoji,
        "score": round(score, 2),
        "polarity": max(min(score / 5.0, 1.0), -1.0),  # Normalize to -1 to 1 range
        "analysis_method": "AFINN Lexicon (Bilingual)" if is_bilingual else "AFINN Lexicon",
        "emotions": emotions,
        "dominant_emotion": dominant_emotion,
        "is_distress": False
    }

def display_emotion_chart(emotions: Dict[str, float]):
    """Display a bar chart of emotion intensities."""
    if not emotions:
        return
        
    import plotly.express as px
    import pandas as pd
    
    # Prepare data for visualization
    df = pd.DataFrame({
        'Emotion': list(emotions.keys()),
        'Intensity': list(emotions.values())
    })
    
    # Create and display the bar chart
    fig = px.bar(df, x='Emotion', y='Intensity', 
                 title='Emotion Analysis',
                 color='Intensity',
                 color_continuous_scale='Viridis')
    fig.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

def display_sentiment_result(analysis, title):
    """Display sentiment analysis results in a consistent format."""
    st.subheader(title)
    
    # Check for distress signals first
    if analysis.get('is_distress', False):
        st.error(analysis['analysis'])
        st.warning("Negative response")
        return
        
    # Display emotions if available
    if 'emotions' in analysis and analysis['emotions']:
        st.markdown("#### Detected Emotions")
        if analysis.get('dominant_emotion'):
            st.markdown(f"**Dominant Emotion:** {analysis['dominant_emotion'].title()}")
        display_emotion_chart(analysis['emotions'])
        st.markdown("---")
    
    # Determine color and emoji based on sentiment
    if 'sentiment' in analysis:
        if analysis['sentiment'] == 'very positive':
            color = 'green'
            emoji = "😊"
        elif analysis['sentiment'] == 'positive':
            color = 'lightgreen'
            emoji = '🙂'
        elif analysis['sentiment'] == 'very negative':
            color = 'red'
            emoji = '😠'
        elif analysis['sentiment'] == 'negative':
            color = 'pink'
            emoji = '🙁'
        else:
            color = 'gray'
            emoji = '😐'
        
        # Display sentiment with emoji
        col1, col2 = st.columns([1, 4])
        with col1:
            st.markdown(f"### {emoji} {analysis['sentiment']}")
            st.markdown(f"### {analysis.get('sentiment_emoji', '')} {analysis.get('sentiment', 'Unknown')}")
        with col2:
            if 'confidence' in analysis:
                st.metric("Confidence", f"{analysis['confidence']:.1f}%")
        
        # Display detailed metrics
        if 'polarity' in analysis:
            st.progress((analysis['polarity'] + 1) / 2)
            st.caption(f"Polarity: {analysis['polarity']:.2f} (Range: -1 to 1)")
        
        if 'subjectivity' in analysis:
            st.progress(analysis['subjectivity'])
            st.caption(f"Subjectivity: {analysis['subjectivity']:.2f} (0 = Objective, 1 = Subjective)")
        
        # Display key phrases if available
        if 'key_phrases' in analysis and analysis['key_phrases']:
            st.write("🔑 Key Phrases:", ", ".join([f"`{p}`" for p in analysis['key_phrases']]))
        
        # Display word count metrics
        if 'word_count' in analysis:
            cols = st.columns(3)
            with cols[0]:
                st.metric("Words", analysis['word_count'])
            if 'sentence_count' in analysis:
                with cols[1]:
                    st.metric("Sentences", analysis['sentence_count'])
            if 'avg_sentence_length' in analysis:
                with cols[2]:
                    st.metric("Avg. Words/Sentence", analysis['avg_sentence_length'])
        
        # Display analysis method
        if 'analysis_method' in analysis:
            st.caption(f"*Analysis method: {analysis['analysis_method']}*")

def sentiment_analyzer_interface():
    # Page header with description
    st.title("📊 Sentiment Analysis Tool")
    st.markdown("""
    Analyze the sentiment and emotions in your text. Supports both English and Hindi text analysis.
    """)
    
    # Input section with clear visual separation
    with st.container():
        st.subheader("📝 Input Text")
        text = st.text_area(
            "Enter your text below:", 
            height=150,
            placeholder="Type or paste your text here...",
            help="Enter text in English or Hindi for sentiment analysis"
        )
    
    # Analysis options in a clean layout
    with st.container():
        st.subheader("⚙️ Analysis Options")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            analysis_type = st.radio(
                "Analysis Type",
                ["Standard (English)", "Bilingual (Hindi-English)"],
                index=0,
                help="Choose standard analysis for English or bilingual for Hindi-English text"
            )
        
        with col2:
            st.write("")
            analyze_btn = st.button(
                "🔍 Analyze Sentiment", 
                use_container_width=True,
                type="primary"
            )
    
    if analyze_btn:
        if not text.strip():
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔍 Analyzing sentiment... This may take a moment..."):
                try:
                    st.divider()
                    st.subheader("📊 Analysis Results")
                    
                    if analysis_type == "Bilingual (Hindi-English)":
                        # Bilingual analysis
                        with st.container():
                            # Get bilingual analysis
                            bilingual_result = analyze_bilingual_text(text)
                            
                            # Language detection
                            lang = detect_language(text)
                            lang_col, _ = st.columns([2, 3])
                            with lang_col:
                                st.info(f"🌐 Detected Language: **{'Hindi' if lang == 'hi' else 'English'}**")
                            
                            # Display results in a card
                            with st.container():
                                display_sentiment_result(bilingual_result, "Bilingual Analysis")
                            
                            # Show translation if needed
                            if lang == 'hi':
                                with st.expander("🌐 View Translation", expanded=False):
                                    translated = translate_hindi_to_english(text)
                                    st.markdown("**Translated Text:**")
                                    st.write(translated)
                                    st.caption("Note: This is an automated translation for reference")
                    else:
                        # Standard English analysis
                        # Analyze using TextBlob and AFINN
                        tb_analysis = analyze_textblob(text)
                        afinn_analysis = analyze_afinn(text)
                        
                        # Display results in tabs for better organization
                        tab1, tab2 = st.tabs(["TextBlob Analysis", "AFINN Analysis"])
                        
                        with tab1:
                            display_sentiment_result(tb_analysis, "TextBlob Analysis")
                        
                        with tab2:
                            display_sentiment_result(afinn_analysis, "AFINN Analysis")
                        
                        # Add comparison section
                        st.divider()
                        st.subheader("📈 Method Comparison")
                        
                        # Prepare comparison data
                        comparison_data = []
                        for name, result in [("TextBlob", tb_analysis), ("AFINN", afinn_analysis)]:
                            if 'error' not in result:
                                comp = {
                                    "Method": name,
                                    "Sentiment": f"{result.get('sentiment_emoji', '')} {result.get('sentiment', 'Unknown').title()}",
                                    "Confidence": f"{result.get('confidence', 0):.1%}" if 'confidence' in result else "N/A"
                                }
                                if 'polarity' in result:
                                    comp["Polarity"] = f"{result['polarity']:.2f}"
                                if 'score' in result:
                                    comp["Score"] = f"{result['score']:.2f}"
                                comparison_data.append(comp)
                        
                        # Display comparison table if we have data
                        if comparison_data:
                            # Visual comparison
                            try:
                                if all('polarity' in r for r in [tb_analysis, afinn_analysis] if 'error' not in r):
                                    import plotly.express as px
                                    fig = px.bar(
                                        x=["TextBlob", "AFINN"],
                                        y=[r.get('polarity', 0) for r in [tb_analysis, afinn_analysis] if 'error' not in r],
                                        labels={'x': 'Method', 'y': 'Polarity Score'},
                                        title='Sentiment Polarity Comparison',
                                        text_auto=True,
                                        color_discrete_sequence=['#4B9CD3']
                                    )
                                    fig.update_layout(showlegend=False)
                                    st.plotly_chart(fig, use_container_width=True)
                            except Exception as e:
                                st.warning(f"Could not generate comparison chart: {str(e)}")
                            
                            # Display comparison table
                            st.markdown("### Method Comparison Table")
                            st.table(comparison_data)
                            
                            # Add explanation
                            with st.expander("ℹ️ About the Analysis Methods", expanded=False):
                                st.markdown("""
                                - **TextBlob**: Uses pattern-based sentiment analysis with a focus on contextual understanding
                                - **AFINN**: Uses a word-list based approach with predefined sentiment scores for words
                                - **Polarity**: Ranges from -1 (negative) to 1 (positive)
                                - **Confidence**: Indicates the certainty of the sentiment classification
                                """)
                        else:
                            st.info("No comparable data available for the selected analysis type.")
                            
                except Exception as e:
                    st.error("❌ An error occurred during analysis. Please try again.")
                    with st.expander("Show error details", expanded=False):
                        st.exception(e)
                    
                    # Add a button to try again
                    if st.button("🔄 Try Again", type="secondary"):
                        st.rerun()
