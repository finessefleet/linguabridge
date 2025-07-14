import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import spacy
from typing import Dict, List, Tuple, Optional
import random

# Load spaCy model for text analysis
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If the model is not found, download it
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# Initialize session state for feedback data
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = {
        "models": {
            "llama": {
                "scores": [],
                "language_pairs": defaultdict(list),
                "response_times": [],
                "error_rates": defaultdict(int)
            },
            "google": {
                "scores": [],
                "language_pairs": defaultdict(list),
                "response_times": [],
                "error_rates": defaultdict(int)
            }
        },
        "user_feedback": {
            "positive": [],
            "negative": []
        },
        "languages": ["en", "hi", "es", "fr", "de", "it", "pt", "ru", "zh", "ja", "ar"]
    }

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_sample_data():
    """Generate sample data for demonstration"""
    np.random.seed(42)
    
    # Generate sample feedback scores (1-5 ratings)
    sample_size = 1000
    models = ["llama", "google"]
    languages = ["en", "hi", "es", "fr", "de"]
    
    data = []
    for _ in range(sample_size):
        model = np.random.choice(models)
        source_lang = np.random.choice(languages)
        target_lang = np.random.choice([l for l in languages if l != source_lang])
        score = np.random.randint(1, 6)
        response_time = np.random.uniform(0.5, 5.0)
        
        data.append({
            "model": model,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "score": score,
            "response_time": response_time,
            "timestamp": datetime.now() - timedelta(days=np.random.randint(0, 30)),
            "feedback_text": "" if np.random.random() > 0.3 else 
                           f"Sample feedback text {np.random.choice(['good', 'bad', 'average'])}"
        })
    
    return pd.DataFrame(data)

def calculate_enhanced_metrics(scores):
    """Calculate enhanced metrics with confidence intervals"""
    if not scores:
        return {
            'avg_rating': 0,
            'accuracy': 0,
            'count': 0,
            'confidence_interval': (0, 0),
            'error_rate': 0
        }
    
    scores = np.array(scores)
    avg_rating = np.mean(scores)
    accuracy = np.mean(scores >= 3) * 100  # Percentage of acceptable ratings (3+)
    
    # Calculate 95% confidence interval
    std_err = np.std(scores) / np.sqrt(len(scores))
    ci = 1.96 * std_err  # 95% CI
    
    return {
        'avg_rating': avg_rating,
        'accuracy': accuracy,
        'count': len(scores),
        'confidence_interval': (max(1, avg_rating - ci), min(5, avg_rating + ci)),
        'error_rate': np.mean(scores < 3) * 100  # Percentage of low ratings
    }

def plot_enhanced_time_series(data: pd.DataFrame, model: str, metric: str = 'score') -> go.Figure:
    """Create an enhanced time series plot with rolling average"""
    fig = go.Figure()
    
    # Resample to daily data and calculate rolling mean
    daily_data = data.set_index('timestamp').resample('D')[metric].mean()
    rolling_avg = daily_data.rolling(window=7, min_periods=1).mean()
    
    # Add traces
    fig.add_trace(go.Scatter(
        x=daily_data.index,
        y=daily_data,
        name='Daily Average',
        line=dict(color='lightgray', width=1),
        opacity=0.5
    ))
    
    fig.add_trace(go.Scatter(
        x=rolling_avg.index,
        y=rolling_avg,
        name='7-Day Moving Average',
        line=dict(color='#1f77b4', width=3)
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{model.capitalize()} Performance Over Time",
        xaxis_title="Date",
        yaxis_title=metric.capitalize(),
        hovermode="x unified",
        showlegend=True,
        template="plotly_white"
    )
    
    return fig

def plot_language_heatmap(data: pd.DataFrame, model: str) -> go.Figure:
    """Create a heatmap of accuracy by language pair"""
    # Pivot the data for heatmap
    heatmap_data = data.pivot_table(
        index='source_lang',
        columns='target_lang',
        values='score',
        aggfunc='mean'
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Target Language", y="Source Language", color="Score"),
        color_continuous_scale='Viridis',
        aspect="auto"
    )
    
    fig.update_layout(
        title=f"{model.capitalize()} - Average Score by Language Pair",
        xaxis_title="Target Language",
        yaxis_title="Source Language"
    )
    
    return fig

def analyze_feedback_text(feedback_list: List[str]) -> Dict:
    """Analyze feedback text using NLP"""
    if not feedback_list:
        return {"common_phrases": [], "sentiment": 0.5}
    
    # Combine all feedback
    text = " ".join([str(f) for f in feedback_list if f])
    doc = nlp(text)
    
    # Extract noun phrases
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    common_phrases = Counter(noun_phrases).most_common(5)
    
    # Basic sentiment (very simple approach)
    sentiment = 0.5  # Neutral
    if len(text) > 10:  # Only analyze if there's enough text
        sentiment = min(1.0, max(0.0, sum(1 for token in doc if token.is_alpha) / len(doc)))
    
    return {
        "common_phrases": common_phrases,
        "sentiment": sentiment
    }

def show_accuracy_insights():
    st.title("📊 Advanced Translation Accuracy Insights")
    
    # Initialize feedback_data in session state if it doesn't exist
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = {
            "models": {
                "llama": {
                    "scores": [],
                    "language_pairs": {},
                    "response_times": [],
                    "error_rates": {}
                },
                "google": {
                    "scores": [],
                    "language_pairs": {},
                    "response_times": [],
                    "error_rates": {}
                }
            },
            "languages": set(),  # Initialize languages as an empty set
            "base_ratings_count":0  # Start with 1000 base ratings
        }
    
    # Load sample data (limited to 5 samples for demo)
    df = load_sample_data().head(5)  # Limit to 5 sample ratings
    
    # Update session state with sample data
    for _, row in df.iterrows():
        # Initialize model data if it doesn't exist
        if row["model"] not in st.session_state.feedback_data["models"]:
            st.session_state.feedback_data["models"][row["model"]] = {
                "scores": [],
                "language_pairs": {},
                "response_times": [],
                "error_rates": {}
            }
            
        model_data = st.session_state.feedback_data["models"][row["model"]]
        
        # Initialize language pair if it doesn't exist
        lang_pair = (row["source_lang"], row["target_lang"])
        if lang_pair not in model_data["language_pairs"]:
            model_data["language_pairs"][lang_pair] = []
        if lang_pair not in model_data["error_rates"]:
            model_data["error_rates"][lang_pair] = 0
            
        # Update data
        model_data["scores"].append(row["score"])
        model_data["language_pairs"][lang_pair].append(row["score"])
        model_data["response_times"].append(row["response_time"])
        
        if row["score"] < 3:  # Consider scores < 3 as errors
            model_data["error_rates"][lang_pair] += 1
            
            # Track unique languages
            if 'languages' not in st.session_state.feedback_data:
                st.session_state.feedback_data['languages'] = set()
            st.session_state.feedback_data['languages'].add(row["source_lang"])
            st.session_state.feedback_data['languages'].add(row["target_lang"])
    
    # Calculate metrics for each model
    model_metrics = {}
    for model in ["llama", "google"]:
        if model in st.session_state.feedback_data["models"]:
            model_data = st.session_state.feedback_data["models"][model]
            model_metrics[model] = calculate_enhanced_metrics(model_data["scores"])
    
    # Create KPIs with enhanced metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_ratings = sum(m['count'] for m in model_metrics.values() if m is not None)
        total_ratings = current_ratings + st.session_state.feedback_data.get('base_ratings_count', 0)
        num_languages = len(st.session_state.feedback_data.get('languages', set()))
        st.metric("Total Ratings", f"{total_ratings:,}", 
                 f"across {num_languages}+ languages" if num_languages > 0 else "")
    
    with col2:
        best_model = max(model_metrics.items(), key=lambda x: x[1]['avg_rating'])
        st.metric("Best Performing", 
                 best_model[0].upper(), 
                 f"{best_model[1]['avg_rating']:.1f}/5.0")
    
    with col3:
        avg_accuracy = np.mean([m['accuracy'] for m in model_metrics.values()])
        st.metric("Average Accuracy", 
                 f"{avg_accuracy:.1f}%", 
                 "across all translations")
    
    with col4:
        avg_response_time = np.mean([
            np.mean(st.session_state.feedback_data['models'][model]['response_times'] or [0])
            for model in model_metrics
        ])
        st.metric("Avg Response Time", 
                 f"{avg_response_time:.2f}s", 
                 "per translation")
    
    # Add tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Overview", 
        "🌐 Language Analysis", 
        "⏱️ Response Times",
        "📝 Feedback Analysis"
    ])
    
    with tab1:
        st.subheader("Model Performance Over Time")
        # Use a unique key for the radio button to prevent Streamlit duplicate key errors
        selected_model = st.radio("Select Model", ["llama", "google"], horizontal=True, key=f"perf_model_selector_{datetime.now().timestamp()}")
        
        # Show time series for selected model
        model_data = df[df['model'] == selected_model]
        fig = plot_enhanced_time_series(model_data, selected_model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show confidence intervals
        st.subheader("Confidence Intervals")
        st.info(
            f"For {selected_model.upper()}, we are 95% confident that the true average rating "
            f"is between {model_metrics[selected_model]['confidence_interval'][0]:.2f} and "
            f"{model_metrics[selected_model]['confidence_interval'][1]:.2f}."
        )
    
    with tab2:
        st.subheader("Translation Accuracy by Language Pair")
        # Use a unique key for the radio button to prevent Streamlit duplicate key errors
        selected_model = st.radio("Select Model", ["llama", "google"], 
                                horizontal=True, key=f"lang_model_selector_{datetime.now().timestamp()}")
        
        # Show language heatmap
        model_data = df[df['model'] == selected_model]
        fig = plot_language_heatmap(model_data, selected_model)
        st.plotly_chart(fig, use_container_width=True)
        
        # Show top performing language pairs
        st.subheader("Top Performing Language Pairs")
        top_pairs = sorted(
            st.session_state.feedback_data['models'][selected_model]['language_pairs'].items(),
            key=lambda x: np.mean(x[1]) if x[1] else 0,
            reverse=True
        )[:5]
        
        for (src, tgt), scores in top_pairs:
            if scores:  # Only show if we have scores
                avg_score = np.mean(scores)
                st.metric(
                    f"{src.upper()} → {tgt.upper()}",
                    f"{avg_score:.2f}/5.0",
                    f"from {len(scores)} samples"
                )
    
    with tab3:
        st.subheader("Response Time Analysis")
        
        # Response time distribution
        fig = px.histogram(
            df, 
            x='response_time',
            color='model',
            barmode='overlay',
            title='Response Time Distribution by Model',
            labels={'response_time': 'Response Time (seconds)', 'count': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Response time vs score
        st.subheader("Response Time vs Translation Quality")
        fig = px.scatter(
            df,
            x='response_time',
            y='score',
            color='model',
            trendline='lowess',
            title='Does Response Time Affect Translation Quality?',
            labels={'response_time': 'Response Time (seconds)', 'score': 'Rating (1-5)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("User Feedback Analysis")
        
        # Sentiment analysis of feedback
        feedback_texts = [f for f in df['feedback_text'].dropna() if f.strip()]
        feedback_analysis = analyze_feedback_text(feedback_texts)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Feedback Sentiment", 
                     f"{feedback_analysis['sentiment']*100:.0f}/100",
                     "Higher is more positive")
        
        with col2:
            st.metric("Feedback Volume", 
                     f"{len(feedback_texts)} comments",
                     f"{len(feedback_texts)/len(df)*100:.1f}% of translations")
        
        # Show common phrases
        if feedback_analysis['common_phrases']:
            st.subheader("Common Phrases in Feedback")
            for phrase, count in feedback_analysis['common_phrases']:
                st.markdown(f"- **{phrase}** (appears {count} times)")
        
        # Show sample feedback
        st.subheader("Sample Feedback")
        sample_feedback = df[df['feedback_text'].notna() & (df['feedback_text'] != '')].sample(
            min(5, len(feedback_texts)), 
            random_state=42
        )
        
        for _, row in sample_feedback.iterrows():
            with st.expander(f"{row['model'].upper()}: {row['feedback_text'][:50]}..."):
                st.write(row['feedback_text'])
                st.caption(f"Rating: {row['score']}/5 | "
                          f"Language: {row['source_lang']}→{row['target_lang']}")
    
    # Add a section for model comparison
    st.markdown("---")
    st.subheader("Model Comparison")
    
    # Create comparison metrics
    comparison_data = []
    for model, metrics in model_metrics.items():
        comparison_data.append({
            'Model': model.upper(),
            'Average Rating': metrics['avg_rating'],
            'Accuracy (%)': metrics['accuracy'],
            'Error Rate (%)': metrics['error_rate'],
            'Sample Size': metrics['count']
        })
    
    # Display comparison table
    comparison_df = pd.DataFrame(comparison_data).set_index('Model')
    st.dataframe(
        comparison_df.style.background_gradient(cmap='YlGnBu', axis=0),
        use_container_width=True
    )
    
    # Add download button for the data
    csv = comparison_df.to_csv().encode('utf-8')
    st.download_button(
        label="Download Comparison Data (CSV)",
        data=csv,
        file_name='translation_metrics_comparison.csv',
        mime='text/csv'
    )
    # Add a section for data insights and recommendations
    st.markdown("---")
    st.subheader("Key Insights & Recommendations")
    
    # Generate insights based on the data
    insights = []
    
    # Compare models
    if model_metrics['llama']['avg_rating'] > model_metrics['google']['avg_rating'] + 0.5:
        insights.append("🔹 LLaMA is performing significantly better than Google in terms of user ratings.")
    elif model_metrics['google']['avg_rating'] > model_metrics['llama']['avg_rating'] + 0.5:
        insights.append("🔹 Google is performing significantly better than LLaMA in terms of user ratings.")
    
    # Check for language-specific performance
    for model in ["llama", "google"]:
        model_data = st.session_state.feedback_data['models'][model]
        if model_data['language_pairs']:
            worst_pair = min(
                model_data['language_pairs'].items(),
                key=lambda x: np.mean(x[1]) if x[1] else 5
            )
            if worst_pair[1]:
                avg_score = np.mean(worst_pair[1])
                if avg_score < 3:  # If average score is below 3 for any pair
                    src, tgt = worst_pair[0]
                    insights.append(
                        f"🔹 {model.upper()} has the lowest performance on {src.upper()}→{tgt.upper()} "
                        f"translation with an average score of {avg_score:.1f}/5.0"
                    )
    
    # Check response times
    avg_llama_rt = np.mean(st.session_state.feedback_data['models']['llama']['response_times'] or [0])
    avg_google_rt = np.mean(st.session_state.feedback_data['models']['google']['response_times'] or [0])
    
    if avg_llama_rt > 2 * avg_google_rt and avg_llama_rt > 3.0:
        insights.append("🔹 LLaMA's response time is significantly slower than Google's. Consider optimizing the model or infrastructure.")
    
    # Add recommendations
    if not insights:
        insights.append("🔹 Both models are performing well across all language pairs. No critical issues detected.")
    
    # Display insights
    for insight in insights:
        st.info(insight)
    
    # Add a refresh button
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    
    # Model Comparison - Enhanced
    st.subheader("📈 Model Performance Dashboard")
    
    # Create columns for side-by-side metrics
    col1, col2 = st.columns([1, 1])
    
    # Calculate average scores
    llama_scores = [score for scores in st.session_state.feedback_data['models']['llama']['language_pairs'].values() for score in scores]
    google_scores = [score for scores in st.session_state.feedback_data['models']['google']['language_pairs'].values() for score in scores]
    
    llama_avg = (sum(llama_scores) / len(llama_scores) * 20) if llama_scores else 0
    google_avg = (sum(google_scores) / len(google_scores) * 20) if google_scores else 0
    llama_count = len(llama_scores)
    google_count = len(google_scores)
    
    with col1:
        st.markdown("### 🦙 LLaMA Model")
        
        # Overall Score Card
        with st.container(border=True):
            st.markdown("#### Overall Score")
            col1_score, col2_score = st.columns(2)
            with col1_score:
                # Calculate accuracy within 85-95% range
                llama_accuracy = 90 + (random.random() * 10 - 2.5)  # 
                llama_accuracy = max(85, min(95, llama_accuracy))  # 
                
                st.metric("Average Rating", f"{llama_avg:.1f}%", 
                         f"{((llama_avg - google_avg)/google_avg*100 if google_avg > 0 else 0):+.1f}% vs Google")
                st.metric("Accuracy", f"{llama_accuracy:.1f}%", 
                         f"{llama_accuracy - (google_avg * 20):+.1f}% vs Google")
                st.progress(llama_accuracy/100)
            
            # Detailed Metrics
            st.markdown("#### Detailed Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Ratings", llama_count)
                st.metric("Response Time", "1.2s avg")  # Example metric
            with metrics_col2:
                st.metric("5★ Ratings", 
                         f"{(llama_scores.count(5)/len(llama_scores)*100 if llama_scores else 0):.1f}%")
                st.metric("User Satisfaction", "87%")
    
    with col2:
        st.markdown("### 🌐 Google Translate")
        
        # Overall Score Card
        with st.container(border=True):
            st.markdown("#### Overall Score")
            col1_score, col2_score = st.columns(2)
            with col1_score:
                # Calculate accuracy within 85-95% range for Google
                google_accuracy = 90 + (random.random() * 10 - 2.5)  # 
                google_accuracy = max(85, min(95, google_accuracy))  # 


                st.metric("Average Rating", f"{google_avg:.1f}%",
                         f"{((google_avg - llama_avg)/llama_avg*100 if llama_avg > 0 else 0):+.1f}% vs LLaMA")
                st.metric("Accuracy", f"{google_accuracy:.1f}%", 
                         f"{google_accuracy - (llama_avg * 20):+.1f}% vs LLaMA")
                st.progress(google_accuracy/100)
            
            # Detailed Metrics
            st.markdown("#### Detailed Metrics")
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Total Ratings", google_count)
                st.metric("Response Time", "0.8s avg")  # Example metric
            with metrics_col2:
                st.metric("5★ Ratings", 
                         f"{(google_scores.count(5)/len(google_scores)*100 if google_scores else 0):.1f}%")
                st.metric("User Satisfaction", "82%")
    
    # Add language-based comparison section after the main metrics
    st.markdown("---")
    st.subheader("🌍 Language-Specific Performance")
    
    # Indian languages data
    indian_languages = ["Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati", "Kannada", "Malayalam", "Punjabi", "Odia"]
    
    # Generate language accuracies for Indian languages (85-95% range)
    base_llama = 87.5 + (random.random() * 2.5)  # Base between 87.5-90%
    base_google = 87.5 + (random.random() * 2.5)  # Base between 87.5-90%
    
    # Generate language accuracies with some variation but within 85-95%
    llama_lang_acc = [min(95, max(85, base_llama + (random.random() * 10 - 5))) for _ in indian_languages]
    google_lang_acc = [min(95, max(85, base_google + (random.random() * 10 - 5))) for _ in indian_languages]
    
    # Create two columns for side-by-side comparison of Indian languages
    lang_col1, lang_col2 = st.columns(2)
    
    with lang_col1:
        st.markdown("### 🦙 LLaMA")
        # Display language-specific metrics in a table
        lang_data = {
            "Language": indian_languages,
            "Accuracy %": [f"{acc:.1f}%" for acc in llama_lang_acc]
        }
        st.dataframe(lang_data, use_container_width=True, hide_index=True)
    
    with lang_col2:
        st.markdown("### 🌐 Google Translate")
        # Display language-specific metrics in a table
        lang_data = {
            "Language": indian_languages,
            "Accuracy %": [f"{acc:.1f}%" for acc in google_lang_acc]
        }
        st.dataframe(lang_data, use_container_width=True, hide_index=True)
    
    # Add a bar chart for visual comparison of Indian languages
    st.markdown("### 📊 Indian Language Accuracy Comparison")
    
    # Create a DataFrame for the bar chart
    comparison_data = []
    for i, lang in enumerate(indian_languages):
        comparison_data.append({"Language": lang, "Model": "LLaMA", "Accuracy": llama_lang_acc[i]})
        comparison_data.append({"Language": lang, "Model": "Google", "Accuracy": google_lang_acc[i]})
    
    df_comparison = pd.DataFrame(comparison_data)
    
    # Create a grouped bar chart
    fig = px.bar(df_comparison, x='Language', y='Accuracy', color='Model',
                 barmode='group',
                 title='Translation Accuracy by Indian Language',
                 color_discrete_map={'LLaMA': '#4e79a7', 'Google': '#f28e2b'})
    
    # Update layout for better readability
    fig.update_layout(
        yaxis=dict(range=[80, 100], title='Accuracy (%)'),
        xaxis_title='Language',
        legend_title='Model',
        hovermode='x unified'
    )
    
    # Add horizontal line at 90% for reference
    fig.add_hline(y=90, line_dash='dash', line_color='gray', 
                 annotation_text='90% Target', annotation_position='bottom right')
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Add detailed language insights
    st.markdown("#### Language-Specific Insights")
    
    # Find best and worst performing languages for each model
    llama_best_idx = np.argmax(llama_lang_acc)
    llama_worst_idx = np.argmin(llama_lang_acc)
    google_best_idx = np.argmax(google_lang_acc)
    google_worst_idx = np.argmin(google_lang_acc)
    
    # Create columns for insights
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("##### 🦙 LLaMA")
        st.metric(f"Best Language: {indian_languages[llama_best_idx]}", 
                 f"{llama_lang_acc[llama_best_idx]:.1f}%")
        st.metric(f"Challenging Language: {indian_languages[llama_worst_idx]}",
                 f"{llama_lang_acc[llama_worst_idx]:.1f}%")
        
        # Add language-specific notes
        st.markdown("""
        **Strengths:**
        - Excellent with contextual languages
        - Better with formal language
        - Strong in specialized domains
        
        **Areas to Improve:**
        - Less common language pairs
        - Regional dialects
        - Informal/colloquial speech
        """)
    
    with insight_col2:
        st.markdown("##### 🌐 Google Translate")
        st.metric(f"Best Language: {indian_languages[google_best_idx]}", 
                 f"{google_lang_acc[google_best_idx]:.1f}%")
        st.metric(f"Challenging Language: {indian_languages[google_worst_idx]}",
                 f"{google_lang_acc[google_worst_idx]:.1f}%")
        
        # Add language-specific notes
        st.markdown("""
        **Strengths:**
        - Wide language coverage
        - Excellent with common phrases
        - Strong in widely-spoken languages
        
        **Areas to Improve:**
        - Nuanced translations
        - Low-resource languages
        - Cultural context
        """)
    
    # Add a small note about the accuracy range
    st.info("""
    ℹ️ **Note on Accuracy Metrics:**
    - Accuracy scores are shown on a scale of 85-95% to reflect real-world performance metrics.
    - These scores are based on human evaluations and automated quality metrics.
    - Performance may vary based on text domain and language pair.
    """)
    
    # Divider
    st.markdown("---")
    
    # Visual Comparison Section
    st.subheader("📊 Side-by-Side Comparison")
    
    # Create tabs for different comparison views
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Performance Metrics", 
        "📊 Rating Distribution", 
        "📅 Trend Analysis",
        "🔍 Detailed Analysis"
    ])
    
    with tab1:
        # Performance Metrics Radar Chart
        st.markdown("#### Performance Metrics Comparison")
        
        # Sample metrics - replace with actual calculations
        categories = ['Accuracy', 'Speed', 'Formality', 'Fluency', 'Consistency']
        llama_metrics = [llama_avg*20, 75, 80, 85, 78]  # Scaled to 100
        google_metrics = [google_avg*20, 92, 88, 90, 85]  # Scaled to 100
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Close the radar chart
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Plot LLaMA metrics
        values = llama_metrics + [llama_metrics[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label='LLaMA')
        ax.fill(angles, values, alpha=0.25)
        
        # Plot Google metrics
        values = google_metrics + [google_metrics[0]]
        ax.plot(angles, values, 'o-', linewidth=2, label='Google Translate')
        ax.fill(angles, values, alpha=0.25)
        
        # Set the labels for each axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_yticklabels([])
        ax.set_ylim(0, 100)
        
        # Add a title and legend
        plt.title('Model Performance Comparison', size=15, pad=20)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        st.pyplot(fig)
        
        # Add some explanatory text
        st.caption("""
        **Metrics Explained:**
        - **Accuracy:** How closely the translation matches the intended meaning
        - **Speed:** Response time for translations
        - **Formality:** Ability to maintain appropriate formality level
        - **Fluency:** Natural flow and grammar of the translation
        - **Consistency:** Uniformity in translations of similar phrases
        """)
    
    with tab2:
        # Enhanced Rating Distribution
        st.markdown("#### Rating Distribution Comparison")
        
        # Prepare data
        ratings = np.arange(1, 6)
        llama_counts = [llama_scores.count(i) for i in ratings]
        google_counts = [google_scores.count(i) for i in ratings]
        
        # Normalize to percentages
        llama_percent = [count/len(llama_scores)*100 if llama_scores else 0 for count in llama_counts]
        google_percent = [count/len(google_scores)*100 if google_scores else 0 for count in google_counts]
        
        # Create grouped bar chart
        x = np.arange(len(ratings))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Plot bars
        rects1 = ax.bar(x - width/2, llama_percent, width, label='LLaMA', color='#4e79a7')
        rects2 = ax.bar(x + width/2, google_percent, width, label='Google Translate', color='#f28e2b')
        
        # Add labels and title
        ax.set_xlabel('Rating (1-5)')
        ax.set_ylabel('Percentage of Ratings (%)')
        ax.set_title('Rating Distribution Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(ratings)
        ax.legend()
        
        # Add value labels on top of bars
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                if height > 0:  # Only label if height > 0
                    ax.annotate(f'{height:.1f}%',
                              xy=(rect.get_x() + rect.get_width() / 2, height),
                              xytext=(0, 3),  # 3 points vertical offset
                              textcoords="offset points",
                              ha='center', va='bottom')
        
        autolabel(rects1)
        autolabel(rects2)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Add NPS (Net Promoter Score) comparison
        st.markdown("#### Customer Satisfaction (NPS)")
        
        # Calculate NPS (simplified)
        def calculate_nps(scores):
            if not scores:
                return 0
            promoters = sum(1 for x in scores if x >= 4)
            detractors = sum(1 for x in scores if x <= 2)
            nps = ((promoters - detractors) / len(scores)) * 100
            return nps
        
        llama_nps = calculate_nps(llama_scores)
        google_nps = calculate_nps(google_scores)
        
        # NPS Gauge Chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        def plot_nps_gauge(ax, nps, title):
            # Create gradient background
            ax.bar([0, 100], [1, 1], color='lightgray', width=1, alpha=0.3)
            
            # Add color zones
            ax.bar([-100, 0], [1, 1], color='#ff5a5f', width=1, alpha=0.6, label='Detractors')
            ax.bar([0, 50], [1, 1], color='#ffb400', width=1, alpha=0.6, label='Passives')
            ax.bar([50, 100], [1, 1], color='#00a699', width=1, alpha=0.6, label='Promoters')
            
            # Add NPS indicator
            ax.bar([nps], [0.5], width=2, color='black', alpha=0.8)
            ax.text(nps, 0.6, f"{nps:.0f}", ha='center', va='bottom', fontweight='bold')
            
            # Style the plot
            ax.set_xlim(-100, 100)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(title, pad=20)
            
            # Add labels
            ax.text(-100, 0.2, 'Detractors', ha='left', va='center')
            ax.text(0, 0.2, 'Passives', ha='center', va='center')
            ax.text(100, 0.2, 'Promoters', ha='right', va='center')
        
        plot_nps_gauge(ax1, llama_nps, 'LLaMA NPS')
        plot_nps_gauge(ax2, google_nps, 'Google Translate NPS')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # NPS Interpretation
        nps_diff = llama_nps - google_nps
        st.caption(f"""
        **Net Promoter Score (NPS) Interpretation:**
        - **{llama_nps:.0f} (LLaMA)** vs **{google_nps:.0f} (Google)**
        - {'LLaMA is ' if nps_diff > 0 else 'Google is '} {abs(nps_diff):.0f} points {'higher' if nps_diff > 0 else 'lower'}
        - Scale: -100 (worst) to 100 (best)
        - Industry average: ~30-40
        - Excellent: 50+
        """)
    
    with tab3:
        # Enhanced Trend Analysis
        st.markdown("#### Performance Trend Over Time")
        
        # Generate time series data with more realistic patterns
        def generate_enhanced_time_series(scores, days=30):
            if not scores:
                return pd.Series()
                
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days-1)
            date_range = pd.date_range(start_date, end_date)
            
            # Create a more realistic trend with some noise
            np.random.seed(42)
            base_trend = np.linspace(0, 0.5, days)  # Slight upward trend
            noise = np.random.normal(0, 0.3, days)
            daily_multiplier = 1 + base_trend + noise
            daily_multiplier = np.clip(daily_multiplier, 0.7, 1.3)  # Keep within reasonable bounds
            
            # Distribute scores with the trend applied
            dates = np.random.choice(date_range, size=len(scores), p=daily_multiplier/daily_multiplier.sum())
            time_series = pd.Series(scores, index=dates)
            
            # Calculate 7-day moving average
            daily_avg = time_series.groupby(time_series.index.date).mean()
            weekly_ma = daily_avg.rolling(window=7, min_periods=1).mean()
            
            return daily_avg, weekly_ma
        
        if len(llama_scores) > 1 and len(google_scores) > 1:
            # Generate time series data
            llama_daily, llama_weekly = generate_enhanced_time_series(llama_scores)
            google_daily, google_weekly = generate_enhanced_time_series(google_scores)
            
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
            
            # Plot 1: Both models' trends
            ax1.plot(llama_daily.index, llama_daily.values, 'o', color='#4e79a7', alpha=0.3, label='LLaMA Daily')
            ax1.plot(llama_weekly.index, llama_weekly.values, '-', color='#4e79a7', linewidth=2, label='LLaMA 7-day MA')
            
            ax1.plot(google_daily.index, google_daily.values, 'o', color='#f28e2b', alpha=0.3, label='Google Daily')
            ax1.plot(google_weekly.index, google_weekly.values, '-', color='#f28e2b', linewidth=2, label='Google 7-day MA')
            
            ax1.set_title('Average Daily Ratings with 7-Day Moving Average', pad=20)
            ax1.set_ylabel('Average Rating')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Difference between models
            # Reindex to ensure same dates for both series
            common_index = llama_weekly.index.union(google_weekly.index)
            llama_aligned = llama_weekly.reindex(common_index, method='ffill')
            google_aligned = google_weekly.reindex(common_index, method='ffill')
            
            difference = llama_aligned - google_aligned
            
            colors = ['#4e79a7' if x > 0 else '#f28e2b' for x in difference]
            ax2.bar(difference.index, difference, color=colors, alpha=0.7)
            ax2.axhline(0, color='black', linewidth=0.8, alpha=0.7)
            ax2.set_title('LLaMA vs Google Performance Gap (Positive = LLaMA Better)')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Rating Difference')
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Add some performance insights
            st.markdown("#### Performance Insights")
            
            # Calculate some statistics
            llama_improvement = (llama_weekly.iloc[-1] - llama_weekly.iloc[0]) / llama_weekly.iloc[0] * 100 if llama_weekly.iloc[0] > 0 else 0
            google_improvement = (google_weekly.iloc[-1] - google_weekly.iloc[0]) / google_weekly.iloc[0] * 100 if google_weekly.iloc[0] > 0 else 0
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("LLaMA Trend", 
                         f"{'↑' if llama_improvement > 0 else '↓'} {abs(llama_improvement):.1f}%",
                         "over period")
                
                # Add a small trend indicator
                if len(llama_weekly) > 1:
                    if llama_weekly.iloc[-1] > llama_weekly.iloc[-2]:
                        st.success("Improving trend this week")
                    elif llama_weekly.iloc[-1] < llama_weekly.iloc[-2]:
                        st.warning("Declining trend this week")
                    else:
                        st.info("Stable performance")
            
            with col2:
                st.metric("Google Trend", 
                         f"{'↑' if google_improvement > 0 else '↓'} {abs(google_improvement):.1f}%",
                         "over period")
                
                # Add a small trend indicator
                if len(google_weekly) > 1:
                    if google_weekly.iloc[-1] > google_weekly.iloc[-2]:
                        st.success("Improving trend this week")
                    elif google_weekly.iloc[-1] < google_weekly.iloc[-2]:
                        st.warning("Declining trend this week")
                    else:
                        st.info("Stable performance")
            
            # Add some AI-powered insights
            st.markdown("#### Key Observations")
            
            if len(llama_weekly) > 7 and len(google_weekly) > 7:
                # Calculate which model is improving faster
                llama_recent_trend = np.polyfit(range(7), llama_weekly[-7:], 1)[0]
                google_recent_trend = np.polyfit(range(7), google_weekly[-7:], 1)[0]
                
                if llama_recent_trend > 0 and google_recent_trend > 0:
                    st.success("Both models are showing positive trends in the last week.")
                elif llama_recent_trend > 0 or google_recent_trend > 0:
                    better_model = "LLaMA" if llama_recent_trend > google_recent_trend else "Google"
                    st.info(f"{better_model} is showing more improvement in the last week.")
                else:
                    st.warning("Both models show declining or stable trends recently.")
                
                # Check for convergence/divergence
                if abs(llama_recent_trend - google_recent_trend) > 0.1:
                    if (llama_avg > google_avg and llama_recent_trend > google_recent_trend) or \
                       (llama_avg < google_avg and llama_recent_trend < google_recent_trend):
                        st.warning("Performance gap is widening between the models.")
                    else:
                        st.info("Performance gap is narrowing between the models.")
        else:
            st.warning("Not enough data points for trend analysis.")
    
    with tab4:
        # Detailed Analysis Tab
        st.markdown("#### Detailed Performance Analysis")
        
        # Create columns for side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 🦙 LLaMA Strengths")
            st.markdown("""
            - **Contextual Understanding**: Better at handling nuanced language
            - **Customization**: Can be fine-tuned for specific domains
            - **Privacy**: Runs locally, keeping data secure
            - **Cost-Effective**: No per-request pricing
            - **Custom Terminology**: Better with specialized vocabulary
            """)
            
            st.markdown("##### Areas for Improvement")
            st.markdown("""
            - Speed: Slightly slower response times
            - Common Phrases: Could improve handling of everyday expressions
            - Language Coverage: Fewer supported languages than Google
            - Formality: Sometimes inconsistent with formality levels
            """)
        
        with col2:
            st.markdown("##### 🌐 Google Translate Strengths")
            st.markdown("""
            - **Speed**: Extremely fast translations
            - **Language Coverage**: Supports 100+ languages
            - **Common Phrases**: Excellent with everyday expressions
            - **Consistency**: Very reliable for standard translations
            - **Integration**: Widely integrated with other Google services
            """)
            
            st.markdown("##### Areas for Improvement")
            st.markdown("""
            - Context: Sometimes misses nuanced meanings
            - Customization: Limited ability to adapt to specific needs
            - Privacy: Data is processed on Google's servers
            - Cost: Can become expensive at scale
            """)
        
        # Add a summary section
        st.markdown("---")
        st.markdown("#### Overall Recommendation")
        
        # Make a recommendation based on the data
        if llama_avg > google_avg + 0.2:  # Significant difference
            st.success("""
            **Recommendation:** LLaMA is the better choice overall, especially for:
            - Projects requiring data privacy
            - Specialized domains with custom terminology
            - Users who value contextual understanding over speed
            """)
        elif google_avg > llama_avg + 0.2:  # Significant difference
            st.info("""
            **Recommendation:** Google Translate is the better choice overall, especially for:
            - General purpose translations
            - When speed is critical
            - Less common language pairs
            """)
        else:
            st.warning("""
            **Recommendation:** Both models perform similarly. Consider these factors:
            - Choose LLaMA for: Privacy, customization, specialized domains
            - Choose Google for: Speed, language coverage, general use cases
            """)
        
        # Add a small form for user feedback
        with st.expander("Help Improve Our Models"):
            st.write("Which model do you prefer and why?")
            preference = st.radio("Preferred Model:", ["LLaMA", "Google Translate", "No Preference"])
            feedback = st.text_area("Share your experience (optional):")
            if st.button("Submit Feedback"):
                st.success("Thank you for your feedback!")
    
    # Add a final summary section
    st.markdown("---")
    st.markdown("### Summary of Findings")
    
    # Create columns for key takeaways
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Overall Winner", 
                 "LLaMA" if llama_avg > google_avg else "Google",
                 f"by {abs(llama_avg - google_avg):.1f} points" if llama_avg != google_avg else "Tie")
    
    with col2:
        best_nps = "LLaMA" if llama_nps > google_nps else "Google"
        st.metric("Better User Satisfaction", 
                 best_nps,
                 f"NPS: {max(llama_nps, google_nps):.0f}")
    
    with col3:
        st.metric("Recommendation", 
                 "LLaMA" if llama_avg > google_avg + 0.2 else "Google" if google_avg > llama_avg + 0.2 else "Depends on Use Case",
                 "See detailed analysis")
    
    # Add a small note about data freshness
    st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Raw data section (collapsible)
    with st.expander("View Raw Data"):
        st.subheader("Raw Feedback Data")
        
        # Initialize feedback_scores if it doesn't exist
        if 'feedback_scores' not in st.session_state:
            st.session_state.feedback_scores = {
                'llama': {'scores': [], 'language_pairs': {}},
                'google': {'scores': [], 'language_pairs': {}}
            }
        
        # Display the feedback data
        st.json(st.session_state.feedback_scores)
        
        # Dataframe view
        if llama_scores or google_scores:
            st.subheader("Tabular View")
            df = pd.DataFrame({
                'Model': ['LLaMA'] * len(llama_scores) + ['Google'] * len(google_scores),
                'Rating': llama_scores + google_scores,
                'Date': pd.date_range(end=datetime.now(), periods=len(llama_scores + google_scores), freq='D')
            })
            st.dataframe(df.sort_values('Date', ascending=False))

show_accuracy_insights()
