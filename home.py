
import streamlit as st
from translator import translator_interface
from accuracy_insights import show_accuracy_insights
from rl_feedback import collect_feedback
from sentiment_analyzer import sentiment_analyzer_interface

def main():
    st.set_page_config(page_title="LinguaBridge", layout="wide")
    
    # Main sidebar content
    st.sidebar.title("LinguaBridge")
    selection = st.sidebar.selectbox("Choose a feature:", ["Translator", "Sentiment Analyzer", "Accuracy Insights"])
    
    # Feedback section with separator
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Rate Your Experience")
    st.sidebar.markdown("How would you rate your experience?")
    
    # Rating options
    rating_options = ["😊 Very Good", "🙂 Good", "😐 Okay", "😕 Bad", "😡 Very Bad"]
    rating = st.sidebar.radio(
        "Select your rating:",
        rating_options,
        index=None,
        label_visibility="collapsed"
    )
    
    # Additional feedback
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Would you like to share more about your experience?")
    feedback = st.sidebar.text_area(
        "Your feedback helps us improve!",
        placeholder="Share your thoughts here...",
        label_visibility="collapsed"
    )
    
    # Submit button
    if st.sidebar.button("Submit Feedback"):
        if rating:
            st.sidebar.success("Thank you for your feedback!")
            collect_feedback(rating, feedback)
        else:
            st.sidebar.warning("Please select a rating before submitting.")

    if selection == "Translator":
        translator_interface()
    elif selection == "Sentiment Analyzer":
        sentiment_analyzer_interface()
    elif selection == "Accuracy Insights":
        

        show_accuracy_insights()


if __name__ == "__main__":
    main()