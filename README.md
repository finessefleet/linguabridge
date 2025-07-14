# LinguaBridge

## 🌟 Overview
LinguaBridge is a powerful, user-friendly application that provides Multilingual translation and advanced sentiment analysis. Built with Streamlit, it combines custom language processing with an intuitive interface to deliver accurate translations and insightful sentiment analysis. The application now features enhanced distress detection and a comprehensive emotion analysis system to better understand user input.

## 📸 Screenshots

### Translation Interface
![Translation Interface](Screenshot%202025-07-14%20191404.png)

### Sentiment Analysis Results
![Sentiment Analysis](Screenshot%202025-07-14%20193532.png)

### Detailed Emotion Analysis
![Emotion Analysis](Screenshot%202025-07-14%20193549.png)

## ✨ Key Features

### 🔄 Bilingual Translation
- **Hybrid Translation System**: Combines multiple approaches for accurate English-Hindi translation
- **Real-time Processing**: Instant translation as you type
- **Context Preservation**: Maintains meaning and context during translation
- **Language Detection**: Automatically detects input language

### 📊 Advanced Sentiment Analysis
- **Multi-method Analysis**: Combines TextBlob and AFINN for robust sentiment detection
- **Comprehensive Emotion Detection**: Identifies 32+ emotions including Joy, Sadness, Anger, Fear, and more
- **Bilingual Processing**: Accurate sentiment analysis for both English and Hindi text
- **Enhanced Distress Detection**: Automatically detects concerning content and provides supportive resources
- **Prioritized Sentiment Analysis**: Ensures distress phrases are always flagged with appropriate sentiment
- **Detailed Metrics**: Polarity, subjectivity, and confidence scoring with normalized values

### 📈 Performance Insights
- **Model Comparison**: Side-by-side comparison of different translation models
- **User Feedback**: Collects and analyzes user ratings and feedback
- **Interactive Visualizations**: Charts and graphs for performance metrics

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- Internet connection (for translation services and NLTK data download)

### Installation
0. **Create a .env file**:
   ##Store the groq api in the file then proceed
   
2. **Clone the repository**:
   ```bash
   git clone [your-repository-url]
   cd Translator-app
   ```

3. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate  # Windows
   source venv/bin/activate  # macOS/Linux
   ```

4. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Download NLTK data**:
   ```bash
   python -c "import nltk; nltk.download(['stopwords', 'punkt', 'wordnet', 'averaged_perceptron_tagger', 'omw-1.4'])"
   ```

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run home.py
   ```

2. **Open your browser and navigate to**:
   ```
   http://localhost:8501
   ```

## 🛠 Project Structure

- `home.py`: Main application entry point
- `translator.py`: Handles bilingual translation functionality
- `sentiment_analyzer.py`: Implements sentiment analysis with distress detection
- `accuracy_insights.py`: Manages performance metrics and visualizations
- `rl_feedback.py`: Handles user feedback and reinforcement learning

## 🤝 Contributing

We welcome contributions to improve LinguaBridge! Here's how you can help:

1. Fork the repository
2. Create a new branch for your feature: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Create a new Pull Request

### Areas for Contribution:
- Expand the distress phrase database
- Add support for more Indian languages
- Improve emotion detection accuracy
- Enhance the UI/UX
- Add more comprehensive tests


## 📧 Contact

For any questions or feedback, please open an issue in the repository.

## 🙏 Acknowledgments

- Built with ❤️ using Streamlit
- Utilizes NLTK, TextBlob, and AFINN for natural language processing
- Special thanks to the open-source community for their valuable contributions
