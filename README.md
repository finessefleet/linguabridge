# 🌐 Linguabridge Pro - Enhanced Code-Mix Translation

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Linguabridge Pro is an advanced AI-powered translation system specializing in code-mixed languages. It provides high-quality translations between English and various code-mixed languages like Hinglish, Banglish, Tanglish, and more, with advanced features like semantic matching and neural machine translation.

## 🚀 Key Features

### 🔤 Enhanced Code-Mix Translation
- **Multiple Code-Mix Languages**: Support for Hinglish, Banglish, Tanglish, Manglish, and more
- **Neural Machine Translation**: Powered by state-of-the-art transformer models
- **Semantic Matching**: Advanced matching using sentence embeddings for better context understanding
- **Fuzzy Matching**: Intelligent fuzzy matching for partial or slightly incorrect inputs
- **Word-by-Word Fallback**: Graceful fallback to word-by-word translation when needed

### 🎯 Smart Features
- **Context-Aware Translation**: Understands and maintains context during translation
- **Alternative Translations**: Provides multiple translation options when available
- **Language Detection**: Automatic detection of input language
- **Performance Optimized**: Fast and efficient with intelligent caching

### 🛠 Technical Highlights
- **Transformer Models**: Utilizes mBART and other transformer architectures
- **Sentence Embeddings**: Uses sentence-transformers for semantic similarity
- **Modular Architecture**: Easy to extend with new languages and models
- **Production Ready**: Built with scalability and performance in mind

### 📈 Insights & Analytics
- **Model Performance**: Compare accuracy across different models
- **Confusion Matrices**: Visualize model performance for different languages
- **Feedback Analysis**: Track user feedback and ratings
- **Interactive Charts**: Built with Plotly for detailed exploration

### 🎨 User Experience
- **Modern UI**: Clean, responsive interface built with Streamlit
- **Dark/Light Mode**: Choose your preferred theme
- **Translation History**: Keep track of your translations
- **Responsive Design**: Works on desktop and mobile devices

## 🛠 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/linguabridge-pro.git
   cd linguabridge-pro
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   Note: You might need to install PyTorch separately based on your system configuration. 
   Visit [PyTorch's official website](https://pytorch.org/get-started/locally/) for installation instructions.

4. **Download pre-trained models**:
   The first time you run the application, it will automatically download the required models.
   
   For offline use, you can download them manually:
   ```bash
   python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
   ```

5. **Run the enhanced application**:
   ```bash
   streamlit run app_enhanced.py
   ```

6. **Open your browser** and navigate to `http://localhost:8501`

## 🚀 Quick Start

### Using the Enhanced Code-Mix Translator

1. **Select Target Language**:
   - Choose from the available code-mix languages (Hinglish, Banglish, Tanglish, etc.)
   - The system will automatically detect the input language (English)

2. **Enter Your Text**:
   - Type or paste your English text in the input area
   - The system supports both short phrases and longer paragraphs

3. **Translation Options**:
   - **Neural Machine Translation**: Toggle to use advanced NMT models (recommended)
   - **Show Alternatives**: Enable to see multiple translation options

4. **Get Results**:
   - Click "Translate Now" to see the code-mixed translation
   - View alternative translations if available
   - Copy the result with the copy button

### Example Translations

Try these examples to see the enhanced code-mix translation in action:

- **Hinglish**: "What are you doing today?" → "Aaj tum kya kar rahe ho?"
- **Banglish**: "I am going to the market" → "Ami bazar jacchi"
- **Tanglish**: "How much does this cost?" → "Ithuku enna vilai?"
- **Manglish**: "Where is the nearest restaurant?" → "Adivasiya restaurant evideya?"

### Advanced Features

- **Context-Aware Translation**: The system maintains context for better translations
- **Semantic Matching**: Finds the most appropriate translation even with slightly different phrasing
- **Word-by-Word Fallback**: Provides reasonable translations even for unknown phrases
- **Performance Optimized**: Uses caching for faster translations of repeated phrases
   - View translation and sentiment analysis

## 🧰 Project Structure

```
Linguabridge-Pro/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration settings and constants
├── translation_models.py  # Translation model implementations
├── utils.py               # Utility functions and helpers
├── requirements_updated.txt  # Project dependencies
└── README.md              # This file
```

## 📚 Documentation

### Models

1. **TextBlob**
   - Rule-based sentiment analysis
   - Fast but less accurate for Indian languages
   - Best for English and simple text

2. **Indic-BERT**
   - BERT-based model fine-tuned for Indian languages
   - High accuracy for formal text
   - Supports 11 Indian languages

3. **CNN-LSTM**
   - Hybrid deep learning model
   - Good for code-mixed text
   - Balanced performance across languages

### API Reference

#### `translate(text: str, source_lang: str, target_lang: str) -> Dict`
Translates text from source language to target language.

**Parameters:**
- `text`: Input text to translate
- `source_lang`: Source language code (e.g., 'hi', 'en')
- `target_lang`: Target language code

**Returns:**
```json
{
  "translation": "translated text",
  "source_lang": "source language code",
  "target_lang": "target language code",
  "confidence": 0.95
}
```

#### `analyze_sentiment(text: str, model: str = 'textblob') -> Dict`
Analyzes sentiment of the input text using the specified model.

**Parameters:**
- `text`: Input text to analyze
- `model`: Model to use ('textblob', 'indic-bert', 'cnn-lstm')

**Returns:**
```json
{
  "sentiment": "positive/negative/neutral",
  "score": 0.85,
  "model": "model_name"
}
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for pre-trained models
- [Streamlit](https://streamlit.io/) for the amazing web framework
- [TextBlob](https://textblob.readthedocs.io/) for sentiment analysis
- [Indic NLP Library](https://github.com/anoopkunchukuttan/indic_nlp_library) for Indian language support
- **Sentiment Analysis**: Understand the emotional tone of translated text
- **Interactive Chat**: Have conversations that automatically translate between languages
- **Translation History**: Keep track of your recent translations
- **Confidence Scoring**: See how confident the system is in its translations
- **Responsive UI**: Clean, modern interface built with Streamlit

## 🌍 Supported Languages

The translator supports the following Indian languages:

| Language | Code | Script |
|----------|------|---------|
| Assamese | as | Assamese |
| Bengali | bn | Bengali |
| Bodo | brx | Devanagari |
| Dogri | doi | Devanagari |
| English | en | Latin |
| Gujarati | gu | Gujarati |
| Hindi | hi | Devanagari |
| Kannada | kn | Kannada |
| Kashmiri | ks | Arabic |
| Konkani | gom | Devanagari |
| Maithili | mai | Devanagari |
| Malayalam | ml | Malayalam |
| Manipuri (Meitei) | mni | Bengali |
| Marathi | mr | Devanagari |
| Nepali | ne | Devanagari |
| Odia | or | Odia |
| Punjabi | pa | Gurmukhi |
| Sanskrit | sa | Devanagari |
| Santali | sat | Ol Chiki |
| Sindhi | sd | Arabic |
| Tamil | ta | Tamil |
| Telugu | te | Telugu |
| Urdu | ur | Arabic |

### Code-Mixed Language Support

The translator also handles these common code-mixed variations:

- **Hinglish**: Hindi + English
- **Tanglish**: Tamil + English  
- **Manglish**: Malayalam + English
- **Kanglish**: Kannada + English
- **Banglish**: Bengali + English
- **Punglish**: Punjabi + English
## 🚀 Installation

### Using pip (Recommended)

```bash
pip install indian-language-translator
```

### From Source

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/indian-language-translator.git
   cd indian-language-translator
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the package in development mode**:
   ```bash
   pip install -e .
   ```

4. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/indian-language-translator.git
   cd indian-language-translator
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download models** (if not included):
   - The application will automatically download required models on first run
   - For offline use, download models in advance and place them in the `models/` directory

## 💻 Usage

### Command Line Interface (CLI)

```bash
# Basic translation
indian-translator "नमस्ते दुनिया" --source hi --target en

# Interactive mode
indian-translator

# Help menu
indian-translator --help
```

### Python API

```python
from indiantranslator import translate, detect_language

# Simple translation
translation = translate("नमस्ते", source="hi", target="en")
print(f"Translation: {translation}")

# Auto-detect language
detected = detect_language("नमस्ते")
print(f"Detected language: {detected}")

# Batch translation
translations = translate_batch(
    ["नमस्ते", "धन्यवाद"],
    source="hi",
    target="ta"
)
```

### Web Interface

```bash
# Start the Streamlit web app
streamlit run indiantranslator/web/app.py
```

## 🛠 Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=indiantranslator
```

### Code Formatting

```bash
# Auto-format code
black .

# Sort imports
isort .
```

### Building Documentation

```bash
# Build HTML documentation
cd docs
make html
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, or suggest new features.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this project.
- Special thanks to the open-source community for the amazing libraries that made this project possible.

## 📞 Contact

For questions or support, please open an issue on GitHub or contact [Your Name] at your.email@example.com.

1. **Start the application**:
   ```bash
   streamlit run main.py
   ```

2. **Using the interface**:
   - **Translate Tab**: Enter text, select source and target languages, and click "Translate"
   - **Chat Tab**: Have interactive conversations that automatically translate between languages
   - **Insights Tab**: View your translation history and sample translations

3. **Key Features**:
   - **Auto-detect language**: Select "Auto-detect" to automatically identify the source language
   - **Sentiment Analysis**: Toggle to analyze the emotional tone of translated text
   - **Model Selection**: The app automatically selects the best translation model based on feedback
   - **Feedback**: Rate translations to help improve model selection

## 🛠️ Configuration

Customize the application by creating a `.env` file in the project root:

```env
# API Keys (if using premium services)
GOOGLE_TRANSLATE_API_KEY=your_api_key_here

# Model Paths
INDIC_BERT_PATH=./models/indic-bert

# UI Settings
THEME=light  # or dark
LANGUAGE=en  # UI language
```

## 🧪 Testing

Run the test suite with:

```bash
python -m pytest tests/
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Google Research](https://research.google/) for the IndicTrans models
- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for their work on Indian language technologies
- The open-source community for their contributions to NLP and machine learning

---

<div align="center">
  Made with ❤️ in India | <a href="https://github.com/yourusername/indian-language-translator/issues">Report Issues</a>
</div>

## Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:finessefleet/linguabridge.git
   cd linguabridge
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run translatorapp.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Enter the text you want to translate in the input box

4. Select the target language from the dropdown menu

5. Click the "Translate" button to see the results

## How It Works

The translator uses a combination of techniques to provide accurate translations:

1. **Code-Mixed Language Detection**: Uses TF-IDF vectorization and cosine similarity to detect the most likely language of the input text

2. **Translation**: 
   - First tries to find a direct match in the code-mixed dataset
   - If no good match is found, falls back to Google Translate
   - For code-mixed languages, uses a combination of pattern matching and statistical methods

3. **Confidence Scoring**: Provides a confidence score for each translation based on the similarity to known examples

## Data

The translator uses a dataset of code-mixed phrases and their translations. The dataset is stored in `code_mix.csv` and includes:

- Source language
- Code-mixed text
- English translation

## Contributing

Contributions are welcome! Here are some ways you can contribute:

1. Add more code-mixed language examples to the dataset
2. Improve the translation algorithms
3. Add support for more languages
4. Fix bugs and improve the user interface

To contribute:

1. Fork the repository
2. Create a new branch for your feature
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing web framework
- [Google Translate](https://translate.google.com/) for the translation API
- [scikit-learn](https://scikit-learn.org/) for machine learning utilities
- All contributors who have helped improve this project
