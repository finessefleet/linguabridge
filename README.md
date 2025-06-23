# 🌐 LinguaBridge - translater.py

## 📄 File: translater.py

`translater.py` is the main Streamlit application file that provides a user interface for the LinguaBridge translation and sentiment analysis system.

## 🚀 Key Features

### 🔄 Advanced Translation
- **Multi-engine Support**: Google, Microsoft, and LibreTranslate APIs
- **Language Detection**: Automatic detection of input language
- **Code-Mixing**: Handles mixed-language text (e.g., Hinglish, Spanglish)
- **Batch Processing**: Translate multiple texts at once
- **Translation Memory**: Caches frequent translations for speed
- **Quality Estimation**: Confidence scores for translations

### 🧠 Sentiment Analysis
- **Multiple Models**: TextBlob, VADER, SentiWordNet, and transformer-based models
- **Emotion Detection**: Identifies 8+ emotions in text
- **Sarcasm Detection**: Recognizes ironic or sarcastic content
- **Context-Aware**: Considers surrounding text for better accuracy
- **Multi-language Support**: Works across all supported languages

### 🌐 Language Support
- **100+ Languages**: Comprehensive global language coverage
- **Indian Languages**: Specialized support for 20+ Indian languages
- **Language Families**: Optimized for Indo-Aryan, Dravidian, and other language groups
- **Script Conversion**: Handles different writing systems

### 📊 Visualization & Analytics
- **Sentiment Distribution**: Visual breakdown of sentiment scores
- **Emotion Intensity**: Graphical representation of detected emotions
- **Word Clouds**: Visualize most frequent terms
- **Performance Metrics**: Track translation and analysis quality

### ⚙️ Technical Features
- **Asynchronous Processing**: Non-blocking operations
- **Caching System**: Improves response times
- **Modular Architecture**: Easy to extend and customize
- **Error Handling**: Graceful fallbacks and recovery
- **Logging**: Detailed logging for debugging

## 📸 Screenshots

### Main Translation Interface
![Main Translation Interface](Screenshot%202025-06-23%20231726.png)
*The main interface showing translation between English and Hindi with sentiment analysis*

### Language Selection
![Language Selection](Screenshot%202025-06-23%20231852.png)
*Selecting from over 100 supported languages including major Indian languages*

### Sentiment Analysis
![Sentiment Analysis](Screenshot%202025-06-23%20231903.png)
*Detailed sentiment analysis showing emotion distribution and confidence scores*

### Translation History
![Translation History](Screenshot%202025-06-23%20231923.png)
*View and manage your translation history with search and filter options*

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd project
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Download NLTK Data
```python
import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4', 'stopwords', 'sentiwordnet', 'averaged_perceptron_tagger', 'words'])
```

### Step 5: Set Up Environment Variables
Create a `.env` file in the project root with your API keys:
```
# Required for Google Translate
GOOGLE_TRANSLATE_API_KEY=your_google_api_key

# Required for Microsoft Translator
MICROSOFT_TRANSLATE_API_KEY=your_microsoft_api_key
MICROSOFT_TRANSLATE_REGION=your_region

# Optional: For LibreTranslate
LIBRE_TRANSLATE_API_KEY=your_api_key
LIBRE_TRANSLATE_URL=https://libretranslate.example.com

# For transformer models
HF_TOKEN=your_huggingface_token

# Application Settings
CACHE_SIZE=1000
LOG_LEVEL=INFO
ENVIRONMENT=development
```

### Step 6: Install spaCy Model
```bash
python -m spacy download en_core_web_sm
```

## ⚙️ Advanced Configuration

### Translation Settings
```python
# Configure translation settings
translator = Translator(
    service='google',  # 'google', 'microsoft', or 'libre'
    api_key='your_api_key',  # Optional if set in environment
    cache_size=1000,  # Number of translations to cache
    timeout=30,  # Request timeout in seconds
    retries=3  # Number of retry attempts
)

# Enable/disable features
translator.enable_cache(True)  # Enable/disable caching
translator.set_log_level('INFO')  # DEBUG, INFO, WARNING, ERROR
```

### Sentiment Analysis Configuration
```python
# Configure sentiment analysis
analyzer = SentimentAnalyzer(
    use_ml=True,  # Enable machine learning models
    use_transformers=True,  # Enable transformer models
    cache_size=1000,  # Cache size for analysis results
    language='en'  # Default language
)

# Customize emotion detection
analyzer.add_emotion_category(
    name='excitement',
    keywords=['excited', 'thrilled', 'ecstatic'],
    weight=0.8
)
```

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_TRANSLATE_API_KEY` | Google Cloud Translation API key | - |
| `MICROSOFT_TRANSLATE_KEY` | Microsoft Translator API key | - |
| `MICROSOFT_TRANSLATE_REGION` | Azure region | - |
| `LIBRE_TRANSLATE_URL` | Custom LibreTranslate URL | Public instance |
| `LIBRE_TRANSLATE_API_KEY` | LibreTranslate API key | - |
| `HF_TOKEN` | Hugging Face authentication token | - |
| `CACHE_SIZE` | Maximum cache size | 1000 |
| `LOG_LEVEL` | Logging level | INFO |
| `ENVIRONMENT` | Runtime environment | development |

## 🛠️ Troubleshooting

### Common Issues

#### API Key Errors
```bash
# Error: Missing API key for Google Translate
# Solution: Set the environment variable
export GOOGLE_TRANSLATE_API_KEY='your-api-key-here'
```

#### Installation Issues
```bash
# Error: NLTK data not found
python -c "import nltk; nltk.download(['punkt', 'wordnet', 'stopwords'])"

# Error: spaCy model not found
python -m spacy download en_core_web_sm
```

#### Performance Issues
```python
# Enable logging to identify bottlenecks
import logging
logging.basicConfig(level=logging.DEBUG)

# Reduce cache size if memory usage is high
translator = Translator(cache_size=500)
```

### Debugging Tips
1. Check the logs for detailed error messages
2. Verify API keys and service availability
3. Test with a simple translation to isolate issues
4. Check network connectivity for API services
5. Try with a different translation service

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**
   - Check existing issues first
   - Provide detailed reproduction steps
   - Include error messages and screenshots

2. **Suggest Features**
   - Open an issue with the 'enhancement' label
   - Explain the use case and benefits

3. **Code Contributions**
   ```bash
   # Fork the repository
   git clone https://github.com/yourusername/translater.git
   cd translater
   
   # Create a feature branch
   git checkout -b feature/your-feature
   
   # Make your changes
   # Add tests if applicable
   
   # Run tests
   python -m pytest
   
   # Commit and push
   git commit -am 'Add some feature'
   git push origin feature/your-feature
   ```

4. **Documentation**
   - Update README.md with new features
   - Add docstrings to new functions
   - Create examples and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Python
- Uses [Streamlit](https://streamlit.io/) for the web interface
- Powered by [Hugging Face Transformers](https://huggingface.co/transformers/)
- Translation services: Google, Microsoft, LibreTranslate

## 📚 Resources

- [API Documentation](https://github.com/yourusername/translater/docs)
- [Examples](https://github.com/yourusername/translater/examples)
- [Changelog](https://github.com/yourusername/translater/CHANGELOG.md)
- [Contributing Guidelines](https://github.com/yourusername/translater/CONTRIBUTING.md)

## 📞 Support

For support, please open an issue on [GitHub](https://github.com/yourusername/translater/issues) or email support@example.com

## 🚀 Quick Start

### Web Interface
1. **Launch the application**:
   ```bash
   streamlit run translater.py
   ```

2. **Basic Translation**:
   - Enter text in the input box
   - Select source language (or 'Auto Detect')
   - Choose target language
   - Select translation service (Google, Microsoft, or Libre)
   - Click 'Translate' button

3. **Sentiment Analysis**:
   - View automatic sentiment analysis results below translation
   - Explore detailed emotion breakdown
   - Check for sarcasm and contextual meaning

### Command Line Interface
```bash
# Basic translation
python translater.py --text "Hello, how are you?" --target hi

# Specify source language
python translater.py --text "नमस्ते" --source hi --target en

# Use specific translation service
python translater.py --text "Bonjour" --source fr --target en --service microsoft

# Batch translation from file
python translater.py --file input.txt --target es --output translated.txt

# Enable detailed sentiment analysis
python translater.py --text "I love this product!" --analyze
```

### API Usage
```python
from translater import Translator

# Initialize translator
translator = Translator(service='google')

# Basic translation
result = translator.translate("Hello, world!", target_lang='es')
print(f"Translation: {result['translation']}")
print(f"Detected language: {result['detected_language']}")

# Sentiment analysis
analysis = translator.analyze_sentiment("I'm really excited about this project!")
print(f"Sentiment: {analysis['sentiment']}")
print(f"Emotions: {analysis['emotions']}")

# Batch processing
translations = translator.translate_batch(
    ["Hello", "How are you?", "Goodbye"],
    target_lang='fr'
)
```

## 🛠️ Dependencies

### Core Dependencies
| Package | Version | Description |
|---------|---------|-------------|
| `streamlit` | >=1.24.0 | Web application framework |
| `nltk` | >=3.8.1 | Natural Language Toolkit |
| `spacy` | >=3.5.0 | Advanced NLP processing |
| `torch` | >=2.0.0 | PyTorch for deep learning |
| `transformers` | >=4.26.0 | Transformer models |
| `textblob` | >=0.17.1 | Sentiment analysis |
| `deep_translator` | >=1.10.1 | Translation services |
| `scikit-learn` | >=1.2.0 | Machine learning models |
| `pandas` | >=1.5.0 | Data manipulation |
| `numpy` | >=1.23.0 | Numerical operations |
| `matplotlib` | >=3.6.0 | Basic visualizations |
| `seaborn` | >=0.12.0 | Statistical visualizations |
| `plotly` | >=5.11.0 | Interactive visualizations |
| `fuzzywuzzy` | >=0.18.0 | String matching |
| `python-dotenv` | >=0.21.0 | Environment management |

#### Translation Services
1. **Google Translate**
   - Uses `GoogleTranslator` from `deep_translator`
   - Requires API key for production use

2. **Microsoft Translator**
   - Uses `MicrosoftTranslator` from `deep_translator`
   - Requires Azure subscription and API key

3. **LibreTranslate**
   - Uses `LibreTranslator` from `deep_translator`
   - Can be self-hosted or use public instance

#### Sentiment Analysis Models
1. **Rule-based**
   - VADER (Valence Aware Dictionary and sEntiment Reasoner)
   - TextBlob

2. **Machine Learning**
   - Naive Bayes
   - Support Vector Machines (SVM)
   - Logistic Regression

3. **Deep Learning**
   - BERT-based models
   - RoBERTa
   - IndicBERT (for Indian languages)

#### Language Support
- Comprehensive support for Indian languages including:
  - Hindi (hi)
  - Bengali (bn)
  - Tamil (ta)
  - Telugu (te)
  - Marathi (mr)
  - Gujarati (gu)
  - And many more...

### 🗄️ Data Storage
- Uses in-memory caching for translations
- Implements translation memory system
- Supports saving/loading translation history

### ⚙️ Configuration
- Uses environment variables for API keys and settings
- Configuration can be managed through `.env` file
- Supports different environments (development, production)

---

# 🌐 LinguaBridge

LinguaBridge is an advanced translation and sentiment analysis tool that combines multiple translation engines with powerful natural language processing capabilities. It supports a wide range of languages and provides accurate sentiment analysis using state-of-the-art machine learning models.

> **Note**: This is a Windows-optimized version. For other operating systems, see the [Alternative Installation](#alternative-installation) section.

## ✨ Main Application Features

### 🔤 Translation
- **Multiple Translation Engines**: Google Translate, LibreTranslate, and custom models
- **100+ Languages**: Comprehensive language support including:
  - Major world languages (English, Spanish, French, etc.)
  - 20+ Indian languages (Hindi, Bengali, Tamil, etc.)
  - Support for code-mixed text (e.g., Hinglish, Spanglish)
- **Advanced Features**:
  - Batch processing for multiple texts
  - Translation memory and caching
  - Quality estimation with confidence scores
  - Automatic language detection
  - Back-translation for quality verification

### 😊 Enhanced Sentiment Analysis
- **Multiple Analysis Methods**:
  - Rule-based analysis (VADER, TextBlob)
  - Machine Learning models (Naive Bayes, SVM, Logistic Regression)
  - Transformer models (BERT, RoBERTa, IndicBERT)
  - Ensemble methods for improved accuracy
- **Advanced Features**:
  - Emotion detection and analysis
  - Sarcasm detection
  - Context-aware sentiment analysis
  - Multi-language support including Indic languages
  - Performance optimization with caching

### 📊 Text Processing
- Language Detection
- Text Cleaning and Normalization
- Keyword Extraction
- Text Summarization
- Profanity Filtering

## 🛠️ Setup and Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Git (for cloning the repository)
- Windows 10/11 (recommended) or Windows Server 2016+
- At least 4GB RAM (8GB recommended)
- 2GB free disk space

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/linguabridge.git
cd linguabridge
```

### 2. Create and Activate Virtual Environment (Required)
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows PowerShell)
.\venv\Scripts\Activate.ps1

# If you get execution policy error, run:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# For Command Prompt (cmd.exe):
# .\venv\Scripts\activate.bat

# If you see a permission error, try:
# Set-ExecutionPolicy -ExecutionPolicy Unrestricted -Scope CurrentUser
# Then try activating again
```

### 3. Install Dependencies
```powershell
# First upgrade pip
python -m pip install --upgrade pip

# Install dependencies with specific versions
pip install -r requirements.txt

# If you encounter any errors, try installing with --no-cache-dir
# pip install --no-cache-dir -r requirements.txt

# For GPU support (if you have an NVIDIA GPU)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 4. Set Up Environment Variables
Create a `.env` file in the project root and add the following variables:
```env
# Application Settings
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key-here

# Translation API Keys (optional but recommended for better performance)
GOOGLE_TRANSLATE_API_KEY=your-google-api-key
LIBRE_TRANSLATE_API_KEY=your-libre-api-key

# HuggingFace Token (required for some models)
HF_TOKEN=your-huggingface-token

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log
LOG_MAX_SIZE=10  # MB
LOG_BACKUP_COUNT=5

# Cache Settings
CACHE_DIR=./.cache
CACHE_TTL=86400  # 24 hours in seconds

# Performance Settings
MAX_WORKERS=4  # Adjust based on your CPU cores
BATCH_SIZE=32  # For batch processing
```

### 5. Download Required Language Resources
```powershell
# Download NLTK data (run in Python)
python -c "
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('vader_lexicon')
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
nltk.download('stopwords')
nltk.download('words')"

# Download spaCy models
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm  # For language detection

# Install additional language models (optional but recommended)
pip install spacytextblob
python -m spacy download en_core_web_lg  # Larger model for better accuracy
```

### 6. Verify Installation
```powershell
# Basic Python package check
python -c "
import nltk, spacy, torch, transformers, streamlit
print('✓ Core packages imported successfully')
try:
    nlp = spacy.load('en_core_web_sm')
    print('✓ spaCy model loaded successfully')
except Exception as e:
    print(f'✗ Error loading spaCy model: {e}')

print('\nInstallation verification complete!')
"

# Test Streamlit (should open a browser window)
streamlit hello
```

### 7. Start the Application
```powershell
# Start the Streamlit app
streamlit run translater.py

# For development with auto-reload
# set FLASK_DEBUG=1 && streamlit run translater.py

# If you need to specify a different port (default is 8501)
# streamlit run translater.py --server.port 8080
```
```

## 🚀 Running the Application

### Development Mode (Local)

1. **Start the development server**:
   ```powershell
   # Set up environment variables (PowerShell)
   $env:FLASK_APP = "app.py"
   $env:FLASK_ENV = "development"
   $env:SECRET_KEY = "your-secret-key"
   
   # For better performance, you can also set:
   $env:TOKENIZERS_PARALLELISM = "true"
   $env:PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512"
   
   # Run the Streamlit app
   streamlit run translater.py
   
   # Or run with specific settings
   # streamlit run translater.py --server.port 8080 --server.headless true
   ```

2. **Access the application**:
   Open your browser and navigate to `http://localhost:8501` (default Streamlit port)
   
   If the browser doesn't open automatically, you can manually open the URL shown in the console.
   
   > **Note**: If you see a "Port already in use" error, either stop the other process using that port or specify a different port using `--server.port` flag.

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### 1. Virtual Environment Issues
**Issue**: `'venv' is not recognized as an internal or external command`  
**Solution**:
```powershell
# Make sure Python is in your PATH
python --version  # Should show Python 3.8+

# If not, add Python to PATH or use full path to Python executable
C:\Users\YourUsername\AppData\Local\Programs\Python\Python39\python -m venv venv
```

#### 2. Package Installation Failures
**Issue**: `ERROR: Could not build wheels for...`  
**Solution**:
```powershell
# Install C++ Build Tools (required for some packages)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Then try installing with:
pip install --upgrade pip setuptools wheel
pip install --only-binary :all: -r requirements.txt
```

#### 3. NLTK Data Download Issues
**Issue**: `[SSL: CERTIFICATE_VERIFY_FAILED]`  
**Solution**:
```python
# Run this in Python to disable SSL verification temporarily
import ssl
import nltk

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('all')  # Download all NLTK data
```

#### 4. CUDA/GPU Related Issues
**Issue**: `CUDA out of memory` or `CUDA not available`  
**Solution**:
```powershell
# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If not available, you may need to install the correct PyTorch version
# For CUDA 11.8
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only version
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### 5. Streamlit Connection Issues
**Issue**: `Streamlit is not recognized`  
**Solution**:
```powershell
# Make sure Streamlit is installed in your virtual environment
pip install streamlit

# If you get a permission error, try:
python -m pip install --user streamlit

# Or run with Python module
python -m streamlit run translater.py
```

#### 6. Port Already in Use
**Issue**: `Port 8501 is already in use`  
**Solution**:
```powershell
# Find the process using the port
netstat -ano | findstr :8501

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or simply use a different port
streamlit run translater.py --server.port 8502
```

## 🔄 Alternative Installation Methods

### Using Conda (Recommended for Windows)
```powershell
# Create a new conda environment
conda create -n linguabridge python=3.9
conda activate linguabridge

# Install PyTorch with CUDA (if you have an NVIDIA GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### Using Docker (For Production)
```dockerfile
# Build the Docker image
docker build -t linguabridge .

# Run the container
docker run -p 8501:8501 linguabridge
```

## 📊 Performance Tips

1. **For Better Performance**:
   - Use a GPU for faster inference
   - Increase batch size in settings for bulk operations
   - Enable caching for frequently translated text
   - Use smaller models for faster inference with slightly lower accuracy

2. **Memory Optimization**:
   - Reduce batch size if you encounter memory errors
   - Use `--server.maxUploadSize=1024` to limit upload size
   - Enable garbage collection with `--server.enableCORS=false`

## 📞 Support

For additional help, please open an issue on our [GitHub repository](https://github.com/yourusername/linguabridge/issues) or contact support@example.com.



#### 1. Translate Text
```http
POST /api/translate
Content-Type: application/json

{
  "text": "Hello, how are you?",
  "source_lang": "en",
  "target_lang": "es"
}
```

**Response:**
```json
{
  "translation": "Hola, ¿cómo estás?",
  "source_lang": "en",
  "target_lang": "es",
  "confidence": 0.95,
  "detected_lang": "en"
}
```

#### 2. Analyze Sentiment
```http
POST /api/analyze/sentiment
Content-Type: application/json

{
  "text": "I love this product! It's amazing!",
  "lang": "en"
}
```

**Response:**
```json
{
  "sentiment": "positive",
  "confidence": 0.92,
  "scores": {
    "positive": 0.92,
    "neutral": 0.05,
    "negative": 0.03,
    "compound": 0.89
  },
  "emotions": {
    "joy": 0.85,
    "trust": 0.78,
    "surprise": 0.15
  },
  "key_phrases": ["love this product", "amazing"],
  "language": "en"
}
```

#### 3. Detect Language
```http
POST /api/detect-language
Content-Type: application/json

{
  "text": "Bonjour, comment ça va?"
}
```

**Response:**
```json
{
  "language": "fr",
  "confidence": 0.98,
  "reliable": true
}
```

## 🚀 Deployment

### Docker Compose (Recommended for Production)

1. **Create a `docker-compose.prod.yml` file**:
   ```yaml
   version: '3.8'
   
   services:
     app:
       build: .
       ports:
         - "5000:5000"
       environment:
         - FLASK_APP=app.py
         - FLASK_ENV=production
         - SECRET_KEY=your-secure-secret-key
         - LOG_LEVEL=INFO
       volumes:
         - ./logs:/app/logs
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
   ```

2. **Build and start the production stack**:
   ```bash
   docker-compose -f docker-compose.prod.yml up -d --build
   ```

### Manual Deployment with Gunicorn

1. **Install Gunicorn and production WSGI server**:
   ```bash
   pip install gunicorn gevent
   ```

2. **Run with Gunicorn**:
   ```bash
   gunicorn -k gevent -w 4 -b 0.0.0.0:5000 --access-logfile - --error-logfile - app:app
   ```

   For better performance, use a process manager like systemd or supervisor.

### Kubernetes Deployment

1. **Create a Kubernetes deployment file** (`k8s-deployment.yaml`):
   ```yaml
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: lingua-bridge
     labels:
       app: lingua-bridge
   spec:
     replicas: 3
     selector:
       matchLabels:
         app: lingua-bridge
     template:
       metadata:
         labels:
           app: lingua-bridge
       spec:
         containers:
         - name: lingua-bridge
           image: your-registry/lingua-bridge:latest
           ports:
           - containerPort: 5000
           env:
           - name: FLASK_ENV
             value: "production"
           - name: SECRET_KEY
             valueFrom:
               secretKeyRef:
                 name: lingua-secrets
                 key: secret-key
           resources:
             limits:
               cpu: "1"
               memory: "1Gi"
             requests:
               cpu: "0.5"
               memory: "512Mi"
           livenessProbe:
             httpGet:
               path: /health
               port: 5000
             initialDelaySeconds: 30
             periodSeconds: 10
           readinessProbe:
             httpGet:
               path: /health
               port: 5000
             initialDelaySeconds: 5
             periodSeconds: 5
   ```

## ⚙️ Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FLASK_APP` | No | `app.py` | The main application file |
| `FLASK_ENV` | No | `development` | Set to `production` in production |
| `SECRET_KEY` | Yes | - | Secret key for session management and security |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `LOG_FILE` | No | `app.log` | Path to the log file |
| `DATABASE_URL` | No | `sqlite:///app.db` | Database connection string |
| `CACHE_TYPE` | No | `simple` | Cache type (simple, redis, memcached) |
| `CACHE_REDIS_URL` | If using Redis | - | Redis connection URL |
| `GOOGLE_TRANSLATE_API_KEY` | For Google Translate | - | Google Cloud Translation API key |
| `LIBRE_TRANSLATE_API_KEY` | For LibreTranslate | - | LibreTranslate API key |

### Configuration File

You can also use a configuration file (`config.py`) for more complex setups:

```python
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-123'
    FLASK_ENV = os.environ.get('FLASK_ENV') or 'development'
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or 'sqlite:///app.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Cache
    CACHE_TYPE = os.environ.get('CACHE_TYPE', 'simple')
    CACHE_REDIS_URL = os.environ.get('CACHE_REDIS_URL')
    
    # Translation Services
    GOOGLE_TRANSLATE_API_KEY = os.environ.get('GOOGLE_TRANSLATE_API_KEY')
    LIBRE_TRANSLATE_API_KEY = os.environ.get('LIBRE_TRANSLATE_API_KEY')
    
    # Rate limiting
    RATELIMIT_DEFAULT = "200 per day;50 per hour"
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    FLASK_ENV = 'production'
    PREFERRED_URL_SCHEME = 'https'

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
```

### Logging Configuration

Logs are automatically configured based on the environment. In production, logs are written to both the console and a file:

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
def configure_logging(app):
    # Disable default Flask logger
    app.logger.handlers = []
    
    # Set log level
    log_level = getattr(logging, app.config['LOG_LEVEL'].upper())
    app.logger.setLevel(log_level)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    app.logger.addHandler(console_handler)
    
    # File handler (only in production)
    if app.config['FLASK_ENV'] == 'production':
        file_handler = RotatingFileHandler(
            app.config['LOG_FILE'],
            maxBytes=1024 * 1024 * 10,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        app.logger.addHandler(file_handler)
    
    return app.logger
```

## 🧪 Testing

### Running Tests

1. **Install test dependencies**:
   ```bash
   pip install -r requirements-test.txt
   ```

2. **Run all tests**:
   ```bash
   pytest tests/ -v
   ```

3. **Run tests with coverage**:
   ```bash
   pytest --cov=app --cov-report=term-missing tests/
   ```

4. **Generate HTML coverage report**:
   ```bash
   pytest --cov=app --cov-report=html tests/
   open htmlcov/index.html  # View the report
   ```

### Test Structure

```
tests/
├── conftest.py           # Test configuration and fixtures
├── unit/                # Unit tests
│   ├── test_models.py
│   ├── test_utils.py
│   └── test_services/
├── integration/         # Integration tests
│   ├── test_api.py
│   └── test_auth.py
└── e2e/                 # End-to-end tests
    └── test_user_flows.py
```

## 🛠 Development

### Code Style

This project follows strict code style guidelines. We use:

- **Black** for code formatting
- **isort** for import sorting
- **Flake8** for linting
- **mypy** for static type checking

### Development Setup

1. **Install development dependencies**:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

3. **Run formatters and linters**:
   ```bash
   # Format code with Black
   black .
   
   # Sort imports with isort
   isort .
   
   # Check for style issues
   flake8
   
   # Run static type checking
   mypy .
   ```

### Development Workflow

1. Create a new branch for your feature or bugfix:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Add your commit message"
   ```

3. Push your changes and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

4. After code review, your changes will be merged into the main branch.

### Debugging

1. **Using VS Code**:
   - Set breakpoints in your code
   - Press F5 to start debugging
   - Use the debug console to inspect variables

2. **Using pdb**:
   ```python
   import pdb; pdb.set_trace()  # Add this line where you want to break
   ```

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Report Bugs**: File an issue describing the bug
2. **Suggest Features**: Suggest new features or improvements
3. **Submit Pull Requests**: Fix bugs or add features

### Pull Request Guidelines

1. Fork the repository and create your branch from `main`
2. Ensure your code passes all tests and linters
3. Update the documentation as needed
4. Include tests for new features or bug fixes
5. Keep pull requests focused and small
6. Use descriptive commit messages

### Code Review Process

1. All pull requests require at least one review
2. CI must pass before merging
3. Maintainers will review your code and provide feedback
4. Once approved, your PR will be squashed and merged

### Reporting Issues

When reporting issues, please include:
- A clear description of the problem
- Steps to reproduce the issue
- Expected vs. actual behavior
- Screenshots if applicable
- Version information

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Thanks to all contributors who have helped improve this project
- Built with ❤️ using Python, Flask, and other amazing open-source tools
- Special thanks to the NLTK, spaCy, and Transformers communities

## 🏗️ Project Structure

```
project/
├── static/                 # Static files (CSS, JS, images)
│   └── css/
│       └── style.css      # Main stylesheet
├── templates/              # HTML templates
│   └── index.html         # Main application page
├── .env                   # Environment variables
├── app.py                 # Main Streamlit application
├── config.py              # Application configuration
├── sentiment_analyzer.py  # Sentiment analysis module
└── requirements.txt       # Python dependencies
```

## 🌐 API Endpoints

- `POST /api/translate` - Translate text
  ```json
  {
    "text": "Hello, world!",
    "source_lang": "auto",
    "target_lang": "es"
  }
  ```

- `POST /api/sentiment` - Analyze sentiment
  ```json
  {
    "text": "I'm feeling great today!"
  }
  ```

## 🙏 Acknowledgments

- NLTK & spaCy - For natural language processing
- Font Awesome - For icons
- Google Fonts - For typography