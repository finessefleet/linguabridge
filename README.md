# 🌐 Linguabridge

**Linguabridge** is an advanced multilingual code-mixed language translation and sentiment analysis tool built with Streamlit. It uses deep learning, rule-based NLP, and reinforcement learning to deliver highly accurate translations and sentiment analysis, especially for Indic languages and code-mixed texts.


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

---

## 🚀 Features

* 🔤 **Code-Mixed Language Translation**
  Translate Hinglish, Bengalish, Tanglish, Telugulish, Odialish, Urdlish and more to their respective native scripts and languages.

* 💬 **Multilingual Sentiment Analysis**
  Analyze sentiment in over 20+ Indian and international languages using:

  * TextBlob
  * VADER
  * SentiWordNet
  * BERT/IndicBERT models

* 🤖 **Reinforcement Learning Feedback System**
  Continuously improve translation quality based on user feedback with an RL-based model.

* 📊 **Visualization & Insights**
  Real-time graphs, word clouds, confusion matrices, and model performance indicators.

---

## 🛠️ Tech Stack

| Component        | Technology                    |
| ---------------- | ----------------------------- |
| Frontend         | Streamlit                     |
| Backend          | Python, PyTorch, Transformers |
| Translation      | HuggingFace Transformers      |
| Sentiment Models | TextBlob, VADER, IndicBERT    |
| Feedback Loop    | Custom RL-based PyTorch Model |
| Visualization    | Seaborn, Plotly, WordCloud    |

---

## 🧩 Modules Overview

| File                    | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `translater.py`         | Main Streamlit UI integrating all components                                      |
| `section.py`            | UI section for selecting translation type                                         |
| `translation_module.py` | All translation functions using IndicTrans2, MBART, MarianMT, and transliteration |
| `rl_feedback.py`        | Reinforcement learning feedback model for translation quality improvement         |

---

## 🗺️ Supported Language Pairs

* **Hinglish ➝ Hindi**
* **Bengalish ➝ Bengali**
* **Bengali ➝ English**
* **Tanglish ➝ Tamil**
* **Telugulish ➝ Telugu**
* **Odialish ➝ Odia**
* **Urdlish ➝ Urdu**

---

## 📥 Installation

1. **Clone the repository**

```bash
git clone https://github.com/finessefleet/linguabridge.git
cd linguabridge
```

2. **Install requirements**

```bash
pip install -r requirements.txt
```

3. **Set your Hugging Face token (if needed)**

```bash
export HF_TOKEN=your_hf_token_here
```

4. **Run the application**

```bash
streamlit run translater.py
```

---

## 📊 Feedback Model

The RL-based feedback system improves over time:

* Accepts user ratings (`good`, `needs_work`, `incorrect`)
* Computes feature embeddings (e.g., length ratio, code-mix flags)
* Trains a PyTorch model to predict translation quality
* Model auto-updates every few minutes based on feedback frequency

---

## 🧠 Models Used

* `ai4bharat/IndicTrans2-*` (translation)
* `shadabtanjeed/mbart-banglish-to-bengali-transliteration`
* `Helsinki-NLP/opus-mt-bn-en`
* `gowtham58/T_TL` for Tanglish
* `TextBlob`, `VADER`, `spaCy`, `IndicBERT` for sentiment

---

## 🧪 Example Use

* Enter a code-mixed sentence like:

  ```
  "mujhe ghar jana hai"
  ```
* Choose "Hinglish ➝ Hindi"
* Output: `मुझे घर जाना है`

---

## 🛡️ License
Got the license from FinesseFleet Foundation
