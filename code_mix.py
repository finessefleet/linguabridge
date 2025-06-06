# Add these imports at the top of the file

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import evaluate
import pandas as pd
import random
from tqdm import tqdm
import json
import os
import logging

# Set up logging
logger = logging.getLogger(__name__)

class EnhancedCodeMixTranslator(object):
    def __init__(self):
        super().__init__()
        # Additional metrics
        self.rouge = Rouge()
        self.bertscore = evaluate.load("bertscore")
        self.bleurt = evaluate.load("bleurt", "bleurt-large-512")
        self.comet_model = None
        self.other_models = {}
        self.code_mix_patterns = {}
        self._load_code_mix_patterns()  # Load code-mix patterns on init
        
        # Language mapping for code-mixing
        self.lang_mapping = {
            'hi': 'Hinglish',
            'bn': 'Banglish',
            'ta': 'Tanglish',
            'te': 'Tanglish',
            'kn': 'Kanglish',
            'ml': 'Manglish',
            'mr': 'Marlish',
            'gu': 'Gujlish',
            'pa': 'Punglish',
            'or': 'Odiya English',
            'as': 'Assamlish',
            'mni': 'Manipuri English',
            'kok': 'Konkani English'
        }
        
    def load_comet_model(self):
        """Load COMET model for quality estimation"""
        try:
            from comet import download_model, load_from_checkpoint
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
        except ImportError:
            logger.warning("COMET not installed. Install with: pip install unbabel-comet")
            self.comet_model = None
    
    def load_other_models(self):
        """Load other translation models for comparison"""
        try:
            # Load OPUS-MT models for comparison
            logger.info("Loading comparison models...")
            self.other_models['opus-mt-hi-en'] = pipeline(
                "translation_hi_to_en", 
                model="Helsinki-NLP/opus-mt-hi-en"
            )
            self.other_models['m2m100'] = pipeline(
                "translation", 
                model="facebook/m2m100_418M"
            )
            logger.info("Loaded comparison models")
        except Exception as e:
            logger.warning(f"Could not load all comparison models: {e}")
    
    def calculate_bleu(self, reference, hypothesis):
        """Calculate BLEU score with smoothing and improved tokenization"""
        try:
            # Import NLTK and download necessary data
            import nltk
            try:
                nltk.data.find('punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Tokenize the reference and hypothesis
            ref_tokens = [nltk.word_tokenize(reference.lower())]
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            
            # Calculate BLEU score with smoothing
            smoothie = SmoothingFunction().method4
            score = sentence_bleu(
                ref_tokens,
                hyp_tokens,
                smoothing_function=smoothie,
                weights=(0.4, 0.3, 0.2, 0.1)  # Higher weight for unigrams and bigrams
            )
            return min(float(score) * 100, 100.0)  # Scale to 0-100 range and cap at 100
        except Exception as e:
            logger.warning(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def calculate_meteor(self, reference, hypothesis):
        """Calculate METEOR score with improved tokenization"""
        try:
            # Import NLTK and download necessary data
            import nltk
            try:
                nltk.data.find('wordnet')
                nltk.data.find('omw-1.4')
            except LookupError:
                nltk.download('wordnet')
                nltk.download('omw-1.4')
            
            # Tokenize the reference and hypothesis
            ref_tokens = nltk.word_tokenize(reference.lower())
            hyp_tokens = nltk.word_tokenize(hypothesis.lower())
            
            # Calculate METEOR score with alpha=0.9 to give more weight to recall
            score = meteor_score(
                [ref_tokens],
                hyp_tokens,
                alpha=0.9,
                gamma=0.3,
                beta=3.0
            )
            return min(float(score) * 100, 100.0)  # Scale to 0-100 range and cap at 100
        except Exception as e:
            logger.warning(f"Error calculating METEOR score: {e}")
            return 0.0
            
    def calculate_rouge(self, reference, hypotheses):
        """Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) with improved handling"""
        try:
            # Import NLTK for better tokenization
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize
            
            # Convert single reference/hypothesis to list if needed
            if isinstance(reference, str):
                reference = [reference]
            if isinstance(hypotheses, str):
                hypotheses = [hypotheses]
            
            # Preprocess text (lowercase, remove extra whitespace)
            reference = [' '.join(word_tokenize(ref.lower().strip())) for ref in reference]
            hypotheses = [' '.join(word_tokenize(hyp.lower().strip())) for hyp in hypotheses]
            
            # Calculate ROUGE scores with alpha=0.5 (balancing precision and recall)
            scores = self.rouge.get_scores(
                hyps=hypotheses,
                refs=reference,
                avg=True,
                ignore_empty=True
            )
            
            # Extract and scale the scores to 0-100 range
            return {
                'rouge-1': min(scores['rouge-1']['f'] * 100, 100.0),
                'rouge-2': min(scores['rouge-2']['f'] * 100, 100.0),
                'rouge-l': min(scores['rouge-l']['f'] * 100, 100.0)
            }
        except Exception as e:
            logger.warning(f"Error calculating ROUGE scores: {e}")
            return {
                'rouge-1': 0.0,
                'rouge-2': 0.0,
                'rouge-l': 0.0
            }
    
    def calculate_bertscore(self, references, hypotheses):
        """Calculate BERTScore"""
        return self.bertscore.compute(
            predictions=hypotheses,
            references=references,
            lang="en"  # or detect language
        )
    
    def calculate_bleurt(self, references, hypotheses):
        """Calculate BLEURT score"""
        try:
            return self.bleurt.compute(
                predictions=hypotheses,
                references=references
            )
        except:
            return {"scores": [0] * len(references)}
    
    def train(self, X1, X2, y, X1_val, X2_val, y_val, epochs=20, batch_size=64):
        """Train the model with callbacks and learning rate scheduling"""
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import f1_score
        
        logger.info("Training model...")
        
        # Initialize metrics_history if it doesn't exist
        if not hasattr(self, 'metrics_history'):
            self.metrics_history = {'train_f1': [], 'val_f1': []}
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-6
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_model.h5',
                save_best_only=True,
                monitor='val_loss',
                mode='min'
            )
        ]
        
        # Convert data to numpy arrays if they're not already
        X1 = np.array(X1)
        X2 = np.array(X2)
        y = np.array(y).reshape(-1, 1)  # Ensure y is 2D
        X1_val = np.array(X1_val)
        X2_val = np.array(X2_val)
        y_val = np.array(y_val).reshape(-1, 1)  # Ensure y_val is 2D
        
        try:
            # Train the model
            history = self.model.fit(
                [X1, X2], 
                y,
                validation_data=([X1_val, X2_val], y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
            # Calculate predictions and metrics
            y_train_pred = (self.model.predict([X1, X2], verbose=0) > 0.5).astype(int)
            y_val_pred = (self.model.predict([X1_val, X2_val], verbose=0) > 0.5).astype(int)
            train_f1 = f1_score(y.flatten(), y_train_pred.flatten())
            val_f1 = f1_score(y_val.flatten(), y_val_pred.flatten())
            
            # Update metrics history
            self.metrics_history['train_f1'].append(train_f1)
            self.metrics_history['val_f1'].append(val_f1)
            
            logger.info(f"Training complete. Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
            logger.info(f"Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")
            
            return history
            
        except Exception as e:
            logger.error(f"Error during training: {str(e)}")
            raise
        
    def calculate_comet(self, sources, hypotheses, references):
        """Calculate COMET score for quality estimation"""
        if self.comet_model is None:
            return {"mean_score": 0, "scores": [0] * len(sources)}
            
        data = [{
            "src": src,
            "mt": mt,
            "ref": ref
        } for src, mt, ref in zip(sources, hypotheses, references)]
        
        try:
            scores = self.comet_model.predict(data, batch_size=8, gpus=1)
            return {
                "mean_score": float(scores.mean()),
                "scores": scores.tolist()
            }
        except:
            return {"mean_score": 0, "scores": [0] * len(sources)}
    
    def compare_with_other_models(self, texts, source_lang, target_lang='en'):
        """Compare translations with other models"""
        results = []
        
        for text in tqdm(texts, desc="Comparing with other models"):
            # Get our translation
            our_result = self.translate_and_evaluate(
                text, 
                target_lang=target_lang,
                source_lang=source_lang
            )
            
            # Get translations from other models
            model_translations = {}
            for model_name, model in self.other_models.items():
                try:
                    if 'hi' in model_name and source_lang == 'hi':
                        # Special handling for Hindi models
                        translation = model(text, max_length=400)[0]['translation_text']
                    else:
                        # For multilingual models
                        translation = model(text, 
                                         src_lang=source_lang,
                                         tgt_lang=target_lang,
                                         max_length=400)[0]['translation_text']
                    model_translations[model_name] = translation
                except Exception as e:
                    logger.warning(f"Error with {model_name}: {e}")
                    model_translations[model_name] = ""
            
            # Calculate metrics for each model
            model_results = {}
            for model_name, translation in model_translations.items():
                if not translation:
                    model_results[model_name] = {
                        'bleu': 0,
                        'meteor': 0,
                        'rouge': 0,
                        'translation': ""
                    }
                    continue
                    
                # Calculate automatic metrics
                bleu = self.calculate_bleu(our_result['translation'], translation)
                meteor = self.calculate_meteor(our_result['translation'], translation)
                rouge = self.calculate_rouge(our_result['translation'], translation)
                
                model_results[model_name] = {
                    'bleu': bleu,
                    'meteor': meteor,
                    'rouge_l': rouge['rouge-l']['f'],
                    'translation': translation
                }
            
            results.append({
                'source': text,
                'our_translation': our_result['translation'],
                'model_comparisons': model_results
            })
        
        return results
    
    def human_evaluation(self, texts, output_file='human_evaluation.csv', num_samples=10):
        """Generate a CSV for human evaluation"""
        if len(texts) > num_samples:
            texts = random.sample(texts, num_samples)
        
        results = []
        for text in tqdm(texts, desc="Preparing human evaluation"):
            # Get our translation
            our_result = self.translate_and_evaluate(text, target_lang='en')
            
            # Get translations from other models
            model_translations = {}
            for model_name, model in self.other_models.items():
                try:
                    translation = model(text, max_length=400)[0]['translation_text']
                    model_translations[model_name] = translation
                except:
                    model_translations[model_name] = ""
            
            # Prepare row for CSV
            row = {
                'source_text': text,
                'our_translation': our_result['translation'],
                'our_quality_score': our_result['quality_score']
            }
            
            # Add other model translations
            for model_name, translation in model_translations.items():
                row[f"{model_name}_translation"] = translation
            
            results.append(row)
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        logger.info(f"Saved human evaluation file to {output_file}")
        return df
    
    def _load_code_mix_patterns(self):
        """Load code-mixing patterns from the CSV file"""
        try:
            import pandas as pd
            import os
            
            # Get the directory of the current script
            current_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(current_dir, 'code_mix_cleaned.csv')
            
            if not os.path.exists(csv_path):
                logger.error(f"Code-mix CSV file not found at: {csv_path}")
                return False
                
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Ensure required columns exist
            if not all(col in df.columns for col in ['language', 'code_mixed', 'english']):
                logger.error("CSV file is missing required columns: 'language', 'code_mixed', 'english'")
                return False
            
            # Group by language and create a dictionary of patterns
            for _, row in df.iterrows():
                try:
                    lang = str(row['language']).strip()
                    mixed = str(row['code_mixed']).strip()
                    english = str(row['english']).strip().lower()
                    
                    if not lang or not mixed or not english:
                        continue
                        
                    if lang not in self.code_mix_patterns:
                        self.code_mix_patterns[lang] = {}
                    
                    # Store both the code-mixed text and the language
                    self.code_mix_patterns[lang][english] = {
                        'mixed': mixed,
                        'language': lang
                    }
                    
                except Exception as e:
                    logger.warning(f"Error processing row {_}: {e}")
                    continue
                
            logger.info(f"Loaded code-mix patterns for {len(self.code_mix_patterns)} languages")
            return True
            
        except Exception as e:
            logger.error(f"Error loading code-mix patterns: {str(e)}")
            return False
            
    def _get_code_mix_variant(self, lang_code, english_text):
        """Get the code-mixed variant for the given English text"""
        if not self.code_mix_patterns:
            if not self._load_code_mix_patterns():
                return None
        
        # Clean the input text
        english_text = str(english_text).lower().strip()
        if not english_text:
            return None
        
        # First try exact language match
        if lang_code in self.lang_mapping:
            lang_name = self.lang_mapping[lang_code]
            if lang_name in self.code_mix_patterns:
                # Try exact match first
                for eng_phrase, data in self.code_mix_patterns[lang_name].items():
                    if eng_phrase.lower() == english_text:
                        return data['mixed']
                
                # Try partial match
                for eng_phrase, data in self.code_mix_patterns[lang_name].items():
                    if eng_phrase in english_text or english_text in eng_phrase:
                        return data['mixed']
        
        # If no match in specified language, try other languages
        for lang_name, patterns in self.code_mix_patterns.items():
            for eng_phrase, data in patterns.items():
                if eng_phrase.lower() == english_text:
                    return data['mixed']
        
        # Try partial match in any language
        for lang_name, patterns in self.code_mix_patterns.items():
            for eng_phrase, data in patterns.items():
                if eng_phrase in english_text or english_text in eng_phrase:
                    return data['mixed']
                    
        return None
        
    def apply_code_mixing(self, text, lang_code):
        """
        Apply code-mixing to the given text based on learned patterns.
        
        Args:
            text (str): The input English text to be code-mixed
            lang_code (str): The target language code (e.g., 'hi' for Hindi, 'bn' for Bengali)
            
        Returns:
            str: The code-mixed text if a match is found, otherwise the original text
        """
        if not text or not isinstance(text, str):
            return text
            
        # Try to find an exact match for the entire text first
        mixed = self._get_code_mix_variant(lang_code, text)
        if mixed:
            return mixed
            
        # If no exact match, try to mix at the sentence level
        try:
            import nltk
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
                
            sentences = nltk.sent_tokenize(text)
            mixed_sentences = []
            
            for sent in sentences:
                # Skip very short sentences
                if len(sent.split()) < 2:
                    mixed_sentences.append(sent)
                    continue
                    
                # Try to find a pattern for this sentence
                mixed_sent = self._get_code_mix_variant(lang_code, sent)
                if mixed_sent:
                    mixed_sentences.append(mixed_sent)
                else:
                    # Try to mix individual phrases
                    words = sent.split()
                    mixed_words = []
                    i = 0
                    
                    while i < len(words):
                        # Try to find the longest matching phrase (up to 5 words)
                        matched = False
                        for length in range(min(5, len(words) - i), 0, -1):
                            phrase = ' '.join(words[i:i+length])
                            mixed_phrase = self._get_code_mix_variant(lang_code, phrase)
                            
                            if mixed_phrase:
                                mixed_words.append(mixed_phrase)
                                i += length
                                matched = True
                                break
                                
                        if not matched:
                            # If no pattern matches, keep the original word
                            mixed_words.append(words[i])
                            i += 1
                            
                    mixed_sentences.append(' '.join(mixed_words))
                    
            return ' '.join(mixed_sentences)
            
        except Exception as e:
            logger.error(f"Error in apply_code_mixing: {str(e)}")
            return text  # Return original text if any error occurs
        
    def evaluate_translation_quality(self, source_text, translated_text, source_lang='en', target_lang='hi'):
        """
        Enhanced translation quality evaluation with multiple metrics.
        
        Args:
            source_text (str): Source text in the source language
            translated_text (str): Translated text to evaluate
            source_lang (str): Source language code (default: 'en')
            target_lang (str): Target language code (default: 'hi')
            
        Returns:
            dict: Dictionary containing various quality metrics
        """
        try:
            # Initialize metrics with default values
            metrics = {
                'base_score': 0.0,
                'bleu': 0.0,
                'meteor': 0.0,
                'rouge': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0},
            }
            
            # Import NLTK and download necessary data
            import nltk
            try:
                nltk.data.find('punkt')
                nltk.data.find('wordnet')
                nltk.data.find('omw-1.4')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            
            # Tokenize the source and translated texts
            source_tokens = nltk.word_tokenize(source_text.lower())
            translated_tokens = nltk.word_tokenize(translated_text.lower())
            
            # Calculate BLEU score with smoothing
            try:
                smoothie = SmoothingFunction().method4
                metrics['bleu'] = sentence_bleu(
                    [source_tokens],
                    translated_tokens,
                    smoothing_function=smoothie,
                    weights=(0.5, 0.3, 0.2)  # Focus on unigrams, bigrams, and trigrams
                )
            except Exception as e:
                logger.warning(f"BLEU calculation warning: {str(e)}")
            
            # Calculate METEOR score with language-specific parameters
            try:
                metrics['meteor'] = meteor_score(
                    [source_tokens],
                    translated_tokens,
                    alpha=0.9,  # Controls unigram precision vs. recall
                    gamma=0.3,  # Controls fragmentation penalty
                    beta=3.0    # Controls fragmentation penalty
                )
            except Exception as e:
                logger.warning(f"METEOR calculation warning: {str(e)}")
            
            # Calculate ROUGE scores with improved parameters
            try:
                rouge = Rouge(
                    metrics=['rouge-1', 'rouge-2', 'rouge-l'],
                    stats=['f'],
                    max_n=2,
                    limit_length=False,
                    length_limit=1000,
                    length_limit_type='words',
                    apply_avg=True,
                    apply_best=False,
                    alpha=0.5,  # Alpha for ROUGE: 0.5 balances precision and recall
                    weight_factor=1.2,
                    stemming=True
                )
                
                # Convert token lists back to strings for ROUGE
                source_str = ' '.join(source_tokens)
                translated_str = ' '.join(translated_tokens)
                
                # Get ROUGE scores
                scores = rouge.get_scores(translated_str, source_str)[0]
                
                # Extract F1 scores (harmonic mean of precision and recall)
                metrics['rouge'] = {
                    'rouge-1': scores['rouge-1']['f'],
                    'rouge-2': scores['rouge-2']['f'],
                    'rouge-l': scores['rouge-l']['f']
                }
            except Exception as e:
                logger.warning(f"ROUGE calculation warning: {str(e)}")
            
            # Calculate a weighted base score (favoring METEOR and ROUGE-L)
            weights = {
                'bleu': 0.2,
                'meteor': 0.3,
                'rouge-1': 0.1,
                'rouge-2': 0.1,
                'rouge-l': 0.3
            }
            
            metrics['base_score'] = (
                metrics['bleu'] * weights['bleu'] +
                metrics['meteor'] * weights['meteor'] +
                metrics['rouge']['rouge-1'] * weights['rouge-1'] +
                metrics['rouge']['rouge-2'] * weights['rouge-2'] +
                metrics['rouge']['rouge-l'] * weights['rouge-l']
            )
            
            # Ensure scores are within [0, 1] range
            metrics['base_score'] = max(0.0, min(1.0, metrics['base_score']))
            metrics['bleu'] = max(0.0, min(1.0, metrics['bleu']))
            metrics['meteor'] = max(0.0, min(1.0, metrics['meteor']))
            
            # Scale metrics to 0-100 range for better readability
            for key in ['bleu', 'meteor']:
                if key in metrics:
                    metrics[key] = round(metrics[key] * 100, 2)
            
            for key in metrics['rouge']:
                metrics['rouge'][key] = round(metrics['rouge'][key] * 100, 2)
            
            metrics['base_score'] = round(metrics['base_score'] * 100, 2)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluate_translation_quality: {str(e)}")
            # Return default metrics with error flag
            return {
                'error': str(e),
                'base_score': 0.0,
                'bleu': 0.0,
                'meteor': 0.0,
                'rouge': {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0},
            }

def print_example(title, english_text, lang_code, translator):
    """Helper function to print code-mixing examples with formatting"""
    print(f"\n{'='*50}")
    print(f"{title} (Language Code: {lang_code})")
    print("-"*50)
    print(f"English:    {english_text}")
    mixed_text = translator.apply_code_mixing(english_text, lang_code)
    print(f"Code-Mixed: {mixed_text}")


def main():
    # Initialize the enhanced translator
    print("Initializing EnhancedCodeMixTranslator...")
    translator = EnhancedCodeMixTranslator()
    
    # Load additional models (commented out for faster testing)
    # print("Loading COMET model...")
    # translator.load_comet_model()
    # print("Loading other models for comparison...")
    # translator.load_other_models()
    
    # Test code-mixing with various languages and examples
    examples = [
        # Hinglish (Hindi-English)
        ("What is your name?", 'hi', "Hinglish (Hindi-English) Basic Greeting"),
        ("How are you?", 'hi', "Hinglish Common Phrase"),
        ("I am going to the market.", 'hi', "Hinglish Daily Activity"),
        
        # Banglish (Bengali-English)
        ("What are you doing?", 'bn', "Banglish (Bengali-English) Common Question"),
        ("I am very happy today.", 'bn', "Banglish Expression of Emotion"),
        ("Let's go out for dinner.", 'bn', "Banglish Social Invitation"),
        
        # Tanglish (Tamil-English)
        ("What time is it?", 'ta', "Tanglish (Tamil-English) Time Inquiry"),
        ("I don't understand.", 'ta', "Tanglish Common Phrase"),
        ("Please wait a moment.", 'ta', "Tanglish Polite Request"),
        
        # Bihlish (Bhojpuri-English)
        ("Where are you going?", 'bh', "Bihlish (Bhojpuri-English) Common Question"),
        ("I am very tired.", 'bh', "Bihlish Expression of State"),
        
        # Manglish (Malay-English)
        ("Have you eaten?", 'ms', "Manglish (Malay-English) Common Greeting"),
        ("I want to go home.", 'ms', "Manglish Expression of Desire"),
    ]
    
    print("\n" + "="*60)
    print("CODE-MIXING DEMONSTRATION".center(60))
    print("="*60)
    
    # Display all examples
    for i, (text, lang_code, title) in enumerate(examples, 1):
        print_example(f"Example {i}: {title}", text, lang_code, translator)
    
    # Test translation quality evaluation
    print("\n" + "="*60)
    print("TRANSLATION QUALITY EVALUATION".center(60))
    print("="*60)
    
    test_cases = [
        (
            "This is a test sentence for translation quality evaluation.",
            "यह अनुवाद गुणवत्ता मूल्यांकन के लिए एक परीक्षण वाक्य है।",
            "en",
            "hi",
            "English to Hindi Translation"
        ),
        (
            "The quick brown fox jumps over the lazy dog.",
            "तेज भूरी लोमड़ी आलसी कुत्ते के ऊपर कूदती है।",
            "en",
            "hi",
            "Common English Proverb to Hindi"
        )
    ]
    
    for source, translated, src_lang, tgt_lang, desc in test_cases:
        print(f"\n{desc}:")
        print("-" * len(desc))
        print(f"Source:     {source}")
        print(f"Translated: {translated}")
        
        metrics = translator.evaluate_translation_quality(
            source_text=source,
            translated_text=translated,
            source_lang=src_lang,
            target_lang=tgt_lang
        )
        
        print("\nQuality Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, dict):
                print(f"  {metric}:")
                for k, v in value.items():
                    print(f"    {k}: {v:.4f}" if isinstance(v, (int, float)) else f"    {k}: {v}")
            else:
                print(f"  {metric}: {value:.4f}" if isinstance(value, (int, float)) else f"  {metric}: {value}")
    
    print("\n" + "="*60)
    print("DEMO COMPLETED SUCCESSFULLY".center(60))
    print("="*60)

if __name__ == "__main__":
    main()