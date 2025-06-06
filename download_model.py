from transformers import AutoModelForSequenceClassification, AutoTokenizer, BertForSequenceClassification
import os

def download_model():
    print("Downloading Indic-BERT model...")
    
    # Create model directory if it doesn't exist
    model_path = os.path.join('models', 'indic-bert')
    os.makedirs(model_path, exist_ok=True)
    
    try:
        # Download model and tokenizer
        model_name = "ai4bharat/indic-bert"
        model = BertForSequenceClassification.from_pretrained(model_name, num_labels=14)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Save model and tokenizer
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        
        print("Model downloaded and saved successfully!")
        return True
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    download_model() 