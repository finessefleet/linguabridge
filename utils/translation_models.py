import os
import torch
from functools import lru_cache
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    MBart50TokenizerFast,
    MBartForConditionalGeneration,
    MarianMTModel,
    MarianTokenizer
)

# ---------------------------------------------
# Shared Translation Utility
# ---------------------------------------------
def translate(text, tokenizer, model, tgt_lang_token=None):
    if tgt_lang_token:
        tokenizer.src_lang = tgt_lang_token
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    output_tokens = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]

# ---------------------------------------------
# Hinglish → Hindi
# ---------------------------------------------
# ---------------------------------------------
# Hinglish → Hindi (IndicTrans2 Indic-Indic)
# ---------------------------------------------
@lru_cache()
def load_hinglish_to_hindi():
    tokenizer = AutoTokenizer.from_pretrained(
        "ai4bharat/IndicTrans2-indic-indic-1B",
        trust_remote_code=True
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/IndicTrans2-indic-indic-1B",
        trust_remote_code=True
    )
    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")

def translate_hinglish_to_hindi(text):
    tokenizer, model = load_hinglish_to_hindi()
    input_text = f"hin_Deva hin_Deva {text}"  # Source and target = Hindi
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)


# ---------------------------------------------
# Bengalish → Bengali Script (Transliteration)
# ---------------------------------------------
@lru_cache()
def load_bengalish_to_bengali():
    model_name = "shadabtanjeed/mbart-banglish-to-bengali-transliteration"
    token = os.getenv("HF_TOKEN")

    tokenizer = MBart50TokenizerFast.from_pretrained(model_name, use_auth_token=token)
    model = MBartForConditionalGeneration.from_pretrained(model_name, use_auth_token=token)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    tokenizer.src_lang = "en_XX"  # Source language for MBART
    return tokenizer, model, device

def transliterate_bengalish_to_bengali_script(text: str, max_length: int = 128) -> str:
    tokenizer, model, device = load_bengalish_to_bengali()

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    generated_ids = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# ---------------------------------------------
# Bengali → English
# ---------------------------------------------
@lru_cache()
def load_bengali_to_english():
    tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
    model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-bn-en")
    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")

def translate_bengali_to_english(text):
    tokenizer, model = load_bengali_to_english()
    return translate(text, tokenizer, model)

# ---------------------------------------------
# Tanglish → Tamil
# ---------------------------------------------
# -------------------------------
# Tanglish → Tamil (T_TL Model)
# -------------------------------
@lru_cache()
def load_tanglish_to_tamil():
    tokenizer = AutoTokenizer.from_pretrained("gowtham58/T_TL")
    model = AutoModelForSeq2SeqLM.from_pretrained("gowtham58/T_TL")
    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")

def translate_tanglish_to_tamil(text: str) -> str:
    tokenizer, model = load_tanglish_to_tamil()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(model.device)
    outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------
# Telugulish → Telugu
# ---------------------------------------------
# -------------------------------
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def translate_telugulish_to_telugu(text: str) -> str:
    """
    Converts Telugulish (Romanized Telugu) to native Telugu script.
    """
    return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)

# ---------------------------------------------
# Odialish → Odia
# ---------------------------------------------
from indic_transliteration.sanscript import transliterate

def translate_odia_to_odia(text: str) -> str:
    """
    Converts Odialish (Romanized Odia) to native Odia script using ITRANS.
    Example: "Tume kebe asiba" → "ତୁମେ କେବେ ଆସିବ"
    """
    if not text:
        return ""
    return transliterate(text, "itrans", "oriya")


# ---------------------------------------------
# Urdlish → Urdu
# ---------------------------------------------
@lru_cache()
def load_urdlish_to_urdu():
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
    model = AutoModelForSeq2SeqLM.from_pretrained("ai4bharat/indictrans2-en-indic-1B")
    return tokenizer, model.to("cuda" if torch.cuda.is_available() else "cpu")

def translate_urdlish_to_urdu(text):
    tokenizer, model = load_urdlish_to_urdu()
    return translate(text, tokenizer, model, tgt_lang_token="ur_Arabic")