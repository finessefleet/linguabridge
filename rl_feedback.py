import random
import pandas as pd

# Simple feedback storage (can be extended with database or file I/O)
feedback_scores = {"llama": [], "google": []}

def collect_feedback(score, feedback_text="", section=""):
    model = get_best_model()
    feedback_entry = {
        'score': score,
        'feedback': feedback_text,
        'section': section,
        'timestamp': pd.Timestamp.now()
    }
    feedback_scores[model].append(feedback_entry)

def get_best_model(auto_select=False):
    avg_llama = sum(feedback_scores["llama"])/len(feedback_scores["llama"]) if feedback_scores["llama"] else 3
    avg_google = sum(feedback_scores["google"])/len(feedback_scores["google"]) if feedback_scores["google"] else 3

    if auto_select:
        return "llama" if avg_llama >= avg_google else "google"
    else:
        return random.choice(["llama", "google"])
