"""
Reinforcement Learning-based feedback system for translation quality improvement.
"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import pytz
import pickle
from collections import defaultdict, deque
import random
import logging
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Constants
FEEDBACK_MODEL_PATH = "feedback_model.pth"
FEEDBACK_DATA_PATH = "feedback_data.pkl"
MODEL_UPDATE_INTERVAL = 10  # Update model every N feedbacks
MAX_FEEDBACK_HISTORY = 1000  # Maximum number of feedback entries to keep

class FeedbackDataset:
    """Dataset for storing and managing translation feedback data."""
    def __init__(self, max_size=MAX_FEEDBACK_HISTORY):
        self.data = deque(maxlen=max_size)
        self.max_size = max_size
    
    def add_feedback(self, source_text: str, translation: str, feedback_type: str, 
                     source_lang: str, target_lang: str, model_used: str, 
                     confidence: float, quality: float) -> Dict[str, Any]:
        """Add a new feedback entry to the dataset."""
        feedback = {
            'timestamp': datetime.now(pytz.utc).isoformat(),
            'source_text': source_text,
            'translation': translation,
            'feedback_type': feedback_type,
            'source_lang': source_lang,
            'target_lang': target_lang,
            'model_used': model_used,
            'confidence': confidence,
            'quality': quality,
            'features': self._extract_features(source_text, translation, source_lang, target_lang)
        }
        self.data.append(feedback)
        return feedback
    
    def _extract_features(self, source_text: str, translation: str, 
                          source_lang: str, target_lang: str) -> Dict[str, float]:
        """Extract features from the translation pair."""
        source_words = source_text.split()
        target_words = translation.split()
        source_length = len(source_words)
        target_length = len(target_words)
        length_ratio = target_length / (source_length + 1e-6)
        
        # Simple character-level similarity
        similarity = self._calculate_similarity(source_text.lower(), translation.lower())
        
        return {
            'source_length': source_length,
            'target_length': target_length,
            'length_ratio': length_ratio,
            'similarity': similarity,
            'is_code_mix_source': source_lang.startswith('cm-'),
            'is_code_mix_target': target_lang.startswith('cm-')
        }
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts."""
        if not text1 or not text2:
            return 0.0
            
        # Simple character n-gram similarity
        n = 3
        ngrams1 = set(text1[i:i+n] for i in range(len(text1)-n+1))
        ngrams2 = set(text2[i:i+n] for i in range(len(text2)-n+1))
        
        if not ngrams1 and not ngrams2:
            return 0.0
            
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_batch(self, batch_size=32) -> Optional[List[Dict[str, Any]]]:
        """Get a batch of feedback data for training."""
        if len(self.data) == 0:
            return None
            
        batch = random.sample(self.data, min(batch_size, len(self.data)))
        return batch
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]
    
    def save(self, path: str) -> None:
        """Save the dataset to disk."""
        with open(path, 'wb') as f:
            pickle.dump(list(self.data), f)
    
    @classmethod
    def load(cls, path: str, max_size: int = MAX_FEEDBACK_HISTORY) -> 'FeedbackDataset':
        """Load the dataset from disk."""
        dataset = cls(max_size=max_size)
        try:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    dataset.data = deque(data, maxlen=max_size)
                logger.info(f"Loaded {len(dataset)} feedback entries from {path}")
            else:
                logger.info(f"No existing feedback data found at {path}, starting with empty dataset")
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
        return dataset

class TranslationQualityModel(nn.Module):
    """Neural network model for predicting translation quality."""
    def __init__(self, input_size: int = 6, hidden_size: int = 32, output_size: int = 1):
        super(TranslationQualityModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Output between 0 and 1
        return x

class RLTranslationModel:
    """Reinforcement learning model for improving translations based on feedback."""
    def __init__(self, feedback_dataset: FeedbackDataset):
        self.feedback_dataset = feedback_dataset
        self.quality_model = TranslationQualityModel().to(device)
        self.optimizer = optim.Adam(self.quality_model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        self.last_update_time = time.time()
        self.update_interval = 300  # Update model every 5 minutes
        self.batch_size = 32
        
    def predict_quality(self, source_text: str, translation: str, 
                        source_lang: str, target_lang: str) -> float:
        """Predict the quality of a translation."""
        if len(self.feedback_dataset) < 10:  # Not enough data for meaningful prediction
            return 0.8  # Default confidence
            
        features = self.feedback_dataset._extract_features(
            source_text, translation, source_lang, target_lang)
        
        # Convert features to tensor
        feature_tensor = torch.tensor([
            features['source_length'],
            features['target_length'],
            features['length_ratio'],
            features['similarity'],
            float(features['is_code_mix_source']),
            float(features['is_code_mix_target'])
        ], dtype=torch.float32).unsqueeze(0).to(device)
        
        self.quality_model.eval()
        with torch.no_grad():
            quality = self.quality_model(feature_tensor).item()
        
        return quality
    
    def update_with_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Update the model with new feedback."""
        self.feedback_dataset.add_feedback(**feedback_data)
        
        # Only update the model if we have enough data and enough time has passed
        if len(self.feedback_dataset) >= self.batch_size and \
           time.time() - self.last_update_time > self.update_interval:
            self._train_model()
            self.last_update_time = time.time()
    
    def _train_model(self) -> None:
        """Train the quality prediction model on the feedback data."""
        if len(self.feedback_dataset) < self.batch_size:
            return
            
        batch = self.feedback_dataset.get_batch(self.batch_size)
        if not batch:
            return
            
        # Prepare training data
        features = []
        targets = []
        
        for item in batch:
            features.append([
                item['features']['source_length'],
                item['features']['target_length'],
                item['features']['length_ratio'],
                item['features']['similarity'],
                float(item['features']['is_code_mix_source']),
                float(item['features']['is_code_mix_target'])
            ])
            
            # Convert feedback to target quality score (0-1)
            if item['feedback_type'] == 'good':
                target = 1.0
            elif item['feedback_type'] == 'needs_work':
                target = 0.5
            elif item['feedback_type'] == 'incorrect':
                target = 0.0
            else:  # neutral or other
                target = item.get('quality', 0.7)  # Use existing quality if available
                
            targets.append(target)
        
        # Convert to tensors
        features_tensor = torch.tensor(features, dtype=torch.float32).to(device)
        targets_tensor = torch.tensor(targets, dtype=torch.float32).unsqueeze(1).to(device)
        
        # Train for a few epochs
        self.quality_model.train()
        for _ in range(5):  # Small number of epochs for online learning
            self.optimizer.zero_grad()
            outputs = self.quality_model(features_tensor)
            loss = self.criterion(outputs, targets_tensor)
            loss.backward()
            self.optimizer.step()
        
        logger.info(f"Updated translation quality model with {len(batch)} feedback samples. Loss: {loss.item():.4f}")
        
        # Save the updated model
        self.save_model()
    
    def save_model(self, path: str = FEEDBACK_MODEL_PATH) -> None:
        """Save the model to disk."""
        torch.save({
            'model_state_dict': self.quality_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        self.feedback_dataset.save(FEEDBACK_DATA_PATH)
    
    @classmethod
    def load_model(cls, feedback_dataset: Optional[FeedbackDataset] = None) -> 'RLTranslationModel':
        """Load the model from disk."""
        if feedback_dataset is None:
            feedback_dataset = FeedbackDataset.load(FEEDBACK_DATA_PATH)
            
        model = cls(feedback_dataset)
        
        try:
            checkpoint = torch.load(FEEDBACK_MODEL_PATH, map_location=device)
            model.quality_model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded existing feedback model")
        except (FileNotFoundError, RuntimeError) as e:
            logger.info("No existing feedback model found, initializing new model")
            
        return model

# Initialize or load the feedback model
feedback_dataset = FeedbackDataset.load(FEEDBACK_DATA_PATH)
rl_model = RLTranslationModel.load_model(feedback_dataset)
