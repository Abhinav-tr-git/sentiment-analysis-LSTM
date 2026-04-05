import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.utils.preprocess import clean_text

class SentimentPredictor:
    def __init__(self, model_path, tokenizer_path, max_len=200):
        """
        Initializes the predictor by loading the model and tokenizer.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found at {tokenizer_path}")
            
        self.model = load_model(model_path)
        with open(tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)
        self.max_len = max_len

    def predict(self, text):
        """
        Preprocesses text and returns prediction results.
        """
        # 1. Clean
        cleaned = clean_text(text)
        
        # 2. Tokenize
        seq = self.tokenizer.texts_to_sequences([cleaned])
        
        # 3. Pad
        padded = pad_sequences(seq, maxlen=self.max_len)
        
        # 4. Predict
        prediction = self.model.predict(padded, verbose=0)[0][0]
        
        label = "Positive" if prediction > 0.5 else "Negative"
        confidence = float(prediction) if label == "Positive" else 1 - float(prediction)
        
        return {
            "label": label,
            "confidence": round(confidence * 100, 2),
            "original_score": float(prediction)
        }
