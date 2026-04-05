import os
import sys

# Add current directory to path
sys.path.append(os.getcwd())

from src.model.predictor import SentimentPredictor

def test_inference():
    print("🚀 Starting Inference Verification...")
    
    MODEL_PATH = os.path.join("models", "sentiment_lstm_model.keras")
    TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")
    
    try:
        predictor = SentimentPredictor(MODEL_PATH, TOKENIZER_PATH)
        print("✅ Predictor Initialized.")
        
        test_cases = [
            ("This movie was absolutely amazing! The story was compelling and the acting was top-notch.", "Positive"),
            ("I hated this film. It was boring and a waste of time.", "Negative")
        ]
        
        for text, expected in test_cases:
            result = predictor.predict(text)
            print(f"\nTest Text: {text[:50]}...")
            print(f"Result: {result['label']} ({result['confidence']}% confidence)")
            
            if result['label'] == expected:
                print(f"✅ Pass: Expected {expected}, got {result['label']}")
            else:
                print(f"❌ Fail: Expected {expected}, got {result['label']}")
                
    except Exception as e:
        print(f"❌ Error during verification: {e}")

if __name__ == "__main__":
    test_inference()
