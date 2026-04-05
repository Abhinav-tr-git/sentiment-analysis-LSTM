import os
from flask import Flask, render_template, request, jsonify
from src.model.predictor import SentimentPredictor

app = Flask(__name__)

# Constants
MODEL_PATH = os.path.join("models", "sentiment_lstm_model.keras")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")

# Initialize Predictor
try:
    predictor = SentimentPredictor(MODEL_PATH, TOKENIZER_PATH)
    print("✅ Model and Tokenizer loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model/tokenizer: {e}")
    predictor = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not predictor:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.json
    if not data or "text" not in data:
        return jsonify({"error": "No text provided"}), 400
    
    text = data["text"]
    if not text.strip():
        return jsonify({"error": "Empty text provided"}), 400
        
    try:
        result = predictor.predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
