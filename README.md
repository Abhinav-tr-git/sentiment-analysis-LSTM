Sentiment Analysis using LSTM
 Project Overview

This project implements a Deep Learning-based Sentiment Analysis system that classifies customer reviews as Positive or Negative using an LSTM (Long Short-Term Memory) model.

The model is trained on the IMDb dataset (50,000 reviews) and achieves ~89% accuracy, making it suitable for real-world applications like customer feedback analysis, product reviews, and social media monitoring.

 Problem Statement

Analyzing large volumes of customer feedback manually is inefficient.
This project automates sentiment classification using Natural Language Processing (NLP) and Deep Learning.

 Model Architecture
Input Text
   ↓
Tokenization
   ↓
Padding (Fixed Length)
   ↓
Embedding Layer (64-dim)
   ↓
LSTM (64 units)
   ↓
Dense Layer (Sigmoid)
   ↓
Output (Positive / Negative)
🛠 Tech Stack
Python
TensorFlow / Keras
NumPy & Pandas
Scikit-learn
Matplotlib / Seaborn
 Dataset
IMDb Movie Reviews Dataset
50,000 labeled reviews
Balanced dataset (Positive & Negative)

 Download: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

 Features
Text preprocessing (tokenization, padding)
Vocabulary limitation (top 5000 words)
LSTM for sequential text learning
Model evaluation using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Model saving & loading for deployment
 Results
Accuracy: ~89%
Precision: ~0.91
Recall: ~0.88
F1-score: ~0.89

✔ Good generalization on unseen data
✔ Balanced performance across classes

 Evaluation Metrics
Confusion Matrix
Classification Report
Accuracy & Loss Graphs
💾 Model Saving
Model saved in .keras format
Tokenizer saved using pickle
