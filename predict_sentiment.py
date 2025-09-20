#!/usr/bin/env python3
"""
Simple script to make sentiment predictions using the trained model.
"""

import pandas as pd
import numpy as np
import joblib
import re
from pathlib import Path

def clean_text(text):
    """Clean text data."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_sentiment(text):
    """Predict sentiment for a given text."""
    # Load vectorizer and model
    vectorizer = joblib.load('data/processed/tfidf_vectorizer.pkl')
    model = joblib.load('models/best_model.pkl')
    
    # Clean and vectorize text
    cleaned_text = clean_text(text)
    X_vec = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(X_vec)[0]
    confidence = model.predict_proba(X_vec)[0]
    
    # Map to sentiment labels
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiment = sentiment_map[prediction]
    
    return predicted_sentiment, confidence, cleaned_text

def main():
    print("Sentiment Analysis Prediction Tool")
    print("=" * 50)
    
    # Check if model files exist
    if not Path('models/best_model.pkl').exists():
        print("Error: Model not found. Please run the pipeline first.")
        return
    
    # Interactive prediction
    while True:
        print("\nEnter a text to analyze (or 'quit' to exit):")
        text = input("> ")
        
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not text.strip():
            print("Please enter some text.")
            continue
        
        try:
            sentiment, confidence, cleaned = predict_sentiment(text)
            
            print(f"\nOriginal text: '{text}'")
            print(f"Cleaned text: '{cleaned}'")
            print(f"Predicted sentiment: {sentiment}")
            print(f"Confidence scores:")
            print(f"  Negative: {confidence[0]:.3f}")
            print(f"  Neutral:  {confidence[1]:.3f}")
            print(f"  Positive: {confidence[2]:.3f}")
            print("-" * 50)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
