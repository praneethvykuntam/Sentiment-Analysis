#!/usr/bin/env python3
"""
Simple test script to verify the sentiment analysis pipeline works.
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

def clean_text(text):
    """Clean text data by removing special characters and normalizing."""
    import re
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("=" * 60)
    print("SENTIMENT ANALYSIS PIPELINE TEST")
    print("=" * 60)
    
    # Check if processed data exists
    processed_file = Path('data/processed/processed_reviews.csv')
    if not processed_file.exists():
        print("❌ Processed data not found. Please run the data processing first.")
        return
    
    # Load processed data
    print("📊 Loading processed data...")
    df = pd.read_csv(processed_file)
    print(f"   Loaded {len(df)} samples")
    print(f"   Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
    
    # Prepare features
    print("\n🔧 Preparing features...")
    X = df['cleaned_text'].values
    y = df['sentiment_label'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    print("🤖 Training model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   Model accuracy: {accuracy:.4f}")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Outstanding quality and service!",
        "Worst purchase ever. Complete waste."
    ]
    
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    for text in test_texts:
        cleaned = clean_text(text)
        X_vec = vectorizer.transform([cleaned])
        prediction = model.predict(X_vec)[0]
        confidence = model.predict_proba(X_vec)[0]
        
        predicted_sentiment = sentiment_map[prediction]
        max_confidence = max(confidence)
        
        print(f"   '{text}'")
        print(f"   → {predicted_sentiment} (confidence: {max_confidence:.3f})")
        print()
    
    # Check if models directory exists and has files
    models_dir = Path('models')
    if models_dir.exists():
        model_files = list(models_dir.glob('*.pkl'))
        print(f"📁 Found {len(model_files)} model files in models/ directory")
        for f in model_files:
            print(f"   - {f.name}")
    else:
        print("📁 No models directory found")
    
    # Check if visualizations exist
    figures_dir = Path('reports/figures')
    if figures_dir.exists():
        figure_files = list(figures_dir.glob('*.png'))
        print(f"📊 Found {len(figure_files)} visualization files in reports/figures/")
        for f in figure_files:
            print(f"   - {f.name}")
    else:
        print("📊 No figures directory found")
    
    print("\n✅ Pipeline test completed successfully!")
    print("=" * 60)

if __name__ == '__main__':
    main()
