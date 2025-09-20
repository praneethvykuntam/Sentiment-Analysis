#!/usr/bin/env python3
"""
Simple sentiment analysis pipeline without special characters.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import matplotlib.pyplot as plt

def clean_text(text):
    """Clean text data."""
    import re
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def main():
    print("Starting Sentiment Analysis Pipeline")
    print("=" * 50)
    
    # Step 1: Load and process data
    print("Step 1: Loading data...")
    df = pd.read_csv('data/raw/sample_reviews.csv')
    print(f"Loaded {len(df)} samples")
    
    # Clean text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Map sentiments
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Save processed data
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    df.to_csv('data/processed/processed_reviews.csv', index=False)
    print("Processed data saved")
    
    # Step 2: Prepare features
    print("Step 2: Building features...")
    X = df['cleaned_text'].values
    y = df['sentiment_label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Save features
    np.save('data/processed/X_train_tfidf.npy', X_train_vec.toarray())
    np.save('data/processed/X_test_tfidf.npy', X_test_vec.toarray())
    np.save('data/processed/y_train.npy', y_train)
    np.save('data/processed/y_test.npy', y_test)
    joblib.dump(vectorizer, 'data/processed/tfidf_vectorizer.pkl')
    print("Features saved")
    
    # Step 3: Train models
    print("Step 3: Training models...")
    Path('models').mkdir(parents=True, exist_ok=True)
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'naive_bayes': MultinomialNB(),
        'svm': SVC(kernel='linear', random_state=42, probability=True)
    }
    
    results = {}
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        print(f"{name} accuracy: {accuracy:.4f}")
        
        # Save model
        joblib.dump(model, f'models/{name}.pkl')
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x])
    best_model = models[best_model_name]
    joblib.dump(best_model, 'models/best_model.pkl')
    print(f"Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")
    
    # Step 4: Test predictions
    print("Step 4: Testing predictions...")
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Outstanding quality and service!",
        "Worst purchase ever. Complete waste."
    ]
    
    sentiment_map_reverse = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    for text in test_texts:
        cleaned = clean_text(text)
        X_vec = vectorizer.transform([cleaned])
        prediction = best_model.predict(X_vec)[0]
        confidence = best_model.predict_proba(X_vec)[0]
        
        predicted_sentiment = sentiment_map_reverse[prediction]
        max_confidence = max(confidence)
        
        print(f"Text: '{text}'")
        print(f"Prediction: {predicted_sentiment} (confidence: {max_confidence:.3f})")
        print("-" * 40)
    
    # Step 5: Create simple visualization
    print("Step 5: Creating visualizations...")
    Path('reports/figures').mkdir(parents=True, exist_ok=True)
    
    # Sentiment distribution pie chart
    plt.figure(figsize=(8, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
            autopct='%1.1f%%', colors=colors, startangle=90)
    plt.title('Distribution of Sentiment Labels')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig('reports/figures/sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Model performance bar chart
    plt.figure(figsize=(10, 6))
    model_names = list(results.keys())
    accuracies = list(results.values())
    bars = plt.bar(model_names, accuracies, color=colors[:len(model_names)])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('reports/figures/model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualizations saved")
    
    print("\nPipeline completed successfully!")
    print("=" * 50)
    print("Files created:")
    print("- data/processed/processed_reviews.csv")
    print("- data/processed/*.npy (feature files)")
    print("- models/*.pkl (trained models)")
    print("- reports/figures/*.png (visualizations)")

if __name__ == '__main__':
    main()
