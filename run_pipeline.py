#!/usr/bin/env python3
"""
Simple script to run the complete sentiment analysis pipeline.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def process_data():
    """Process raw data into cleaned data."""
    logger.info("Processing raw data...")
    
    # Read raw data
    raw_data_path = Path('data/raw/sample_reviews.csv')
    df = pd.read_csv(raw_data_path)
    logger.info(f'Loaded {len(df)} samples from raw data')
    
    # Clean the text data
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Map sentiment labels to numeric values
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Create output directory
    output_path = Path('data/processed')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    processed_file = output_path / 'processed_reviews.csv'
    df.to_csv(processed_file, index=False)
    
    logger.info(f'Processed data saved to {processed_file}')
    logger.info(f'Final dataset contains {len(df)} samples')
    logger.info(f'Sentiment distribution: {df["sentiment"].value_counts().to_dict()}')
    
    return df

def build_features(df):
    """Build features from processed data."""
    logger.info("Building features...")
    
    # Prepare features and labels
    X = df['cleaned_text'].values
    y = df['sentiment_label'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=1,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Save features
    features_path = Path('data/processed')
    np.save(features_path / 'X_train_tfidf.npy', X_train_tfidf.toarray())
    np.save(features_path / 'X_test_tfidf.npy', X_test_tfidf.toarray())
    np.save(features_path / 'y_train.npy', y_train)
    np.save(features_path / 'y_test.npy', y_test)
    
    # Save vectorizer
    joblib.dump(tfidf_vectorizer, features_path / 'tfidf_vectorizer.pkl')
    
    logger.info(f'Features saved to {features_path}')
    logger.info(f'Training set: {X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features')
    logger.info(f'Test set: {X_test_tfidf.shape[0]} samples, {X_test_tfidf.shape[1]} features')
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer

def train_models(X_train, X_test, y_train, y_test):
    """Train sentiment analysis models."""
    logger.info("Training models...")
    
    # Create models directory
    model_path = Path('models')
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Define models
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'naive_bayes': MultinomialNB(),
        'svm': SVC(kernel='linear', random_state=42, probability=True)
    }
    
    results = {}
    
    for model_name, model in models.items():
        logger.info(f'Training {model_name}...')
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f'{model_name} - Accuracy: {accuracy:.4f}')
        
        # Save the model
        model_file = model_path / f'{model_name}.pkl'
        joblib.dump(model, model_file)
        
        results[model_name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    logger.info(f'Best model: {best_model_name} with accuracy {results[best_model_name]["accuracy"]:.4f}')
    
    # Save best model
    best_model_file = model_path / 'best_model.pkl'
    joblib.dump(best_model, best_model_file)
    
    # Save results summary
    results_summary = []
    for model_name, result in results.items():
        results_summary.append({
            'model': model_name,
            'accuracy': result['accuracy']
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df.to_csv(model_path / 'model_results.csv', index=False)
    
    logger.info(f'Model training completed. Results saved to {model_path}')
    
    return results, best_model_name

def create_visualizations(df):
    """Create visualizations for the project."""
    logger.info("Creating visualizations...")
    
    # Create output directory
    output_path = Path('reports/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    
    # 1. Sentiment Distribution
    plt.figure(figsize=(10, 6))
    sentiment_counts = df['sentiment'].value_counts()
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', 
            colors=colors, startangle=90)
    plt.title('Distribution of Sentiment Labels', fontsize=16, fontweight='bold')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(output_path / 'sentiment_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Text Length Distribution
    df['text_length'] = df['text'].str.len()
    df['cleaned_text_length'] = df['cleaned_text'].str.len()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.hist(subset['text_length'], alpha=0.7, label=sentiment, bins=10)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Original Text Length Distribution by Sentiment')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.hist(subset['cleaned_text_length'], alpha=0.7, label=sentiment, bins=10)
    plt.xlabel('Cleaned Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Cleaned Text Length Distribution by Sentiment')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f'Visualizations saved to {output_path}')

def test_predictions(tfidf_vectorizer, best_model_name):
    """Test the trained model with sample predictions."""
    logger.info("Testing predictions...")
    
    # Load best model
    model_path = Path('models')
    best_model_file = model_path / 'best_model.pkl'
    model = joblib.load(best_model_file)
    
    # Test texts
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. I hate it.",
        "It's okay, nothing special.",
        "Outstanding quality and service!",
        "Worst purchase ever. Complete waste."
    ]
    
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    
    print("\n" + "="*60)
    print("SENTIMENT ANALYSIS PREDICTIONS")
    print("="*60)
    
    for text in test_texts:
        cleaned = clean_text(text)
        X_vec = tfidf_vectorizer.transform([cleaned])
        prediction = model.predict(X_vec)[0]
        confidence = model.predict_proba(X_vec)[0]
        
        predicted_sentiment = sentiment_map[prediction]
        max_confidence = max(confidence)
        
        print(f"Text: '{text}'")
        print(f"Prediction: {predicted_sentiment} (confidence: {max_confidence:.3f})")
        print(f"Confidence scores: {dict(zip(['negative', 'neutral', 'positive'], confidence))}")
        print("-" * 60)
    
    print(f"Best model used: {best_model_name}")
    print("="*60)

def main():
    """Run the complete sentiment analysis pipeline."""
    print("🚀 Starting Sentiment Analysis Pipeline")
    print("="*60)
    
    try:
        # Step 1: Process data
        df = process_data()
        
        # Step 2: Build features
        X_train, X_test, y_train, y_test, tfidf_vectorizer = build_features(df)
        
        # Step 3: Train models
        results, best_model_name = train_models(X_train, X_test, y_train, y_test)
        
        # Step 4: Create visualizations
        create_visualizations(df)
        
        # Step 5: Test predictions
        test_predictions(tfidf_vectorizer, best_model_name)
        
        print("\n✅ Pipeline completed successfully!")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        print(f"\n❌ Pipeline failed: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
