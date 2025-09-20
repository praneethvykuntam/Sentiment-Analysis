# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from dotenv import find_dotenv, load_dotenv


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


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('features_filepath', type=click.Path(exists=True))
@click.argument('input_text', type=str)
def main(model_filepath, features_filepath, input_text):
    """ Makes predictions using trained models.
    """
    logger = logging.getLogger(__name__)
    logger.info('making predictions with trained model')
    
    # Load the best model
    model_path = Path(model_filepath)
    best_model_file = model_path / 'best_model.pkl'
    
    if not best_model_file.exists():
        logger.error(f'Best model file not found at {best_model_file}')
        return
    
    model = joblib.load(best_model_file)
    logger.info(f'Loaded model from {best_model_file}')
    
    # Load vectorizers
    features_path = Path(features_filepath)
    tfidf_vectorizer = joblib.load(features_path / 'tfidf_vectorizer.pkl')
    count_vectorizer = joblib.load(features_path / 'count_vectorizer.pkl')
    
    # Clean input text
    cleaned_text = clean_text(input_text)
    logger.info(f'Cleaned input text: {cleaned_text}')
    
    # Determine which vectorizer to use based on model name
    # This is a simplified approach - in practice, you'd want to store this info
    # with the model or determine it programmatically
    try:
        # Try TF-IDF first
        X_tfidf = tfidf_vectorizer.transform([cleaned_text])
        prediction = model.predict(X_tfidf)[0]
        prediction_proba = model.predict_proba(X_tfidf)[0]
        vectorizer_used = 'TF-IDF'
    except:
        # Fall back to Count vectorizer
        X_count = count_vectorizer.transform([cleaned_text])
        prediction = model.predict(X_count)[0]
        prediction_proba = model.predict_proba(X_count)[0]
        vectorizer_used = 'Count'
    
    # Map prediction back to sentiment
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_sentiment = sentiment_map[prediction]
    
    # Get confidence scores
    confidence_scores = {
        'negative': prediction_proba[0],
        'neutral': prediction_proba[1],
        'positive': prediction_proba[2]
    }
    
    logger.info(f'Prediction: {predicted_sentiment}')
    logger.info(f'Confidence scores: {confidence_scores}')
    logger.info(f'Vectorizer used: {vectorizer_used}')
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Sentiment Analysis Results")
    print(f"{'='*50}")
    print(f"Input text: {input_text}")
    print(f"Cleaned text: {cleaned_text}")
    print(f"Predicted sentiment: {predicted_sentiment}")
    print(f"Confidence scores:")
    for sentiment, score in confidence_scores.items():
        print(f"  {sentiment}: {score:.4f}")
    print(f"Vectorizer used: {vectorizer_used}")
    print(f"{'='*50}\n")


@click.command()
@click.argument('model_filepath', type=click.Path(exists=True))
@click.argument('features_filepath', type=click.Path(exists=True))
@click.argument('input_file', type=click.Path(exists=True))
def batch_predict(model_filepath, features_filepath, input_file):
    """ Makes batch predictions using trained models.
    """
    logger = logging.getLogger(__name__)
    logger.info('making batch predictions with trained model')
    
    # Load the best model
    model_path = Path(model_filepath)
    best_model_file = model_path / 'best_model.pkl'
    
    if not best_model_file.exists():
        logger.error(f'Best model file not found at {best_model_file}')
        return
    
    model = joblib.load(best_model_file)
    logger.info(f'Loaded model from {best_model_file}')
    
    # Load vectorizers
    features_path = Path(features_filepath)
    tfidf_vectorizer = joblib.load(features_path / 'tfidf_vectorizer.pkl')
    count_vectorizer = joblib.load(features_path / 'count_vectorizer.pkl')
    
    # Load input data
    df = pd.read_csv(input_file)
    logger.info(f'Loaded {len(df)} samples for batch prediction')
    
    # Clean texts
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Make predictions
    try:
        X_tfidf = tfidf_vectorizer.transform(df['cleaned_text'])
        predictions = model.predict(X_tfidf)
        prediction_probas = model.predict_proba(X_tfidf)
        vectorizer_used = 'TF-IDF'
    except:
        X_count = count_vectorizer.transform(df['cleaned_text'])
        predictions = model.predict(X_count)
        prediction_probas = model.predict_proba(X_count)
        vectorizer_used = 'Count'
    
    # Map predictions back to sentiments
    sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    df['predicted_sentiment'] = [sentiment_map[p] for p in predictions]
    df['confidence_negative'] = prediction_probas[:, 0]
    df['confidence_neutral'] = prediction_probas[:, 1]
    df['confidence_positive'] = prediction_probas[:, 2]
    
    # Save results
    output_file = Path(input_file).parent / 'predictions.csv'
    df.to_csv(output_file, index=False)
    
    logger.info(f'Batch predictions saved to {output_file}')
    logger.info(f'Vectorizer used: {vectorizer_used}')
    
    # Print summary
    sentiment_counts = df['predicted_sentiment'].value_counts()
    print(f"\n{'='*50}")
    print(f"Batch Prediction Results")
    print(f"{'='*50}")
    print(f"Total samples: {len(df)}")
    print(f"Sentiment distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)")
    print(f"Vectorizer used: {vectorizer_used}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # For single prediction
    main()
