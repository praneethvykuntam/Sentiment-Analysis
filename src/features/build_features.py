# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs feature engineering scripts to turn processed data into features
        for modeling (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features from processed data')
    
    # Read processed data
    processed_file = Path(input_filepath) / 'processed_reviews.csv'
    df = pd.read_csv(processed_file)
    
    logger.info(f'Loaded {len(df)} processed samples')
    
    # Prepare features and labels
    X = df['cleaned_text'].values
    y = df['sentiment_label'].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Count Vectorization
    count_vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.95
    )
    
    X_train_count = count_vectorizer.fit_transform(X_train)
    X_test_count = count_vectorizer.transform(X_test)
    
    # Create output directory
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save training and test data
    np.save(output_path / 'X_train_tfidf.npy', X_train_tfidf.toarray())
    np.save(output_path / 'X_test_tfidf.npy', X_test_tfidf.toarray())
    np.save(output_path / 'X_train_count.npy', X_train_count.toarray())
    np.save(output_path / 'X_test_count.npy', X_test_count.toarray())
    np.save(output_path / 'y_train.npy', y_train)
    np.save(output_path / 'y_test.npy', y_test)
    
    # Save vectorizers
    import joblib
    joblib.dump(tfidf_vectorizer, output_path / 'tfidf_vectorizer.pkl')
    joblib.dump(count_vectorizer, output_path / 'count_vectorizer.pkl')
    
    # Save text data for reference
    pd.DataFrame({'text': X_train, 'sentiment': y_train}).to_csv(
        output_path / 'train_data.csv', index=False
    )
    pd.DataFrame({'text': X_test, 'sentiment': y_test}).to_csv(
        output_path / 'test_data.csv', index=False
    )
    
    logger.info(f'Features saved to {output_path}')
    logger.info(f'Training set: {X_train_tfidf.shape[0]} samples, {X_train_tfidf.shape[1]} features')
    logger.info(f'Test set: {X_test_tfidf.shape[0]} samples, {X_test_tfidf.shape[1]} features')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
