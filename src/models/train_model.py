# -*- coding: utf-8 -*-
import click
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
from dotenv import find_dotenv, load_dotenv


def train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a model, returning results."""
    logger = logging.getLogger(__name__)
    
    # Train the model
    logger.info(f'Training {model_name}...')
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)
    
    logger.info(f'{model_name} - Accuracy: {accuracy:.4f}')
    logger.info(f'{model_name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})')
    
    return {
        'model': model,
        'accuracy': accuracy,
        'cv_score': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred,
        'classification_report': classification_report(y_test, y_pred)
    }


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('model_filepath', type=click.Path())
def main(input_filepath, model_filepath):
    """ Trains sentiment analysis models and saves them to the models directory.
    """
    logger = logging.getLogger(__name__)
    logger.info('training sentiment analysis models')
    
    # Load features
    features_path = Path(input_filepath)
    X_train_tfidf = np.load(features_path / 'X_train_tfidf.npy')
    X_test_tfidf = np.load(features_path / 'X_test_tfidf.npy')
    X_train_count = np.load(features_path / 'X_train_count.npy')
    X_test_count = np.load(features_path / 'X_test_count.npy')
    y_train = np.load(features_path / 'y_train.npy')
    y_test = np.load(features_path / 'y_test.npy')
    
    logger.info(f'Loaded training data: {X_train_tfidf.shape}')
    logger.info(f'Loaded test data: {X_test_tfidf.shape}')
    
    # Create models directory
    model_path = Path(model_filepath)
    model_path.mkdir(parents=True, exist_ok=True)
    
    # Define models to train
    models = {
        'logistic_regression_tfidf': LogisticRegression(random_state=42, max_iter=1000),
        'logistic_regression_count': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest_tfidf': RandomForestClassifier(n_estimators=100, random_state=42),
        'random_forest_count': RandomForestClassifier(n_estimators=100, random_state=42),
        'naive_bayes_tfidf': MultinomialNB(),
        'naive_bayes_count': MultinomialNB(),
        'svm_tfidf': SVC(kernel='linear', random_state=42),
        'svm_count': SVC(kernel='linear', random_state=42)
    }
    
    # Train models and collect results
    results = {}
    
    # Train TF-IDF models
    for model_name, model in models.items():
        if 'tfidf' in model_name:
            X_train, X_test = X_train_tfidf, X_test_tfidf
        else:
            X_train, X_test = X_train_count, X_test_count
            
        result = train_and_evaluate_model(model, model_name, X_train, X_test, y_train, y_test)
        results[model_name] = result
        
        # Save the model
        model_file = model_path / f'{model_name}.pkl'
        joblib.dump(model, model_file)
        logger.info(f'Saved {model_name} to {model_file}')
    
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
            'accuracy': result['accuracy'],
            'cv_score': result['cv_score'],
            'cv_std': result['cv_std']
        })
    
    results_df = pd.DataFrame(results_summary)
    results_df = results_df.sort_values('accuracy', ascending=False)
    results_df.to_csv(model_path / 'model_results.csv', index=False)
    
    # Save detailed classification report for best model
    with open(model_path / 'best_model_report.txt', 'w') as f:
        f.write(f"Best Model: {best_model_name}\n")
        f.write(f"Accuracy: {results[best_model_name]['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(results[best_model_name]['classification_report'])
    
    logger.info(f'Model training completed. Results saved to {model_path}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
