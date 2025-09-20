# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import re
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


def clean_text(text):
    """Clean text data by removing special characters and normalizing."""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # Read the raw data
    raw_data_path = Path(input_filepath) / 'sample_reviews.csv'
    df = pd.read_csv(raw_data_path)
    
    logger.info(f'Loaded {len(df)} samples from raw data')
    
    # Clean the text data
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Remove empty texts after cleaning
    df = df[df['cleaned_text'].str.len() > 0]
    
    # Map sentiment labels to numeric values
    sentiment_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    df['sentiment_label'] = df['sentiment'].map(sentiment_map)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save processed data
    processed_file = output_path / 'processed_reviews.csv'
    df.to_csv(processed_file, index=False)
    
    logger.info(f'Processed data saved to {processed_file}')
    logger.info(f'Final dataset contains {len(df)} samples')
    logger.info(f'Sentiment distribution: {df["sentiment"].value_counts().to_dict()}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
