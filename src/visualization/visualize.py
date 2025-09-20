# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Creates visualizations for the sentiment analysis project.
    """
    logger = logging.getLogger(__name__)
    logger.info('creating visualizations')
    
    # Create output directory
    output_path = Path(output_filepath)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set style
    try:
        plt.style.use('seaborn-v0_8')
    except:
        plt.style.use('seaborn')
    sns.set_palette("husl")
    
    # Load processed data
    processed_file = Path(input_filepath) / 'processed_reviews.csv'
    df = pd.read_csv(processed_file)
    
    logger.info(f'Loaded {len(df)} samples for visualization')
    
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
    
    # 2. Text Length Distribution by Sentiment
    df['text_length'] = df['text'].str.len()
    df['cleaned_text_length'] = df['cleaned_text'].str.len()
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.hist(subset['text_length'], alpha=0.7, label=sentiment, bins=20)
    plt.xlabel('Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Original Text Length Distribution by Sentiment')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.hist(subset['cleaned_text_length'], alpha=0.7, label=sentiment, bins=20)
    plt.xlabel('Cleaned Text Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Cleaned Text Length Distribution by Sentiment')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path / 'text_length_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Word Count Distribution
    df['word_count'] = df['cleaned_text'].str.split().str.len()
    
    plt.figure(figsize=(12, 6))
    for sentiment in df['sentiment'].unique():
        subset = df[df['sentiment'] == sentiment]
        plt.hist(subset['word_count'], alpha=0.7, label=sentiment, bins=15)
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.title('Word Count Distribution by Sentiment')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path / 'word_count_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Average Text Length by Sentiment
    avg_lengths = df.groupby('sentiment')['cleaned_text_length'].mean()
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_lengths.index, avg_lengths.values, color=colors)
    plt.xlabel('Sentiment')
    plt.ylabel('Average Text Length (characters)')
    plt.title('Average Text Length by Sentiment')
    
    # Add value labels on bars
    for bar, value in zip(bars, avg_lengths.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_path / 'average_text_length.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Model Performance Visualization (if available)
    model_results_file = Path(input_filepath).parent / 'models' / 'model_results.csv'
    if model_results_file.exists():
        model_results = pd.read_csv(model_results_file)
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(model_results)), model_results['accuracy'], 
                      color=plt.cm.viridis(np.linspace(0, 1, len(model_results))))
        plt.xlabel('Models')
        plt.ylabel('Accuracy')
        plt.title('Model Performance Comparison')
        plt.xticks(range(len(model_results)), model_results['model'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, model_results['accuracy'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Cross-validation scores
        plt.figure(figsize=(12, 8))
        x_pos = np.arange(len(model_results))
        bars = plt.bar(x_pos, model_results['cv_score'], 
                      yerr=model_results['cv_std'], 
                      color=plt.cm.plasma(np.linspace(0, 1, len(model_results))),
                      capsize=5, alpha=0.7)
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Score')
        plt.title('Model Cross-Validation Performance')
        plt.xticks(x_pos, model_results['model'], rotation=45, ha='right')
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, model_results['cv_score'])):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(output_path / 'cv_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f'Visualizations saved to {output_path}')
    
    # Create a summary report
    summary_file = output_path / 'visualization_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("Sentiment Analysis Visualization Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples analyzed: {len(df)}\n")
        f.write(f"Sentiment distribution:\n")
        for sentiment, count in sentiment_counts.items():
            f.write(f"  {sentiment}: {count} ({count/len(df)*100:.1f}%)\n")
        f.write(f"\nAverage text length by sentiment:\n")
        for sentiment, length in avg_lengths.items():
            f.write(f"  {sentiment}: {length:.1f} characters\n")
        f.write(f"\nVisualizations created:\n")
        f.write(f"  - sentiment_distribution.png\n")
        f.write(f"  - text_length_distribution.png\n")
        f.write(f"  - word_count_distribution.png\n")
        f.write(f"  - average_text_length.png\n")
        if model_results_file.exists():
            f.write(f"  - model_performance.png\n")
            f.write(f"  - cv_performance.png\n")
    
    logger.info(f'Summary report saved to {summary_file}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
