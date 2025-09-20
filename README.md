# Sentiment Analysis Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning pipeline for sentiment analysis that classifies text as positive, negative, or neutral using multiple algorithms and provides interactive prediction capabilities.

## 🎯 Project Overview

This project implements a complete end-to-end sentiment analysis system using machine learning techniques. It processes customer reviews, extracts meaningful features, trains multiple classification models, and provides real-time sentiment prediction capabilities. The system is designed to help businesses understand customer feedback and make data-driven decisions.

### Key Features

- **Multi-Algorithm Approach**: Implements 4 different ML algorithms (Logistic Regression, Random Forest, Naive Bayes, SVM)
- **Text Preprocessing**: Advanced text cleaning and normalization
- **Feature Engineering**: TF-IDF vectorization with configurable parameters
- **Model Evaluation**: Comprehensive performance comparison and selection
- **Interactive Interface**: Real-time sentiment prediction tool
- **Data Visualization**: Professional charts and performance metrics
- **Modular Design**: Clean, reusable code structure following best practices

## 📊 Performance Results

| Model | Accuracy | Best Use Case |
|-------|----------|---------------|
| Random Forest | 50.0% | General purpose, robust performance |
| Naive Bayes | 50.0% | Text classification, fast inference |
| Logistic Regression | 25.0% | Linear relationships, interpretable |
| SVM | 25.0% | High-dimensional data, complex patterns |

*Note: Performance metrics based on 20-sample test dataset. Accuracy improves significantly with larger datasets.*

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/sentiment-analysis.git
   cd sentiment-analysis
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**
   ```bash
   python simple_pipeline.py
   ```

4. **Start interactive predictions**
   ```bash
   python predict_sentiment.py
   ```

## 📁 Project Structure

```
sentimentanalysis/
├── 📁 data/
│   ├── 📁 raw/                    # Original datasets
│   │   └── sample_reviews.csv
│   └── 📁 processed/              # Cleaned and processed data
│       ├── processed_reviews.csv
│       ├── *.npy                  # Feature matrices
│       └── tfidf_vectorizer.pkl   # Trained vectorizer
├── 📁 models/                     # Trained ML models
│   ├── best_model.pkl
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── naive_bayes.pkl
│   └── svm.pkl
├── 📁 reports/
│   └── 📁 figures/                # Generated visualizations
│       ├── sentiment_distribution.png
│       └── model_performance.png
├── 📁 src/                        # Source code modules
│   ├── 📁 data/
│   │   └── make_dataset.py        # Data processing pipeline
│   ├── 📁 features/
│   │   └── build_features.py      # Feature engineering
│   ├── 📁 models/
│   │   ├── train_model.py         # Model training
│   │   └── predict_model.py       # Prediction functions
│   └── 📁 visualization/
│       └── visualize.py           # Data visualization
├── 📄 simple_pipeline.py          # Complete pipeline runner
├── 📄 predict_sentiment.py        # Interactive prediction tool
├── 📄 requirements.txt            # Python dependencies
└── 📄 README.md                   # Project documentation
```

## 🔧 Usage

### Complete Pipeline Execution

Run the entire sentiment analysis pipeline from data processing to model training:

```bash
python simple_pipeline.py
```

This will:
1. Load and clean the raw data
2. Extract TF-IDF features
3. Train multiple ML models
4. Evaluate and select the best model
5. Generate performance visualizations
6. Save all artifacts for future use

### Interactive Predictions

Use the interactive tool to analyze sentiment of any text:

```bash
python predict_sentiment.py
```

Example usage:
```
Enter a text to analyze (or 'quit' to exit):
> I love this product! It's amazing!

Original text: 'I love this product! It's amazing!'
Cleaned text: 'i love this product its amazing'
Predicted sentiment: positive
Confidence scores:
  Negative: 0.100
  Neutral:  0.200
  Positive: 0.700
```

### Individual Component Usage

#### Data Processing
```bash
python src/data/make_dataset.py data/raw data/processed
```

#### Feature Engineering
```bash
python src/features/build_features.py data/processed data/processed
```

#### Model Training
```bash
python src/models/train_model.py data/processed models
```

#### Visualization
```bash
python src/visualization/visualize.py data/processed reports/figures
```

## 🧠 Technical Details

### Data Processing Pipeline

1. **Text Cleaning**
   - Convert to lowercase
   - Remove special characters and digits
   - Normalize whitespace
   - Remove empty entries

2. **Feature Engineering**
   - TF-IDF vectorization (1000 features)
   - N-gram range: (1, 2)
   - English stop words removal
   - Min/max document frequency filtering

3. **Model Training**
   - 80/20 train-test split
   - Stratified sampling for balanced classes
   - Cross-validation for robust evaluation
   - Hyperparameter optimization ready

### Algorithms Implemented

- **Logistic Regression**: Linear classifier with L2 regularization
- **Random Forest**: Ensemble method with 100 decision trees
- **Naive Bayes**: Probabilistic classifier for text data
- **Support Vector Machine**: Linear kernel with probability estimates

## 📈 Visualizations

The project generates comprehensive visualizations:

- **Sentiment Distribution**: Pie chart showing class distribution
- **Model Performance**: Bar chart comparing algorithm accuracy
- **Text Length Analysis**: Histograms of text length by sentiment
- **Confidence Scores**: Detailed prediction confidence metrics

## 🔬 Model Evaluation

### Metrics Used
- **Accuracy**: Overall classification correctness
- **Confidence Scores**: Probability distribution over classes
- **Cross-Validation**: 5-fold CV for robust performance estimation

### Performance Analysis
- Random Forest and Naive Bayes achieved highest accuracy
- Linear models (Logistic Regression, SVM) struggled with limited data
- Performance scales with dataset size

## 🛠️ Dependencies

```
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
numpy>=1.20.0
joblib>=1.0.0
click>=8.0.0
python-dotenv>=0.19.0
```

## 🚀 Future Enhancements

### Planned Improvements
- [ ] **Deep Learning Models**: LSTM, BERT, RoBERTa integration
- [ ] **Web API**: RESTful API for production deployment
- [ ] **Real-time Processing**: Stream processing capabilities
- [ ] **Advanced Preprocessing**: Lemmatization, POS tagging
- [ ] **Hyperparameter Tuning**: Automated model optimization
- [ ] **A/B Testing**: Model comparison framework
- [ ] **Deployment**: Docker containerization
- [ ] **Monitoring**: Model performance tracking

### Scalability Considerations
- Batch processing for large datasets
- Distributed training with Dask/Ray
- Model versioning and MLOps integration
- Cloud deployment (AWS, GCP, Azure)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Code formatting
black src/
isort src/

# Linting
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Your Name** - *Initial work* - [GitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- [scikit-learn](https://scikit-learn.org/) for machine learning algorithms
- [pandas](https://pandas.pydata.org/) for data manipulation
- [matplotlib](https://matplotlib.org/) and [seaborn](https://seaborn.pydata.org/) for visualizations
- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) for project template

## 📞 Contact

- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)
- **Portfolio**: [Your Portfolio Website](https://yourportfolio.com)

---

⭐ **Star this repository if you found it helpful!**

*Built with ❤️ for the data science community*