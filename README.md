# BotOrNot: Twitter Bot Detection System

A machine learning system for detecting bot accounts on Twitter using XGBoost and BERT embeddings, developed as a final project for the Natural Language Processing course at Università degli Studi di Milano.

## 📋 Overview

This project implements a sophisticated bot detection system that combines:
- **Structural features**: Account metadata, engagement metrics, and user behavior patterns
- **BERT embeddings**: Deep semantic representations of user biographies and tweets
- **XGBoost classifier**: Gradient boosting model optimized for bot detection

The system is trained and evaluated on the **TwiBot-22** dataset, achieving high accuracy in distinguishing between genuine human accounts and automated bots.

You can read the related paper and slides: 
- [paper](documentation/Detecting_AI_Generated_Fake_Personas.pdf)
- [slides](documentation/NLP_Slides.pdf)


## 🎯 Features

- **Multi-modal feature extraction**: Combines textual and structural signals
- **Efficient caching**: Saves computed embeddings to avoid redundant calculations
- **Comprehensive evaluation**: Multiple metrics (F1, ROC-AUC, PR-AUC, calibration, etc.)
- **Modular architecture**: Clean separation between data processing, feature engineering, and modeling
- **Easy-to-use scripts**: Simple command-line interfaces for training and inference

## 📁 Project Structure

```
botornot/
├── src/                          # Source code modules
│   ├── data_processing/          # Data loading and preprocessing
│   │   ├── loader.py            # JSON loading utilities
│   │   └── preprocessor.py      # Tweet preprocessing and aggregation
│   ├── features/                 # Feature engineering
│   │   ├── embeddings.py        # BERT embedding generation
│   │   ├── structural.py        # Structural feature extraction
│   │   └── labels.py            # Label loading and normalization
│   ├── models/                   # Model training and evaluation
│   │   ├── trainer.py           # XGBoost model trainer
│   │   └── evaluator.py         # Evaluation metrics and visualization
│   └── utils/                    # Utilities
│       └── config.py            # Configuration management
├── scripts/                      # Executable scripts
│   ├── train.py                 # Model training pipeline
│   ├── predict.py               # Inference on new data
│   └── preprocess_tweets.py    # Tweet preprocessing
├── notebook.ipynb               # Jupyter notebook for exploration
├── data/                        # Data directory (not included)
│   ├── postprocessing/          # Processed data files
│   └── split.csv               # Train/val/test split
├── models/                      # Trained models (created during training)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, but recommended for faster training)
- 8GB+ RAM (16GB recommended for full dataset)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/MaxKappa/botornot
cd botornot
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the TwiBot-22 dataset**

Download the dataset from the [TwiBot-22 repository](https://github.com/LuoUndergradXJTU/TwiBot-22) and place the files in the `data/` directory:
- `user.json` → `data/postprocessing/user.json`
- `tweet_*.json` → `data/postprocessing/tweet_*.json`
- `split.csv` → `data/split.csv`
- `label.csv` or `label.json` → `data/postprocessing/`

### Quick Start

#### 1. Preprocess Tweets (Optional)

If you have raw tweet files that need preprocessing:

```bash
python scripts/preprocess_tweets.py --input-dir ./data --output-dir ./data_processed
```

#### 2. Train the Model

```bash
python scripts/train.py
```

This will:
- Load and aggregate user and tweet data
- Extract structural features
- Generate BERT embeddings (with caching)
- Train the XGBoost model
- Evaluate on test set
- Save the trained model to `models/xgb_model_trained.joblib`

Training takes approximately 30-60 minutes on a modern GPU (or 2-4 hours on CPU for the full dataset).

#### 3. Make Predictions

```bash
python scripts/predict.py data/postprocessing/user.json --output predictions.csv
```

## 📊 Model Architecture

### Feature Engineering

1. **Structural Features**
   - Account metadata: followers, following, tweet count, account age
   - Profile characteristics: default profile image, verified status
   - Engagement metrics: average likes, retweets, replies, quotes
   - Content diversity: unique text ratio

2. **BERT Embeddings**
   - **Bio embeddings**: 384-dimensional representation of user biography
   - **Tweet embeddings**: Mean pooling of up to 20 tweets per user
   - Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`

3. **Total Feature Dimension**: ~800 features (structural + embeddings)

### XGBoost Classifier

- **Objective**: Binary classification (human vs. bot)
- **Hyperparameters**:
  - Learning rate: 0.02
  - Max depth: 5
  - Subsample: 0.7
  - Column subsample: 0.7
  - Early stopping: 50 rounds
- **Threshold optimization**: F1-score maximization on validation set

## 📈 Performance

On the TwiBot-22 test set:

| Metric              | Score  |
|---------------------|--------|
| Accuracy            | ~70%|
| F1 Score (Bot)      | ~58%|


## 🔧 Configuration

Modify `src/utils/config.py` to customize:
- Data paths
- Model hyperparameters
- BERT model selection
- Batch size and device settings
- Caching behavior

Example:
```python
from src.utils import Config

Config.update(
    BATCH_SIZE=32,
    N_TWEETS_PER_USER=50,
    BERT_MODEL="your-preferred-model"
)
```


## 📚 References

- **TwiBot-22 Dataset**: [Paper](https://arxiv.org/abs/2206.04564) | [Repository](https://github.com/LuoUndergradXJTU/TwiBot-22)
- **BERT Embeddings**: [Sentence Transformers](https://www.sbert.net/)
- **XGBoost**: [Documentation](https://xgboost.readthedocs.io/)


## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Author

**Massimiliano**  
Natural Language Processing Course  
Università degli Studi di Milano

## 🙏 Acknowledgments

- TwiBot-22 dataset creators for providing high-quality annotated data

---

