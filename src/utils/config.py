"""Configuration management"""

import os
import torch


class Config:
    """Central configuration for the bot detection pipeline"""
    
    # Data paths
    DATA_DIR = "./data/postprocessing/"
    TWEETS_GLOB = os.path.join(DATA_DIR, "tweet_*_processed.json")
    USER_JSON = os.path.join(DATA_DIR, "user.json")
    SPLIT_CSV = "./data/split.csv"
    
    # Model paths
    MODEL_DIR = "./models"
    MODEL_PATH = os.path.join(MODEL_DIR, "xgb_model_trained.joblib")
    
    # Cache paths
    BIO_EMBEDDINGS_CACHE = os.path.join(DATA_DIR, "precalc_bio_embeddings.parquet")
    TWEET_EMBEDDINGS_CACHE = os.path.join(DATA_DIR, "precalc_tweet_avg_embeddings.csv")
    
    # BERT model settings
    BERT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    MAX_LENGTH = 512
    BATCH_SIZE = 16
    EMBEDDING_DIM = 384
    
    # Device selection
    @staticmethod
    def get_device():
        """Auto-detect best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    # Training settings
    RANDOM_STATE = 42
    N_TWEETS_PER_USER = 20
    
    # XGBoost hyperparameters
    XGB_PARAMS = {
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'max_depth': 5,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'scale_pos_weight': 1.0,
        'random_state': RANDOM_STATE,
        'early_stopping_rounds': 50
    }
    
    @classmethod
    def update(cls, **kwargs):
        """Update configuration values"""
        for key, value in kwargs.items():
            if hasattr(cls, key):
                setattr(cls, key, value)
            else:
                raise ValueError(f"Unknown config key: {key}")
