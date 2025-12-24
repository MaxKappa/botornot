#!/usr/bin/env python3
"""
Train bot detection model on TwiBot-22 dataset
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_json_any, TweetPreprocessor
from src.features import (
    BERTEmbedder, 
    StructuralFeatureExtractor,
    LabelLoader
)
from src.models import BotDetectorTrainer, ModelEvaluator
from src.utils import Config


def load_and_aggregate_data():
    """Load user and tweet data, perform aggregation"""
    print("\n" + "="*60)
    print("STEP 1: Loading and Aggregating Data")
    print("="*60)
    
    # Load users
    print(f"\nLoading users from {Config.USER_JSON}...")
    user_obj = load_json_any(Config.USER_JSON)
    from src.data_processing.loader import users_to_df
    users = users_to_df(user_obj)
    
    # Extract structural features
    print("\nExtracting structural features...")
    feature_extractor = StructuralFeatureExtractor()
    users = feature_extractor.extract_user_features(users)
    
    # Load and aggregate tweets
    print(f"\nLoading tweets from {Config.TWEETS_GLOB}...")
    tweet_files = sorted(glob.glob(Config.TWEETS_GLOB))
    
    all_tweets = []
    for fp in tqdm(tweet_files, desc="Loading tweet files"):
        try:
            tweet_obj = load_json_any(fp)
            from src.data_processing.loader import tweets_to_df
            tweets_df = tweets_to_df(tweet_obj)
            all_tweets.append(tweets_df)
        except Exception as e:
            print(f"Error loading {fp}: {e}")
    
    tweets = pd.concat(all_tweets, ignore_index=True)
    print(f"Loaded {len(tweets)} tweets")
    
    # Aggregate tweets by user
    print("\nAggregating tweets by user...")
    agg_text, eng_df = TweetPreprocessor.aggregate_user_tweets(
        tweets, 
        max_tweets_per_user=Config.N_TWEETS_PER_USER
    )
    
    # Merge everything
    print("\nMerging user data with tweet aggregations...")
    data = users.merge(
        agg_text, 
        left_on='user_id_num', 
        right_on='author_id', 
        how='left'
    ).merge(
        eng_df, 
        left_on='user_id_num', 
        right_on='author_id', 
        how='left',
        suffixes=('', '_eng')
    )
    
    # Clean up
    for c in ['author_id_x', 'author_id_y']:
        if c in data.columns:
            data = data.drop(columns=[c])
    
    # Fill missing engagement metrics
    for c in ['avg_like', 'avg_retweet', 'avg_reply', 'avg_quote', 
              'n_tweets', 'unique_text_ratio']:
        if c in data.columns:
            data[c] = data[c].fillna(0.0)
    
    print(f"\nFinal dataset: {len(data)} users")
    return data, users


def extract_features(data):
    """Extract all features (structural + embeddings)"""
    print("\n" + "="*60)
    print("STEP 2: Feature Extraction")
    print("="*60)
    
    # Structural features
    print("\nPreparing structural features...")
    extractor = StructuralFeatureExtractor()
    X_struct = extractor.prepare_structural_features(data)
    print(f"Structural features: {X_struct.shape}")
    
    # BERT embeddings
    embedder = BERTEmbedder(
        model_name=Config.BERT_MODEL,
        device=Config.get_device(),
        max_length=Config.MAX_LENGTH,
        batch_size=Config.BATCH_SIZE
    )
    
    # Bio embeddings
    print("\nGenerating bio embeddings...")
    bio_df = embedder.encode_user_bios(
        data,
        cache_path=Config.BIO_EMBEDDINGS_CACHE
    )
    print(f"Bio embeddings: {bio_df.shape}")
    
    # Tweet embeddings
    print("\nGenerating tweet embeddings...")
    tweet_df = embedder.encode_user_tweets(
        data,
        tweets_column='tweets_list',
        cache_path=Config.TWEET_EMBEDDINGS_CACHE
    )
    print(f"Tweet embeddings: {tweet_df.shape}")
    
    # Combine all features
    print("\nCombining all features...")
    X_full = pd.concat([X_struct, bio_df, tweet_df], axis=1)
    
    # Clean column names
    X_full.columns = X_full.columns.astype(str).str.replace(
        r'[\\",\\[\\]<>]', '_', regex=True
    )
    
    print(f"Total features: {X_full.shape}")
    return X_full


def load_labels_and_split(data, users):
    """Load labels and split data"""
    print("\n" + "="*60)
    print("STEP 3: Loading Labels and Splitting Data")
    print("="*60)
    
    # Load labels
    label_loader = LabelLoader(Config.DATA_DIR)
    data = label_loader.merge_labels(data, users)
    
    if 'y' not in data.columns:
        raise ValueError("No labels found! Please provide labels.")
    
    # Load split file
    print(f"\nLoading split file from {Config.SPLIT_CSV}...")
    if not os.path.exists(Config.SPLIT_CSV):
        raise FileNotFoundError(
            f"Split file not found: {Config.SPLIT_CSV}\n"
            "Download from: https://github.com/LuoUndergradXJTU/TwiBot-22"
            "/blob/main/data/split.csv"
        )
    
    split_df = pd.read_csv(Config.SPLIT_CSV)
    split_df['id'] = split_df['id'].astype(str)
    
    # Merge split info
    data = data.merge(
        split_df, 
        left_on='user_id_str', 
        right_on='id', 
        how='inner'
    )
    
    print(f"\nData after filtering to split file: {len(data)} users")
    print("\nSplit distribution:")
    print(data['split'].value_counts())
    
    return data


def train_model(X_train, y_train, X_val, y_val):
    """Train the XGBoost model"""
    print("\n" + "="*60)
    print("STEP 4: Training Model")
    print("="*60)
    
    trainer = BotDetectorTrainer(**Config.XGB_PARAMS)
    trainer.train(X_train, y_train, X_val, y_val, verbose=100)
    
    # Save model
    print(f"\nSaving model to {Config.MODEL_PATH}...")
    trainer.save(Config.MODEL_PATH)
    
    return trainer


def evaluate_model(trainer, X_test, y_test, X_full):
    """Evaluate the model"""
    print("\n" + "="*60)
    print("STEP 5: Model Evaluation")
    print("="*60)
    
    # Get predictions
    y_pred = trainer.predict(X_test)
    y_pred_proba = trainer.predict_proba(X_test)
    
    # Calculate metrics
    metrics = ModelEvaluator.evaluate(y_test, y_pred, y_pred_proba)
    
    # Print report
    ModelEvaluator.print_evaluation_report(metrics)
    
    # Visualizations
    print("\nGenerating visualizations...")
    
    print("\n1. Confusion Matrix")
    ModelEvaluator.plot_confusion_matrix(metrics['confusion_matrix'])
    ModelEvaluator.plot_confusion_matrix(
        metrics['confusion_matrix'], 
        normalize=True
    )
    
    print("\n2. ROC Curve")
    ModelEvaluator.plot_roc_curve(y_test, y_pred_proba)
    
    print("\n3. Precision-Recall Curve")
    ModelEvaluator.plot_precision_recall_curve(y_test, y_pred_proba)
    
    print("\n4. Calibration Curve")
    ModelEvaluator.plot_calibration_curve(y_test, y_pred_proba)
    
    print("\n5. Feature Importance")
    ModelEvaluator.plot_feature_importance(
        trainer.model, 
        list(X_full.columns),
        top_n=30
    )


def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("BOT DETECTION MODEL TRAINING")
    print("="*60)
    print(f"Device: {Config.get_device()}")
    print(f"Random seed: {Config.RANDOM_STATE}")
    
    # Load and aggregate data
    data, users = load_and_aggregate_data()
    
    # Extract features
    X_full = extract_features(data)
    
    # Load labels and split
    data = load_labels_and_split(data, users)
    
    # Prepare train/val/test splits
    train_mask = data['split'] == 'train'
    val_mask = data['split'] == 'val'
    test_mask = data['split'] == 'test'
    
    X_train = X_full.loc[train_mask]
    y_train = data.loc[train_mask, 'y'].astype(int)
    
    X_val = X_full.loc[val_mask]
    y_val = data.loc[val_mask, 'y'].astype(int)
    
    X_test = X_full.loc[test_mask]
    y_test = data.loc[test_mask, 'y'].astype(int)
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Val set:   {len(X_val)} samples")
    print(f"Test set:  {len(X_test)} samples")
    
    # Train model
    trainer = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate model
    evaluate_model(trainer, X_test, y_test, X_full)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)


if __name__ == "__main__":
    main()
