#!/usr/bin/env python3
"""
Make predictions with trained bot detection model
"""

import os
import sys
import argparse
import pandas as pd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_json_any, users_to_df
from src.features import BERTEmbedder, StructuralFeatureExtractor
from src.models import BotDetectorTrainer
from src.utils import Config


def load_model(model_path=None):
    """Load trained model"""
    if model_path is None:
        model_path = Config.MODEL_PATH
    
    print(f"Loading model from {model_path}...")
    trainer = BotDetectorTrainer.load(model_path)
    return trainer


def prepare_user_data(user_json_path, tweets_list=None):
    """
    Prepare user data for prediction.
    
    Args:
        user_json_path: Path to user JSON file
        tweets_list: Optional list of tweets for the user
        
    Returns:
        Feature DataFrame ready for prediction
    """
    print(f"\nLoading user data from {user_json_path}...")
    user_obj = load_json_any(user_json_path)
    users = users_to_df(user_obj)
    
    # Extract structural features
    print("Extracting structural features...")
    extractor = StructuralFeatureExtractor()
    users = extractor.extract_user_features(users)
    
    # Add tweet data if provided
    if tweets_list is not None:
        users['tweets_list'] = [tweets_list] * len(users)
    elif 'tweets_list' not in users.columns:
        users['tweets_list'] = [[] for _ in range(len(users))]
    
    # Initialize embedder
    embedder = BERTEmbedder(
        model_name=Config.BERT_MODEL,
        device=Config.get_device(),
        max_length=Config.MAX_LENGTH,
        batch_size=Config.BATCH_SIZE
    )
    
    # Bio embeddings
    print("Generating bio embeddings...")
    bio_df = embedder.encode_user_bios(users)
    
    # Tweet embeddings
    print("Generating tweet embeddings...")
    tweet_df = embedder.encode_user_tweets(users, tweets_column='tweets_list')
    
    # Structural features
    X_struct = extractor.prepare_structural_features(users)
    
    # Combine features
    X = pd.concat([X_struct, bio_df, tweet_df], axis=1)
    X.columns = X.columns.astype(str).str.replace(
        r'[\\",\\[\\]<>]', '_', regex=True
    )
    
    return X, users


def predict(model, X, users):
    """Make predictions"""
    print("\nMaking predictions...")
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Create results DataFrame
    results = pd.DataFrame({
        'user_id': users['user_id_str'].values,
        'username': users['username'].values if 'username' in users.columns 
                    else [''] * len(users),
        'prediction': ['bot' if p == 1 else 'human' for p in predictions],
        'bot_probability': probabilities,
        'confidence': [max(p, 1-p) for p in probabilities]
    })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Predict bot/human for Twitter users'
    )
    parser.add_argument(
        'user_json',
        help='Path to user JSON file'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Path to trained model (default: use Config.MODEL_PATH)'
    )
    parser.add_argument(
        '--output',
        '-o',
        default='predictions.csv',
        help='Output CSV file for predictions'
    )
    
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Prepare data
    X, users = prepare_user_data(args.user_json)
    
    # Make predictions
    results = predict(model, X, users)
    
    # Save results
    print(f"\nSaving predictions to {args.output}...")
    results.to_csv(args.output, index=False)
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"\nTotal users: {len(results)}")
    print(f"Predicted bots: {(results['prediction'] == 'bot').sum()}")
    print(f"Predicted humans: {(results['prediction'] == 'human').sum()}")
    print(f"\nAverage bot probability: {results['bot_probability'].mean():.3f}")
    print(f"Average confidence: {results['confidence'].mean():.3f}")
    
    print("\nTop 5 most likely bots:")
    print(results.nlargest(5, 'bot_probability')[
        ['user_id', 'username', 'bot_probability']
    ].to_string(index=False))
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
