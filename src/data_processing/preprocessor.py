"""Tweet preprocessing and aggregation"""

import ijson
import json
import os
from collections import defaultdict
from tqdm import tqdm
import pandas as pd
import numpy as np


class TweetPreprocessor:
    """Handles tweet preprocessing and extraction of relevant fields"""
    
    def __init__(self, input_dir="./data", output_dir="./data_processed"):
        """
        Initialize the preprocessor.
        
        Args:
            input_dir: Directory containing raw tweet JSON files
            output_dir: Directory for processed output files
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def process_tweet_file(self, file_index):
        """
        Process a single tweet file.
        
        Args:
            file_index: Index of the file (e.g., 0 for tweet_0.json)
        """
        input_path = os.path.join(self.input_dir, f"tweet_{file_index}.json")
        output_path = os.path.join(self.output_dir, f"tweet_{file_index}_processed.json")
        
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return
        
        tweet_dict = {}
        with open(input_path, 'rb') as f:
            parser = ijson.items(f, 'item')
            for tweet in tqdm(parser, desc=f"Processing tweet_{file_index}.json"):
                try:
                    tweet_id = tweet.get('id')
                    author_id = tweet.get('author_id')
                    text = tweet.get('text', '')
                    metrics = tweet.get('public_metrics', {})
                    
                    tweet_dict[tweet_id] = {
                        'author_id': author_id,
                        'text': text,
                        'like_count': metrics.get('like_count', 0),
                        'retweet_count': metrics.get('retweet_count', 0),
                        'reply_count': metrics.get('reply_count', 0),
                        'quote_count': metrics.get('quote_count', 0)
                    }
                except Exception as e:
                    print(f"Error processing tweet: {e}")
        
        with open(output_path, 'w', encoding='utf-8') as out_file:
            json.dump(tweet_dict, out_file, ensure_ascii=False, indent=2)
        
        print(f"Processed {len(tweet_dict)} tweets -> {output_path}")
    
    def process_all_files(self, num_files=8):
        """
        Process all tweet files.
        
        Args:
            num_files: Number of tweet files to process
        """
        for i in range(num_files):
            self.process_tweet_file(i)
    
    @staticmethod
    def aggregate_user_tweets(tweets_df, max_tweets_per_user=20):
        """
        Aggregate tweets by user and calculate engagement metrics.
        
        Args:
            tweets_df: DataFrame with tweet data
            max_tweets_per_user: Maximum number of tweets to keep per user
            
        Returns:
            Tuple of (text_df, engagement_df) DataFrames
        """
        user_texts = defaultdict(list)
        user_stats = defaultdict(lambda: {
            'like_sum': 0.0,
            'retweet_sum': 0.0,
            'reply_sum': 0.0,
            'quote_sum': 0.0,
            'count': 0,
            'unique_texts': set()
        })
        
        for row in tqdm(tweets_df.itertuples(index=False), 
                       total=len(tweets_df), 
                       desc="Aggregating tweets by user"):
            try:
                author_id = int(row.author_id)
                text = str(row.text) if pd.notna(row.text) else ""
                
                # Limit tweets per user
                if len(user_texts[author_id]) < max_tweets_per_user:
                    user_texts[author_id].append(text)
                
                stats = user_stats[author_id]
                stats['count'] += 1
                stats['unique_texts'].add(text)
                
                # Accumulate metrics
                for metric in ['like', 'retweet', 'reply', 'quote']:
                    val = pd.to_numeric(getattr(row, f'{metric}_count'), errors='coerce')
                    stats[f'{metric}_sum'] += 0.0 if pd.isna(val) else val
                    
            except (TypeError, ValueError):
                continue
        
        # Create text aggregation DataFrame
        agg_text = pd.DataFrame([
            {'author_id': author_id, 'tweets_list': texts}
            for author_id, texts in user_texts.items()
        ])
        
        # Create engagement metrics DataFrame
        eng_data = []
        for author_id, stats in user_stats.items():
            count = stats['count']
            if count == 0:
                continue
            
            eng_data.append({
                'author_id': author_id,
                'avg_like': stats['like_sum'] / count,
                'avg_retweet': stats['retweet_sum'] / count,
                'avg_reply': stats['reply_sum'] / count,
                'avg_quote': stats['quote_sum'] / count,
                'n_tweets': count,
                'unique_text_ratio': len(stats['unique_texts']) / count
            })
        
        eng_df = pd.DataFrame(eng_data)
        
        return agg_text, eng_df
