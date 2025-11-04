import ijson
import json
from tqdm import tqdm

for i in range(0, 8):
    input_path = f"./data/tweet_{i}.json"
    output_path = f"./data_processed/tweet_{i}_processed.json"

    tweet_dict = {}
    with open(input_path, 'rb') as f:
        parser = ijson.items(f, 'item')
        for tweet in tqdm(parser, desc=f"Processing tweet_{i}.json"):
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
