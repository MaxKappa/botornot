"""BERT-based text embedding generation"""

import os
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm


class BERTEmbedder:
    """Generate BERT embeddings for text data"""
    
    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device=None,
        max_length=512,
        batch_size=16
    ):
        """
        Initialize the BERT embedder.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to use ('cuda', 'mps', 'cpu', or None for auto-detect)
            max_length: Maximum sequence length for tokenization
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        
        self.device = device
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        self.embedding_dim = self.model.config.hidden_size
    
    def encode(self, texts):
        """
        Encode a list of texts into embeddings.
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([]).reshape(0, self.embedding_dim)
        
        all_embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get embeddings
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Mean pooling with attention mask
                mask = inputs['attention_mask']
                mask_expanded = mask.unsqueeze(-1).expand(hidden_states.size())
                masked_outputs = hidden_states * mask_expanded
                sum_embeddings = torch.sum(masked_outputs, 1)
                count_safe = torch.clamp(mask.sum(1, keepdim=True), min=1e-9)
                mean_embeddings = sum_embeddings / count_safe
                
                # L2 normalization
                mean_embeddings = torch.nn.functional.normalize(
                    mean_embeddings, p=2, dim=1
                )
                
                all_embeddings.append(mean_embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def encode_user_bios(self, users_df, cache_path=None):
        """
        Encode user biographies/descriptions.
        
        Args:
            users_df: DataFrame with 'description' column
            cache_path: Optional path to cache embeddings
            
        Returns:
            DataFrame with embedding columns
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            try:
                bio_df = pd.read_parquet(cache_path)
                if bio_df.shape[0] == len(users_df) and \
                   bio_df.shape[1] == self.embedding_dim:
                    print(f"Loaded cached bio embeddings from {cache_path}")
                    bio_df.index = users_df.index
                    return bio_df
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Generate embeddings
        bio_texts = users_df['description'].fillna("").tolist()
        print(f"Encoding {len(bio_texts)} user biographies...")
        bio_embeddings = self.encode(bio_texts)
        
        bio_df = pd.DataFrame(
            bio_embeddings,
            index=users_df.index,
            columns=[f'bio_e_{i}' for i in range(self.embedding_dim)]
        )
        
        # Save cache
        if cache_path:
            try:
                bio_df.to_parquet(cache_path, index=True)
                print(f"Cached bio embeddings to {cache_path}")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return bio_df
    
    def encode_user_tweets(self, users_df, tweets_column='tweets_list', 
                          cache_path=None):
        """
        Encode user tweets (averaging multiple tweets per user).
        
        Args:
            users_df: DataFrame with tweets_list column
            tweets_column: Name of column containing list of tweets
            cache_path: Optional path to cache embeddings
            
        Returns:
            DataFrame with embedding columns
        """
        # Check cache
        if cache_path and os.path.exists(cache_path):
            try:
                tweet_df = pd.read_csv(cache_path)
                tweet_df = tweet_df.set_index('original_index')
                if tweet_df.shape[0] == len(users_df) and \
                   tweet_df.shape[1] == self.embedding_dim:
                    print(f"Loaded cached tweet embeddings from {cache_path}")
                    tweet_df.index = users_df.index
                    return tweet_df
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Generate embeddings
        all_embeddings = []
        
        for idx, row in tqdm(users_df.iterrows(), 
                            total=len(users_df),
                            desc="Encoding user tweets"):
            tweet_list = row.get(tweets_column)
            
            # Collect valid texts
            texts_to_embed = []
            if isinstance(tweet_list, list):
                for t in tweet_list:
                    if pd.notna(t) and str(t).strip():
                        texts_to_embed.append(str(t))
            
            # Encode and average
            if not texts_to_embed:
                avg_embedding = np.zeros(self.embedding_dim)
            else:
                user_embeddings = self.encode(texts_to_embed)
                if user_embeddings.shape[0] > 0:
                    avg_embedding = np.mean(user_embeddings, axis=0)
                else:
                    avg_embedding = np.zeros(self.embedding_dim)
            
            all_embeddings.append(avg_embedding)
        
        tweet_df = pd.DataFrame(
            all_embeddings,
            index=users_df.index,
            columns=[f'tweet_e_{i}' for i in range(self.embedding_dim)]
        )
        
        # Save cache
        if cache_path:
            try:
                tweet_df_save = tweet_df.copy()
                tweet_df_save['original_index'] = tweet_df_save.index
                tweet_df_save.to_csv(cache_path, index=False)
                print(f"Cached tweet embeddings to {cache_path}")
            except Exception as e:
                print(f"Error saving cache: {e}")
        
        return tweet_df
