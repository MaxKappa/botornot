"""Structural feature extraction from user profiles"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone
from dateutil import parser as dtparser
from tqdm.auto import tqdm


class StructuralFeatureExtractor:
    """Extract structural features from user data"""
    
    def __init__(self):
        """Initialize the feature extractor"""
        pass
    
    @staticmethod
    def parse_datetime_safe(x):
        """Parse datetime string safely, returning NaT on error"""
        if pd.isna(x):
            return pd.NaT
        try:
            return dtparser.parse(str(x))
        except Exception:
            return pd.NaT
    
    @staticmethod
    def is_default_profile_img(url):
        """Check if profile image URL is a default image"""
        if not isinstance(url, str) or not url:
            return True
        return ('default_profile_images' in url) or ('default_profile' in url)
    
    def extract_user_features(self, users_df):
        """
        Extract all structural features from user DataFrame.
        
        Args:
            users_df: DataFrame with user data
            
        Returns:
            DataFrame with additional feature columns
        """
        df = users_df.copy()
        
        # Extract user ID numeric
        if 'id' not in df.columns:
            if '_key' in df.columns:
                df['id'] = df['_key']
            else:
                df['id'] = None
        
        df['user_id_str'] = df['id'].astype(str)
        df['user_id_num'] = df['user_id_str'].apply(self._extract_numeric_id)
        
        # Expand public metrics
        if 'public_metrics' in df.columns:
            pm = df['public_metrics'].apply(
                lambda x: x if isinstance(x, dict) else {}
            )
            pm_df = pd.json_normalize(pm)
            pm_df.columns = [f'pm.{c}' for c in pm_df.columns]
            
            df = df.drop(columns=['public_metrics'])
            df = pd.concat([df, pm_df], axis=1)
        
        # Parse creation date and calculate account age
        if 'created_at' in df.columns:
            tqdm.pandas(desc="Parsing creation dates")
            df['created_at_dt'] = df['created_at'].progress_apply(
                self.parse_datetime_safe
            )
            now = datetime.now(timezone.utc)
            df['account_age_days'] = (now - df['created_at_dt']).dt.days
        else:
            df['account_age_days'] = np.nan
        
        # Detect default profile image
        if 'profile_image_url' in df.columns:
            tqdm.pandas(desc="Checking profile images")
            df['default_profile_image'] = df['profile_image_url'].progress_apply(
                self.is_default_profile_img
            )
        else:
            df['default_profile_image'] = pd.Series([np.nan] * len(df))
        
        # Ensure description column exists
        if 'description' not in df.columns:
            df['description'] = ""
        
        return df
    
    @staticmethod
    def _extract_numeric_id(uid):
        """Extract numeric ID from user ID string"""
        if pd.isna(uid):
            return np.nan
        import re
        m = re.search(r'(\d+)', str(uid))
        return int(m.group(1)) if m else np.nan
    
    @staticmethod
    def get_feature_columns():
        """
        Get list of structural feature columns to use for modeling.
        
        Returns:
            List of column names
        """
        return [
            'pm.followers_count',
            'pm.following_count',
            'pm.listed_count',
            'pm.tweet_count',
            'account_age_days',
            'default_profile_image',
            'protected',
            'verified',  # may not always be present
            'avg_like',
            'avg_retweet',
            'avg_reply',
            'avg_quote',
            'n_tweets',
            'unique_text_ratio'
        ]
    
    @staticmethod
    def prepare_structural_features(data_df, feature_cols=None):
        """
        Prepare structural features for modeling.
        
        Args:
            data_df: DataFrame with user data
            feature_cols: List of feature columns (or None for default)
            
        Returns:
            DataFrame with cleaned structural features
        """
        if feature_cols is None:
            feature_cols = StructuralFeatureExtractor.get_feature_columns()
        
        # Filter to existing columns
        existing_cols = [c for c in feature_cols if c in data_df.columns]
        X_struct = data_df[existing_cols].copy()
        
        # Convert boolean columns to int
        for col in ['default_profile_image', 'protected', 'verified']:
            if col in X_struct.columns:
                X_struct[col] = X_struct[col].astype(int)
        
        # Fill missing values
        X_struct = X_struct.fillna(0.0)
        
        return X_struct
