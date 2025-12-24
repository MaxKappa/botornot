"""Label loading and normalization"""

import os
import pandas as pd
import numpy as np
from ..data_processing.loader import load_json_any


class LabelLoader:
    """Load and normalize bot/human labels from various sources"""
    
    def __init__(self, data_dir):
        """
        Initialize the label loader.
        
        Args:
            data_dir: Directory containing label files
        """
        self.data_dir = data_dir
    
    @staticmethod
    def normalize_label(x):
        """
        Normalize label to binary (0=human, 1=bot).
        
        Args:
            x: Label value (various formats)
            
        Returns:
            0, 1, or np.nan
        """
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return np.nan
        
        xs = str(x).strip().lower()
        
        # Bot labels
        if xs in {'bot', '1', 'true', 'yes', 'fake', 'ai', 'automated'}:
            return 1
        
        # Human labels
        if xs in {'human', '0', 'false', 'no', 'real', 'genuine'}:
            return 0
        
        return np.nan
    
    def load_labels(self, users_df):
        """
        Load labels from various sources (CSV, JSON, or user DataFrame).
        
        Tries in order:
        1. label.csv in data_dir
        2. 'label' column in users_df
        3. label.json in data_dir
        4. ground_truth.json in data_dir
        
        Args:
            users_df: DataFrame with user data (must have 'user_id_str' column)
            
        Returns:
            DataFrame with columns ['user_id_str', 'y'] or None if not found
        """
        labels = None
        
        # Try CSV file
        lbl_csv_path = os.path.join(self.data_dir, 'label.csv')
        if os.path.exists(lbl_csv_path):
            try:
                tmp = pd.read_csv(
                    lbl_csv_path,
                    header=None,
                    names=['user_id_str', 'label']
                )
                tmp['y'] = tmp['label'].apply(self.normalize_label)
                labels = tmp[['user_id_str', 'y']]
                print(f"Loaded labels from {lbl_csv_path}")
                return labels
            except Exception as e:
                print(f"Error reading {lbl_csv_path}: {e}")
        
        # Try label column in users DataFrame
        if labels is None and 'label' in users_df.columns:
            tmp = users_df[['user_id_str', 'label']].copy()
            tmp['y'] = tmp['label'].apply(self.normalize_label)
            labels = tmp[['user_id_str', 'y']]
            print("Loaded labels from 'label' column in users DataFrame")
            return labels
        
        # Try label.json
        if labels is None:
            lbl_path = os.path.join(self.data_dir, 'label.json')
            if os.path.exists(lbl_path):
                lbl_obj = load_json_any(lbl_path)
                
                if isinstance(lbl_obj, dict):
                    tmp = pd.DataFrame({
                        'user_id_str': list(lbl_obj.keys()),
                        'label': list(lbl_obj.values())
                    })
                elif isinstance(lbl_obj, list):
                    tmp = pd.DataFrame(lbl_obj)
                    if 'user_id' in tmp.columns and 'label' in tmp.columns:
                        tmp = tmp.rename(columns={'user_id': 'user_id_str'})
                else:
                    tmp = pd.DataFrame()
                
                if len(tmp) > 0:
                    tmp['y'] = tmp['label'].apply(self.normalize_label)
                    labels = tmp[['user_id_str', 'y']]
                    print(f"Loaded labels from {lbl_path}")
                    return labels
        
        # Try ground_truth.json
        if labels is None:
            gt_path = os.path.join(self.data_dir, 'ground_truth.json')
            if os.path.exists(gt_path):
                gt_obj = load_json_any(gt_path)
                if isinstance(gt_obj, dict):
                    tmp = pd.DataFrame({
                        'user_id_str': list(gt_obj.keys()),
                        'label': list(gt_obj.values())
                    })
                    tmp['y'] = tmp['label'].apply(self.normalize_label)
                    labels = tmp[['user_id_str', 'y']]
                    print(f"Loaded labels from {gt_path}")
                    return labels
        
        print("Warning: No labels found")
        return None
    
    def merge_labels(self, data_df, users_df):
        """
        Load labels and merge with data.
        
        Args:
            data_df: DataFrame to merge labels into
            users_df: DataFrame with user data (for loading labels)
            
        Returns:
            DataFrame with 'y' column, filtered to labeled examples
        """
        labels = self.load_labels(users_df)
        
        if labels is None:
            print("No labels available - returning data without labels")
            return data_df
        
        # Merge on user_id_str
        result = data_df.merge(labels, on='user_id_str', how='inner')
        
        # Filter out invalid labels
        result = result[~result['y'].isna()].reset_index(drop=True)
        
        if len(result) > 0:
            print("\nLabel distribution:")
            print(result['y'].value_counts(normalize=True))
        
        return result
