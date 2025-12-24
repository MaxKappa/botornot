"""Data loading utilities for JSON files"""

import json
import pandas as pd
import numpy as np


def load_json_any(path):
    """
    Load JSON from a file, handling both standard JSON and newline-delimited JSON.
    
    Args:
        path: Path to the JSON file
        
    Returns:
        Parsed JSON object (dict or list)
    """
    with open(path, 'r', encoding='utf-8') as f:
        txt = f.read().strip()
    
    try:
        obj = json.loads(txt)
        return obj
    except json.JSONDecodeError:
        rows = []
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
        return rows


def tweets_to_df(tweet_obj):
    """
    Convert tweet data (dict or list) to a pandas DataFrame.
    
    Args:
        tweet_obj: Tweet data as dict or list
        
    Returns:
        pandas DataFrame with tweet data
    """
    if isinstance(tweet_obj, dict):
        items = []
        for tw_id, tw in tweet_obj.items():
            rec = {'tweet_id': tw_id}
            rec.update(tw)
            items.append(rec)
        return pd.DataFrame(items)
    elif isinstance(tweet_obj, list):
        return pd.DataFrame(tweet_obj)
    else:
        raise ValueError("tweet_obj must be dict or list")


def users_to_df(user_obj):
    """
    Convert user data (dict or list) to a pandas DataFrame.
    
    Args:
        user_obj: User data as dict or list
        
    Returns:
        pandas DataFrame with user data
    """
    if isinstance(user_obj, dict):
        items = []
        for uid, u in user_obj.items():
            if isinstance(u, dict):
                rec = {'_key': uid}
                rec.update(u)
                items.append(rec)
        if items:
            return pd.DataFrame(items)
        else:
            return pd.DataFrame([user_obj])
    elif isinstance(user_obj, list):
        return pd.DataFrame(user_obj)
    else:
        return pd.DataFrame([user_obj])


def extract_numeric_from_user_id(uid):
    """
    Extract numeric ID from user ID string.
    
    Args:
        uid: User ID (may contain non-numeric characters)
        
    Returns:
        Numeric user ID or NaN
    """
    if pd.isna(uid):
        return np.nan
    import re
    m = re.search(r'(\d+)', str(uid))
    return int(m.group(1)) if m else np.nan
