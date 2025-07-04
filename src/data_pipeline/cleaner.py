import pandas as pd
import numpy as np
import ast
import logging
from typing import List

def convert_numeric_columns(df: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def drop_missing_timestamps(df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
    if timestamp_col in df.columns:
        missing_ts = df[timestamp_col].isnull().sum()
        if missing_ts > 0:
            df = df[df[timestamp_col].notnull()]
            logging.info(f"Dropped {missing_ts} rows with missing timestamp.")
    return df

def safe_list_eval(x, col_name=None):
    # Skip parsing if x is NaN, empty, or the column name itself
    if pd.isna(x) or (isinstance(x, str) and (x.strip() == '' or (col_name and x.strip() == col_name))):
        return []
    try:
        val = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(val, list):
            return val
        else:
            # Only log at debug level for truly unexpected types
            logging.debug(f"Value in column '{col_name}' is not a list, returning empty list.")
            return []
    except Exception:
        # Only log at debug level for truly unexpected parsing errors
        logging.debug(f"Could not parse value in column '{col_name}', returning empty list.")
        return []

def safe_dict_eval(x, col_name=None):
    if pd.isna(x) or (isinstance(x, str) and (x.strip() == '' or (col_name and x.strip() == col_name))):
        return {}
    try:
        val = ast.literal_eval(x) if isinstance(x, str) else x
        if isinstance(val, dict):
            return val
        else:
            logging.debug(f"Value in column '{col_name}' is not a dict, returning empty dict.")
            return {}
    except Exception:
        logging.debug(f"Could not parse value in column '{col_name}', returning empty dict.")
        return {}

def parse_complex_columns(df: pd.DataFrame, list_cols: List[str], dict_cols: List[str]) -> pd.DataFrame:
    for col in list_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_list_eval(x, col_name=col))
    for col in dict_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: safe_dict_eval(x, col_name=col))
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = [
        'likes', 'comments', 'shares', 'avg_time_spent',
        'sentiment_pos', 'sentiment_neg', 'sentiment_neu', 'sentiment_compound', 'content_length'
    ]
    df = convert_numeric_columns(df, num_cols)
    df = drop_missing_timestamps(df, 'timestamp')

    list_cols = ['creator_connections', 'topics']
    dict_cols = [
        'creator_topic_interests', 'creator_format_preferences',
        'creator_creator_preferences', 'creator_expertise_levels'
    ]
    df = parse_complex_columns(df, list_cols, dict_cols)

    logging.info(f"Data types after cleaning:\n{df.dtypes}")
    logging.info(f"Sample of parsed list/dict columns:\n{df[list_cols + dict_cols].head(3)}")
    logging.info(f"Missing values after cleaning:\n{df[num_cols + ['timestamp']].isnull().sum()}")

    return df
