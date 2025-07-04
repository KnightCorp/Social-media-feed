import pandas as pd
import numpy as np
import logging

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Ensure 'timestamp' is datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        # Remove rows with invalid timestamps
        before = len(df)
        df = df[df['timestamp'].notna()].copy()
        after = len(df)
        if before != after:
            logging.info(f"Filtered out {before - after} rows with invalid timestamps.")

    # 2. Engagement Score
    for col in ['likes', 'comments', 'shares']:
        if col not in df.columns:
            df[col] = 0
    df['engagement_score'] = df['likes'] + 2 * df['comments'] + 3 * df['shares']

    # 3. Time Since Post (in hours)
    now = pd.Timestamp.now()
    df['hours_since_post'] = (now - df['timestamp']).dt.total_seconds() / 3600

    # 4. Content Length Category
    if 'content_length' in df.columns:
        df['content_length_category'] = pd.cut(
            df['content_length'],
            bins=[-np.inf, 100, 300, np.inf],
            labels=['short', 'medium', 'long']
        )
    else:
        df['content_length_category'] = 'unknown'

    # 5. Sentiment Category
    if 'sentiment_compound' in df.columns:
        df['sentiment_category'] = pd.cut(
            df['sentiment_compound'],
            bins=[-1, -0.3, 0.3, 1],
            labels=['negative', 'neutral', 'positive']
        )
    else:
        df['sentiment_category'] = 'unknown'

    # 6. Number of Topics
    if 'topics' in df.columns:
        df['num_topics'] = df['topics'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df['num_topics'] = 0

    # 7. Number of Creator Connections
    if 'creator_connections' in df.columns:
        df['num_creator_connections'] = df['creator_connections'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    else:
        df['num_creator_connections'] = 0

    # 8. Top Topic (first topic in list, if available)
    if 'topics' in df.columns:
        df['top_topic'] = df['topics'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'unknown')
    else:
        df['top_topic'] = 'unknown'

    logging.info("Feature engineering complete. New features added:")
    logging.info(df[['engagement_score', 'hours_since_post', 'content_length_category',
                     'sentiment_category', 'num_topics', 'num_creator_connections', 'top_topic']].head(3))

    return df
