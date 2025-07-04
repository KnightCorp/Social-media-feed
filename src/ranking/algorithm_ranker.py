import pandas as pd
import numpy as np
import datetime
import logging
from typing import List, Dict, Any, Optional

class Post:
    def __init__(
        self,
        post_id: str,
        creator_id: str,
        content: str,
        timestamp: datetime.datetime,
        likes: float = 0,
        comments: float = 0,
        shares: float = 0,
        avg_time_spent: float = 0,
        topics: Optional[List[str]] = None,
        content_format: str = "text",
        content_length: float = 0,
        location: Optional[Any] = None,
        sentiment_scores: Optional[Dict[str, float]] = None
    ):
        self.post_id = post_id
        self.creator_id = creator_id
        self.content = content
        self.timestamp = timestamp
        self.likes = likes
        self.comments = comments
        self.shares = shares
        self.avg_time_spent = avg_time_spent
        self.topics = topics or []
        self.content_format = content_format
        self.content_length = content_length
        self.location = location
        self.sentiment_scores = sentiment_scores or {}

class User:
    def __init__(
        self,
        user_id: str,
        connections: Optional[List[str]] = None,
        interaction_history: Optional[Dict[str, int]] = None,
        topic_interests: Optional[Dict[str, float]] = None,
        format_preferences: Optional[Dict[str, float]] = None,
        creator_preferences: Optional[Dict[str, float]] = None,
        location: Optional[Any] = None,
        expertise_levels: Optional[Dict[str, float]] = None
    ):
        self.user_id = user_id
        self.connections = connections or []
        self.interaction_history = interaction_history or {}
        self.topic_interests = topic_interests or {}
        self.format_preferences = format_preferences or {}
        self.creator_preferences = creator_preferences or {}
        self.location = location
        self.expertise_levels = expertise_levels or {}

class ChronologicalAlgorithm:
    def rank_posts(self, posts: List[Post], user: User, **kwargs) -> List[Post]:
        return sorted(posts, key=lambda p: p.timestamp, reverse=True)

class EngagementBasedAlgorithm:
    def rank_posts(self, posts: List[Post], user: User, **kwargs) -> List[Post]:
        return sorted(
            posts,
            key=lambda p: (p.likes + 2 * p.comments + 3 * p.shares),
            reverse=True
        )

def parse_list(val):
    if isinstance(val, list):
        return val
    if pd.isna(val) or val == '' or val is None:
        return []
    try:
        import ast
        return ast.literal_eval(val)
    except Exception:
        return []

def parse_dict(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val) or val == '' or val is None:
        return {}
    try:
        import ast
        return ast.literal_eval(val)
    except Exception:
        return {}

def parse_location(val):
    if isinstance(val, tuple):
        return val
    if pd.isna(val) or val == '' or val is None:
        return None
    try:
        import ast
        return tuple(ast.literal_eval(val))
    except Exception:
        return None

def parse_sentiment(val):
    if isinstance(val, dict):
        return val
    if pd.isna(val) or val == '' or val is None:
        return {}
    try:
        import ast
        return ast.literal_eval(val)
    except Exception:
        return {}

def df_to_posts(df: pd.DataFrame) -> List[Post]:
    posts = []
    for _, row in df.iterrows():
        posts.append(Post(
            post_id=row['post_id'],
            creator_id=row['creator_id'],
            content=row.get('content', ''),
            timestamp=row['timestamp'],
            likes=row.get('likes', 0) or 0,
            comments=row.get('comments', 0) or 0,
            shares=row.get('shares', 0) or 0,
            avg_time_spent=row.get('avg_time_spent', 0) or 0,
            topics=parse_list(row.get('topics', [])),
            content_format=row.get('content_format', 'text'),
            content_length=row.get('content_length', 0) or 0,
            location=parse_location(row.get('location', None)),
            sentiment_scores=parse_sentiment(row.get('sentiment_scores', None))
        ))
    return posts

def run_all_algorithms(posts: List[Post], user: User, top_n: int = 10) -> Dict[str, List[Post]]:
    algorithms = {
        "Chronological": ChronologicalAlgorithm(),
        "Engagement-Based": EngagementBasedAlgorithm(),
        # Add more algorithms as you implement them
    }
    results = {}
    for name, algo in algorithms.items():
        try:
            ranked = algo.rank_posts(posts, user)
            results[name] = ranked[:top_n]
        except Exception as e:
            logging.warning(f"Algorithm {name} failed: {e}")
            results[name] = []
    return results

def display_algo_results(results: Dict[str, List[Post]], user_id: str):
    from IPython.display import display, HTML
    for name, ranked_posts in results.items():
        print(f"\n=== {name} Algorithm: Top Posts for User {user_id} ===")
        data = []
        for i, post in enumerate(ranked_posts, 1):
            data.append([
                i,
                getattr(post, 'post_id', ''),
                getattr(post, 'creator_id', ''),
                f"{getattr(post, 'likes', 0)}+{getattr(post, 'comments', 0)}+{getattr(post, 'shares', 0)}",
                ', '.join(getattr(post, 'topics', [])) if hasattr(post, 'topics') else "",
                getattr(post, 'content_format', ''),
                post.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(post, 'timestamp') else ""
            ])
        df_display = pd.DataFrame(data, columns=["Rank", "Post ID", "Creator", "Engagement", "Topics", "Format", "Time"])
        display(HTML(df_display.to_html(index=False)))
