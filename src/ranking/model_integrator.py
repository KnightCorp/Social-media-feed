import numpy as np
import pandas as pd

def train_and_rank_pipeline(df, feature_cols, model, scaler, user_id=None, top_n=10, score_col='dl_score'):
    """
    Trains model, predicts scores, ranks posts, and saves top N results.
    """
    X = df[feature_cols]
    X_scaled = scaler.transform(X)
    df[score_col] = model.predict(X_scaled).flatten()
    ranked = df.sort_values(score_col, ascending=False).head(top_n)
    print(f"\nTop {top_n} Posts for User by Deep Learning Model:")
    print(ranked[['post_id', 'creator_id', score_col, 'topics', 'timestamp']])
    ranked.to_csv(f'ranked_feed_{score_col}.csv', index=False)
    print("Feed ranking complete and results saved.")
    return ranked

def calculate_social_metrics(df, top_n=10, score_col='dl_score'):
    """
    Calculate and display key social media metrics for the top N ranked posts.
    """
    top_posts = df.sort_values(score_col, ascending=False).head(top_n)
    total_likes = top_posts['likes'].sum()
    total_comments = top_posts['comments'].sum()
    total_shares = top_posts['shares'].sum()
    avg_engagement = (total_likes + total_comments + total_shares) / top_n
    avg_likes = total_likes / top_n
    avg_comments = total_comments / top_n
    avg_shares = total_shares / top_n

    print(f"Top {top_n} Feed Metrics:")
    print(f"  Total Likes: {total_likes}")
    print(f"  Total Comments: {total_comments}")
    print(f"  Total Shares: {total_shares}")
    print(f"  Average Engagement per Post: {avg_engagement:.2f}")
    print(f"  Average Likes per Post: {avg_likes:.2f}")
    print(f"  Average Comments per Post: {avg_comments:.2f}")
    print(f"  Average Shares per Post: {avg_shares:.2f}")

    import matplotlib.pyplot as plt
    metrics = ['Likes', 'Comments', 'Shares']
    values = [avg_likes, avg_comments, avg_shares]
    plt.bar(metrics, values, color=['#1da1f2', '#17bf63', '#ffad1f'])
    plt.title(f'Average Engagement Metrics (Top {top_n} Posts)')
    plt.ylabel('Average per Post')
    plt.show()
