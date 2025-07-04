import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import logging

def statistical_summary(df: pd.DataFrame):
    """Print and return the statistical summary of numeric features."""
    summary = df.describe()
    print("Statistical summary of numeric features:")
    print(summary)
    return summary

def plot_engagement_and_recency(df: pd.DataFrame, save_path=None):
    """Plot engagement score and hours since post distributions."""
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.histplot(df['engagement_score'], bins=30, kde=True)
    plt.title('Engagement Score Distribution')
    plt.subplot(1, 2, 2)
    sns.histplot(df['hours_since_post'], bins=30, kde=True)
    plt.title('Hours Since Post Distribution')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/engagement_recency.png")
    plt.show()

def plot_correlation_heatmap(df: pd.DataFrame, save_path=None):
    """Plot correlation heatmap of key features."""
    plt.figure(figsize=(8, 6))
    corr_cols = ['engagement_score', 'likes', 'comments', 'shares', 'avg_time_spent', 'content_length']
    corr_cols = [col for col in corr_cols if col in df.columns]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}/correlation_heatmap.png")
    plt.show()

def plot_top_topics(df: pd.DataFrame, save_path=None):
    """Plot bar chart of top 10 topics."""
    if 'top_topic' in df.columns:
        top_topics = df['top_topic'].value_counts().head(10)
        plt.figure(figsize=(8, 4))
        sns.barplot(x=top_topics.values, y=top_topics.index)
        plt.title('Top 10 Topics')
        plt.xlabel('Number of Posts')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/top_topics.png")
        plt.show()
    else:
        logging.warning("Column 'top_topic' not found in DataFrame.")

def plot_sentiment_distribution(df: pd.DataFrame, save_path=None):
    """Plot count plot of sentiment categories."""
    if 'sentiment_category' in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x='sentiment_category', data=df)
        plt.title('Sentiment Category Distribution')
        plt.tight_layout()
        if save_path:
            plt.savefig(f"{save_path}/sentiment_distribution.png")
        plt.show()
    else:
        logging.warning("Column 'sentiment_category' not found in DataFrame.")

def run_eda(df: pd.DataFrame, save_path=None):
    """Run all EDA steps."""
    statistical_summary(df)
    plot_engagement_and_recency(df, save_path)
    plot_correlation_heatmap(df, save_path)
    plot_top_topics(df, save_path)
    plot_sentiment_distribution(df, save_path)
