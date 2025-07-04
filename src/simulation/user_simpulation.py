import random
import numpy as np

def simulate_user_engagement(df, top_post_ids, engagement_col='engagement_score'):
    """
    Simulate user engagement on a set of posts.
    Returns the number of posts engaged with (clicked/liked/etc).
    """
    engaged = []
    max_score = df[engagement_col].max()
    for pid in top_post_ids:
        score = df.loc[df['post_id'] == pid, engagement_col].values[0]
        norm_score = min(score / max_score, 1)
        engaged.append(random.random() < norm_score)
    return sum(engaged)

def batch_simulate_engagement(df, top_post_ids_dict, n_users=1000, engagement_col='engagement_score'):
    """
    Simulate engagement for multiple models over many users.
    Returns a dict of total engagements for each model.
    """
    results = {model: 0 for model in top_post_ids_dict}
    for _ in range(n_users):
        for model, top_ids in top_post_ids_dict.items():
            results[model] += simulate_user_engagement(df, top_ids, engagement_col)
    return results

def plot_simulation_results(results, n_users, title='Simulated A/B Test: Model Impact on Engagement'):
    """
    Plot average engaged posts per user for each model.
    """
    import matplotlib.pyplot as plt
    models = list(results.keys())
    avg_engagement = [v / n_users for v in results.values()]
    plt.bar(models, avg_engagement, color=['blue', 'orange', 'green'][:len(models)])
    plt.ylabel('Avg. Engaged Posts per User')
    plt.title(title)
    plt.show()
