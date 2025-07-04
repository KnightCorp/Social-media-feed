import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np

def compare_top_posts(ml_top_ids, algo_top_ids, ml_label='ML Model', algo_label='Algorithm'):
    """
    Compare and visualize overlap between two sets of top post IDs.
    """
    print(f"{ml_label} Top 10 Post IDs:", ml_top_ids)
    print(f"{algo_label} Top 10 Post IDs:", algo_top_ids)
    print("Overlap:", ml_top_ids & algo_top_ids)
    print(f"Unique to {ml_label}:", ml_top_ids - algo_top_ids)
    print(f"Unique to {algo_label}:", algo_top_ids - ml_top_ids)

    plt.figure(figsize=(6, 4))
    venn2([ml_top_ids, algo_top_ids], set_labels=(ml_label, algo_label))
    plt.title('Top 10 Recommendations Overlap')
    plt.show()

def compare_multiple_models(top_sets: dict):
    """
    Compare and visualize overlap between three models' top post IDs.
    top_sets: dict with keys as model names and values as sets of top post IDs.
    """
    if len(top_sets) != 3:
        raise ValueError("This function supports exactly three sets for venn3.")

    labels = list(top_sets.keys())
    sets = list(top_sets.values())
    print(f"{labels[0]} Top 10:", sets[0])
    print(f"{labels[1]} Top 10:", sets[1])
    print(f"{labels[2]} Top 10:", sets[2])

    plt.figure(figsize=(8, 6))
    venn3(sets, set_labels=labels)
    plt.title('Top 10 Recommendations Overlap')
    plt.show()

def plot_xgb_importance(xgb_model, max_num_features=10):
    """
    Plot XGBoost feature importance.
    """
    import xgboost as xgb
    xgb.plot_importance(xgb_model, max_num_features=max_num_features)
    plt.title('XGBoost Feature Importance')
    plt.show()

def plot_shap_summary(xgb_model, X_test, feature_names):
    """
    Plot SHAP summary for XGBoost model.
    """
    import shap
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, feature_names=feature_names)
