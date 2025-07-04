import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score

def tune_random_forest(X_train, y_train, X_test, y_test, X=None, y=None, cv=5, verbose=2):
    from sklearn.ensemble import RandomForestClassifier

    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    rf = RandomForestClassifier(random_state=42)

    # GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=verbose
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV ROC-AUC Score:", grid_search.best_score_)

    # Evaluate best model on test set
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

    print("Classification Report (Tuned RF):")
    print(classification_report(y_test, y_pred))
    print("ROC-AUC Score (Tuned RF):", roc_auc_score(y_test, y_pred_proba))

    # Cross-validation scores on full data if provided
    if X is not None and y is not None:
        cv_scores = cross_val_score(best_rf, X, y, cv=cv, scoring='roc_auc')
        print("Cross-validated ROC-AUC scores:", cv_scores)
        print("Mean CV ROC-AUC:", np.mean(cv_scores))

    return best_rf, grid_search
