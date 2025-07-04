import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def train_xgboost_classifier(X_train, y_train, X_test, y_test, feature_names=None, plot_importance=True):
    # Prepare DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)

    # Set parameters
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 5,
        'eta': 0.1,
        'seed': 42,
        'verbosity': 1
    }

    # Train model
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)

    # Predict and evaluate
    y_pred_proba = xgb_model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("XGBoost Classification Report:")
    print(classification_report(y_test, y_pred))
    print("XGBoost ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # Feature importance plot
    if plot_importance:
        ax = xgb.plot_importance(xgb_model, max_num_features=10, height=0.5)
        plt.title('XGBoost Feature Importance')
        plt.show()

    return xgb_model, y_pred, y_pred_proba
