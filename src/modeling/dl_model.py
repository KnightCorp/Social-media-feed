import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import matplotlib.pyplot as plt

def train_keras_mlp(X_train, y_train, X_test, y_test, feature_names=None, epochs=20, batch_size=32, plot_history=True):
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build model
    model = keras.Sequential([
        keras.layers.Input(shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # Train
    history = model.fit(
        X_train_scaled, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.2,
        verbose=2
    )

    # Evaluate
    y_pred_proba = model.predict(X_test_scaled).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)

    print("Keras MLP Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Keras MLP ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

    # Plot training history
    if plot_history:
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['AUC'], label='Train AUC')
        plt.plot(history.history['val_AUC'], label='Val AUC')
        plt.title('MLP Training History (AUC)')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.show()

    return model, scaler, history
