import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def build_dnn(input_dim):
    model = Sequential([
        Dense(120, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.1),
        Dense(80, activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(40, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005, amsgrad=True), loss='mae')
    return model

def main():
    print("Steps 9-13: Model Training, Evaluation & Saving")
    X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train_pca.csv")).values
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).values.flatten()
    X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test_pca.csv")).values
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).values.flatten()
    
    print("Training DNN...")
    dnn = build_dnn(X_train.shape[1])
    es = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
    dnn.fit(X_train, y_train, validation_split=0.1, epochs=300, batch_size=32, callbacks=[es], verbose=0)
    dnn.save(os.path.join(MODELS_DIR, "dnn_model.keras"))
    
    print("Training SVR...")
    svr = SVR(kernel='rbf', C=10.0, gamma='scale')
    svr.fit(X_train, y_train)
    joblib.dump(svr, os.path.join(MODELS_DIR, "svr_model.joblib"))
    
    print("Training KRR...")
    krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.01)
    krr.fit(X_train, y_train)
    joblib.dump(krr, os.path.join(MODELS_DIR, "krr_model.joblib"))
    
    print("Evaluating models...")
    predictions = {
        "DNN": dnn.predict(X_test, verbose=0).flatten(),
        "SVR": svr.predict(X_test),
        "KRR": krr.predict(X_test)
    }
    
    results = []
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    min_v = min(y_test.min(), min([p.min() for p in predictions.values()])) - 0.5
    max_v = max(y_test.max(), max([p.max() for p in predictions.values()])) + 0.5
    
    for i, (name, y_pred) in enumerate(predictions.items()):
        mae = mean_absolute_error(y_test, y_pred)
        rmse = root_mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results.append({"Model": name, "MAE_V": mae, "RMSE_V": rmse, "R2_Score": r2})
        
        ax = axes[i]
        ax.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
        ax.plot([min_v, max_v], [min_v, max_v], 'k--')
        ax.plot([min_v, max_v], [min_v+0.5, max_v+0.5], 'gray', linestyle=':')
        ax.plot([min_v, max_v], [min_v-0.5, max_v-0.5], 'gray', linestyle=':')
        ax.set_title(f"{name} (MAE = {mae:.3f} V)")
        ax.set_xlabel("V_DFT (V)")
        if i == 0: ax.set_ylabel("V_ML (V)")
        ax.set_xlim(min_v, max_v)
        ax.set_ylim(min_v, max_v)
        
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "parity_plots.png"), dpi=300)
    plt.close()
    
    pd.DataFrame(results).to_csv(os.path.join(RESULTS_DIR, "benchmark_metrics.csv"), index=False)
    print("Evaluation complete. Parity plots and metrics saved.")

if __name__ == '__main__':
    main()
