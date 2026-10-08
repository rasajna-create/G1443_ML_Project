import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -----------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(".")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

NA_SET_FILE = os.path.join(DATA_DIR, "na_set.csv")
K_SET_FILE = os.path.join(DATA_DIR, "k_set.csv")

DNN_PATH = os.path.join(MODELS_DIR, "dnn_model.h5")
SVR_PATH = os.path.join(MODELS_DIR, "svr_model.pkl")
KRR_PATH = os.path.join(MODELS_DIR, "krr_model.pkl")

METADATA_COLS = [
    'Working Ion', 'Battery Formula', 'Formula (Charged State)',
    'Formula (Discharged State)', 'Electrode_data_material_id'
]
TARGET_COL = 'Average Voltage (V)'

# -----------------------------------------------------------------------------
# 2. EVALUATION HELPER
# -----------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    return {
        "MAE_V": round(mean_absolute_error(y_true, y_pred), 4),
        "RMSE_V": round(root_mean_squared_error(y_true, y_pred), 4),
        "R2_Score": round(r2_score(y_true, y_pred), 4)
    }

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - OUT-OF-DISTRIBUTION (OOD) ZERO-SHOT TRANSFER")
    print("=" * 70)

    # 1. Load OOD Data
    print("Loading Sodium (Na-set) and Potassium (K-set) datasets...")
    df_na = pd.read_csv(NA_SET_FILE, low_memory=False)
    df_k = pd.read_csv(K_SET_FILE, low_memory=False)
    
    # Isolate targets and features
    y_na = df_na[TARGET_COL].values
    X_na = df_na.drop(columns=[c for c in METADATA_COLS if c in df_na.columns] + [TARGET_COL]).values
    
    y_k = df_k[TARGET_COL].values
    X_k = df_k.drop(columns=[c for c in METADATA_COLS if c in df_k.columns] + [TARGET_COL]).values

    print(f"Total Na-set samples : {len(df_na)}")
    print(f"Total K-set samples  : {len(df_k)}")

    # 2. Load Models
    print("\nLoading pre-trained models (DNN, SVR, KRR)...")
    dnn_model = load_model(DNN_PATH, compile=False)
    svr_model = joblib.load(SVR_PATH)
    krr_model = joblib.load(KRR_PATH)

    models = {
        "DNN": dnn_model,
        "SVR": svr_model,
        "KRR": krr_model
    }

    # 3. Compute Zero-Shot Predictions & Metrics
    print("Evaluating models on unseen chemistries...")
    results = []
    
    for name, model in models.items():
        # Predict Na-set
        if name == "DNN":
            pred_na = model.predict(X_na, verbose=0).flatten()
            pred_k = model.predict(X_k, verbose=0).flatten()
        else:
            pred_na = model.predict(X_na)
            pred_k = model.predict(X_k)
            
        na_metrics = compute_metrics(y_na, pred_na)
        k_metrics = compute_metrics(y_k, pred_k)
        
        # Store Na results
        results.append({
            "Test_Set": "Sodium (Na)",
            "Model": name,
            "Samples": len(y_na),
            "MAE_V": na_metrics["MAE_V"],
            "RMSE_V": na_metrics["RMSE_V"],
            "R2_Score": na_metrics["R2_Score"]
        })
        
        # Store K results
        results.append({
            "Test_Set": "Potassium (K)",
            "Model": name,
            "Samples": len(y_k),
            "MAE_V": k_metrics["MAE_V"],
            "RMSE_V": k_metrics["RMSE_V"],
            "R2_Score": k_metrics["R2_Score"]
        })

    # 4. Export Benchmark Table
    results_df = pd.DataFrame(results)
    # Sort for cleaner reading
    results_df = results_df.sort_values(by=["Test_Set", "MAE_V"])
    
    transfer_csv_path = os.path.join(RESULTS_DIR, "transfer_benchmark.csv")
    results_df.to_csv(transfer_csv_path, index=False)
    
    print(f"\nZero-Shot Transfer Benchmark Exported to: {transfer_csv_path}")
    print("\n" + results_df.to_string(index=False))
    print("=" * 70)

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
