import os
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -----------------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -----------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(".")

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

T_SET_FILE = os.path.join(DATA_DIR, "t_set.csv")
H_SET_FILE = os.path.join(DATA_DIR, "h_set.csv")
NA_SET_FILE = os.path.join(DATA_DIR, "na_set.csv")
K_SET_FILE = os.path.join(DATA_DIR, "k_set.csv")

SVR_MODEL_PATH = os.path.join(MODELS_DIR, "svr_model.pkl")

METADATA_COLS = [
    'Working Ion', 'Battery Formula', 'Formula (Charged State)',
    'Formula (Discharged State)', 'Electrode_data_material_id'
]
TARGET_COL = 'Average Voltage (V)'

# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def load_and_split(filepath):
    df = pd.read_csv(filepath)
    if len(df) == 0: return None, None
    y = df[TARGET_COL].values
    available_metadata = [c for c in METADATA_COLS if c in df.columns]
    X = df.drop(columns=available_metadata + [TARGET_COL]).values
    return X, y

def evaluate_set(model, X, y_true, name):
    if X is None: return
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name:.<30} MAE = {mae:.4f} V  |  R2 = {r2:7.4f}")

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 11: SUPPORT VECTOR REGRESSION (SVR)")
    print("=" * 70)

    # 1. Load data
    print("Loading datasets...")
    X_train, y_train = load_and_split(T_SET_FILE)
    X_h, y_h = load_and_split(H_SET_FILE)
    X_na, y_na = load_and_split(NA_SET_FILE)
    X_k, y_k = load_and_split(K_SET_FILE)
    
    # 2. Train SVR
    print("\nTraining SVR with RBF Kernel...")
    print("(Using default hyperparameters; Joshi et al. typically optimized C and gamma via GridSearch)")
    
    # We use a robust default that performs well for normalized/PCA data
    svr_model = SVR(kernel='rbf', C=10.0, gamma='scale')
    svr_model.fit(X_train, y_train)
    
    # 3. Save Model
    joblib.dump(svr_model, SVR_MODEL_PATH)
    print(f"Model saved to {SVR_MODEL_PATH}")
    
    # 4. Evaluate
    print("\n--- SVR EVALUATION RESULTS ---")
    evaluate_set(svr_model, X_train, y_train, "Training Set (T-set)")
    evaluate_set(svr_model, X_h, y_h, "Primary Holdout (H-set)")
    evaluate_set(svr_model, X_na, y_na, "Sodium Set (Na-set)")
    evaluate_set(svr_model, X_k, y_k, "Potassium Set (K-set)")
    
    print("=" * 70)

if __name__ == '__main__':
    main()
