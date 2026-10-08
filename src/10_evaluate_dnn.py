import os
import pandas as pd
import numpy as np
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

H_SET_FILE = os.path.join(DATA_DIR, "h_set.csv")
NA_SET_FILE = os.path.join(DATA_DIR, "na_set.csv")
K_SET_FILE = os.path.join(DATA_DIR, "k_set.csv")
DNN_MODEL_PATH = os.path.join(MODELS_DIR, "dnn_model.h5")

METADATA_COLS = [
    'Working Ion', 'Battery Formula', 'Formula (Charged State)',
    'Formula (Discharged State)', 'Electrode_data_material_id'
]
TARGET_COL = 'Average Voltage (V)'

# -----------------------------------------------------------------------------
# 2. EVALUATION FUNCTION
# -----------------------------------------------------------------------------
def evaluate_dataset(model, filepath, dataset_name):
    if not os.path.exists(filepath):
        print(f"File {filepath} not found. Skipping {dataset_name} evaluation.")
        return
        
    df = pd.read_csv(filepath)
    if len(df) == 0:
        print(f"{dataset_name} is empty. Skipping.")
        return
        
    y_true = df[TARGET_COL].values
    available_metadata = [c for c in METADATA_COLS if c in df.columns]
    X = df.drop(columns=available_metadata + [TARGET_COL]).values
    
    y_pred = model.predict(X, verbose=0).flatten()
    
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n--- {dataset_name} Performance ---")
    print(f"Samples Evaluated: {len(df)}")
    print(f"Mean Absolute Error (MAE) : {mae:.4f} V")
    print(f"Root Mean Squared Error   : {rmse:.4f} V")
    print(f"R-squared (R2)            : {r2:.4f}")
    
    return y_true, y_pred

# -----------------------------------------------------------------------------
# 3. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 10: MODEL EVALUATION")
    print("=" * 70)
    
    if not os.path.exists(DNN_MODEL_PATH):
        print(f"Error: {DNN_MODEL_PATH} not found. Ensure Step 9 was run.")
        return
        
    print(f"Loading trained DNN model from {DNN_MODEL_PATH}...")
    model = load_model(DNN_MODEL_PATH, compile=False)
    
    # Evaluate on the Primary Holdout Set (H-set)
    evaluate_dataset(model, H_SET_FILE, "Primary Holdout Set (H-set)")
    
    # Evaluate on the Extended Ion Sets to test extreme generalization
    evaluate_dataset(model, NA_SET_FILE, "Sodium Extended Set (Na-set)")
    evaluate_dataset(model, K_SET_FILE, "Potassium Extended Set (K-set)")
    
    print("\n" + "=" * 70)
    print("Evaluation Complete. Notice how MAE changes on unseen Working Ions!")
    print("=" * 70)

if __name__ == '__main__':
    # Silence TF info logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
