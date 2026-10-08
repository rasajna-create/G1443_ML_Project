import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

INPUT_FILE = os.path.join(DATA_DIR, "features_raw.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "features_normalized.csv")
SCALER_PATH = os.path.join(MODELS_DIR, "standard_scaler.pkl")

# Metadata and Target columns to exclude from scaling
METADATA_COLS = [
    'Working Ion', 'Battery Formula', 'Formula (Charged State)',
    'Formula (Discharged State)', 'Electrode_data_material_id'
]
TARGET_COL = 'Average Voltage (V)'

# -----------------------------------------------------------------------------
# 2. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 6: FEATURE NORMALIZATION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure Step 5 was run.")
        return

    print(f"Loading raw features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Separate Metadata, Target, and Feature Matrix (X)
    available_metadata = [c for c in METADATA_COLS if c in df.columns]
    
    df_meta = df[available_metadata].copy()
    y = df[TARGET_COL].copy()
    
    # Everything else is our feature matrix X
    X = df.drop(columns=available_metadata + [TARGET_COL])
    
    print(f"Found {X.shape[1]} raw numerical features across {X.shape[0]} samples.")
    
    print("Applying StandardScaler (Mean=0, Variance=1)...")
    scaler = StandardScaler()
    
    # Fit and transform
    X_scaled_values = scaler.fit_transform(X)
    
    # Convert back to dataframe with original column names
    X_scaled = pd.DataFrame(X_scaled_values, columns=X.columns, index=X.index)
    
    # Reassemble dataset
    print("Reassembling dataset with normalized features...")
    df_normalized = pd.concat([df_meta, X_scaled, y], axis=1)
    
    # Save datasets and models
    df_normalized.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(scaler, SCALER_PATH)
    
    # Validation checks
    mean_vals = X_scaled.mean()
    std_vals = X_scaled.std()
    
    print("\n--- VALIDATION ---")
    print(f"Max absolute mean of scaled features: {mean_vals.abs().max():.6f} (Should be ~0)")
    print(f"Max standard deviation of scaled features: {std_vals.max():.6f} (Should be ~1)")
    print(f"Total NaN values after scaling: {X_scaled.isna().sum().sum()}")
    
    print(f"\nSaved normalized features to: {OUTPUT_FILE}")
    print(f"Saved scaler model to: {SCALER_PATH}")
    print("=" * 70)

if __name__ == '__main__':
    main()
