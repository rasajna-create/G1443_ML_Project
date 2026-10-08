import os
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. SETUP & CONSTANTS
# -----------------------------------------------------------------------------
try:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    BASE_DIR = os.path.abspath(".")

DATA_DIR = os.path.join(BASE_DIR, "data")

# Check for features_pca.csv or ml_dataset.csv
if os.path.exists(os.path.join(DATA_DIR, "features_pca.csv")):
    INPUT_FILE = os.path.join(DATA_DIR, "features_pca.csv")
else:
    INPUT_FILE = os.path.join(DATA_DIR, "ml_dataset.csv")

# Output files
T_SET_FILE = os.path.join(DATA_DIR, "t_set.csv")    # 90% Primary Training
H_SET_FILE = os.path.join(DATA_DIR, "h_set.csv")    # 10% Primary Holdout
NA_SET_FILE = os.path.join(DATA_DIR, "na_set.csv")  # Sodium extended test
K_SET_FILE = os.path.join(DATA_DIR, "k_set.csv")    # Potassium extended test

PRIMARY_IONS = ['Li', 'Mg', 'Ca', 'Al', 'Zn', 'Y']

# -----------------------------------------------------------------------------
# 2. MAIN EXECUTION
# -----------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BATTERY VOLTAGE PREDICTION - STEP 8: TRAIN/TEST/HOLDOUT SPLIT")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure Step 7 (PCA) was run.")
        return

    print(f"Loading ML dataset from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    # 1. Filter out the Na and K sets
    print("\nSeparating Extended Ion sets (Na, K)...")
    df_na = df[df['Working Ion'].astype(str).str.strip() == 'Na'].copy()
    df_k = df[df['Working Ion'].astype(str).str.strip() == 'K'].copy()
    
    # 2. Isolate the Primary Ion set
    df_primary = df[df['Working Ion'].astype(str).str.strip().isin(PRIMARY_IONS)].copy()
    
    print(f"Total Na-set samples : {len(df_na)}")
    print(f"Total K-set samples  : {len(df_k)}")
    print(f"Total Primary samples: {len(df_primary)}")
    
    # 3. Perform the 90/10 split on the Primary set
    print("\nPerforming 90/10 random split on the Primary set (T-set and H-set)...")
    
    t_set, h_set = train_test_split(
        df_primary, 
        test_size=0.10, 
        random_state=42, 
        stratify=df_primary['Working Ion']
    )
    
    print(f"T-set (Training 90%) : {len(t_set)} samples")
    print(f"H-set (Holdout 10%)  : {len(h_set)} samples")
    
    # 4. Save datasets
    t_set.to_csv(T_SET_FILE, index=False)
    h_set.to_csv(H_SET_FILE, index=False)
    df_na.to_csv(NA_SET_FILE, index=False)
    df_k.to_csv(K_SET_FILE, index=False)
    
    print("\nSaved partitioned datasets:")
    print(f" -> {T_SET_FILE}")
    print(f" -> {H_SET_FILE}")
    print(f" -> {NA_SET_FILE}")
    print(f" -> {K_SET_FILE}")
    
    print("\nData splitting is complete and perfectly aligned with the paper's methodology.")
    print("=" * 70)

if __name__ == '__main__':
    main()
