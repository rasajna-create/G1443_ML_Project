import os
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# PATH CONFIGURATION
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_FILE = os.path.join(DATA_DIR, "mp_full_dataset.csv")
if not os.path.exists(INPUT_FILE):
    INPUT_FILE = os.path.join(DATA_DIR, "mpfulldataset.csv")

OUTPUT_FILE = os.path.join(DATA_DIR, "cleaned_dataset.csv")

def main():
    print("=" * 70)
    print("STEP 3 & 4: ROBUST DATA INSPECTION, SANITIZATION & CLEANING")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found at: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, low_memory=False)
    initial_count = len(df)
    print(f"Loaded raw records: {initial_count}")

    # 1. Numeric coercion & physical voltage bounds (0 < V <= 10)
    voltage_col = 'Average Voltage (V)'
    df[voltage_col] = pd.to_numeric(df[voltage_col], errors='coerce')
    df = df[(df[voltage_col] > 0.0) & (df[voltage_col] <= 10.0)]
    print(f"After physical voltage filter (0 < V <= 10 V): {len(df)} rows")

    # 2. String sanitization: strip whitespace and kill string artifacts
    str_cols = ['Working Ion', 'Formula (Charged State)', 'Formula (Discharged State)']
    for col in str_cols:
        if col in df.columns:
            # Strip leading/trailing whitespaces
            df[col] = df[col].astype(str).str.strip()
            # Replace string literal artifacts with real NaN
            df[col] = df[col].replace(["nan", "None", "null", "undefined", "", "NaN"], np.nan)

    # 3. Drop genuine NaNs across critical chemical keys
    df = df.dropna(subset=str_cols + [voltage_col])
    print(f"After string sanitization and null removal: {len(df)} rows")

    # 4. Multi-Step & Polymorph Aware Deduplication
    # Group by the calculation/battery system AND the exact state transition formulas
    id_candidates = ['Electrode_data_material_id', 'battery_id', 'id', 'Materials ID']
    matched_id = [c for c in id_candidates if c in df.columns]

    if matched_id:
        # Preserves distinct intermediate steps (Chg -> Dischg) within the same battery system
        dedup_keys = matched_id + ['Formula (Charged State)', 'Formula (Discharged State)', 'Working Ion']
    else:
        # Fallback if raw calculation IDs are completely missing
        dedup_keys = ['Formula (Charged State)', 'Formula (Discharged State)', 'Working Ion']
        if 'Spacegroup' in df.columns:
            dedup_keys.append('Spacegroup')

    df = df.drop_duplicates(subset=dedup_keys, keep='first')
    print(f"After multi-step and polymorph-aware deduplication: {len(df)} rows")

    # 5. Export clean dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nFinal cleaned dataset saved to: {OUTPUT_FILE}")
    print(f"Total dropped unphysical/duplicate records: {initial_count - len(df)}")
    print("=" * 70)

if __name__ == '__main__':
    main()
