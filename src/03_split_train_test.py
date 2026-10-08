import os
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "dataset_features_Xy.csv")

def main():
    print("Step 6: Train / Test Split (Before Scaling & PCA)")
    df = pd.read_csv(INPUT_FILE, low_memory=False)
    
    ions = df['Working Ion'].astype(str).str.strip()
    df_na = df[ions == 'Na'].copy()
    df_k = df[ions == 'K'].copy()
    
    PRIMARY = ['Li', 'Mg', 'Ca', 'Al', 'Zn', 'Y']
    df_primary = df[ions.isin(PRIMARY)].copy()
    
    meta_target = ['Working Ion', 'Battery Formula', 'Formula (Charged State)', 'Formula (Discharged State)', 'Electrode_data_material_id', 'Average Voltage (V)']
    available_meta = [c for c in meta_target if c in df.columns]
    
    df_train, df_test = train_test_split(df_primary, test_size=0.10, random_state=42, stratify=df_primary['Working Ion'])
    
    # Save raw sets
    df_train.drop(columns=available_meta, errors='ignore').to_csv(os.path.join(DATA_DIR, "X_train_raw.csv"), index=False)
    df_train[['Average Voltage (V)']].to_csv(os.path.join(DATA_DIR, "y_train.csv"), index=False)
    
    df_test.drop(columns=available_meta, errors='ignore').to_csv(os.path.join(DATA_DIR, "X_test_raw.csv"), index=False)
    df_test[['Average Voltage (V)']].to_csv(os.path.join(DATA_DIR, "y_test.csv"), index=False)
    
    df_na.to_csv(os.path.join(DATA_DIR, "naset.csv"), index=False)
    df_k.to_csv(os.path.join(DATA_DIR, "kset.csv"), index=False)
    
    # Save a master metadata dataframe to maintain indices for later reference if needed
    df_train[available_meta].to_csv(os.path.join(DATA_DIR, "train_metadata.csv"), index=False)
    df_test[available_meta].to_csv(os.path.join(DATA_DIR, "test_metadata.csv"), index=False)

    print("Splits completed. Target leakage removed.")

if __name__ == '__main__':
    main()
