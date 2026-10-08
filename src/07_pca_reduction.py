import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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
PLOTS_DIR = os.path.join(BASE_DIR, "results", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

INPUT_FILE = os.path.join(DATA_DIR, "features_normalized.csv")
OUTPUT_FILE = os.path.join(DATA_DIR, "ml_dataset.csv")
PCA_MODEL_PATH = os.path.join(MODELS_DIR, "pca_model.pkl")

# Metadata and Target columns to exclude from PCA
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
    print("BATTERY VOLTAGE PREDICTION - STEP 7: PCA DIMENSIONALITY REDUCTION")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Ensure Step 6 was run.")
        return

    print(f"Loading normalized features from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE, low_memory=False)

    # Separate Metadata, Target, and Feature Matrix (X)
    available_metadata = [c for c in METADATA_COLS if c in df.columns]
    
    df_meta = df[available_metadata].copy()
    y = df[TARGET_COL].copy()
    
    # Everything else is our normalized feature matrix X
    X_scaled = df.drop(columns=available_metadata + [TARGET_COL])
    
    n_components = 80
    print(f"Found {X_scaled.shape[1]} features. Applying PCA to reduce down to {n_components} components (per Joshi et al.)...")
    
    # Initialize and fit PCA
    pca = PCA(n_components=n_components, random_state=42)
    X_pca_values = pca.fit_transform(X_scaled)
    
    # Calculate explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    total_variance = cumulative_variance[-1] * 100
    
    print(f"Total Explained Variance by {n_components} components: {total_variance:.2f}%")
    
    # Convert back to dataframe with PC1, PC2, etc.
    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    df_pca = pd.DataFrame(X_pca_values, columns=pca_cols, index=X_scaled.index)
    
    # Reassemble final ML dataset
    print("Reassembling final ML dataset with Principal Components...")
    df_final = pd.concat([df_meta, df_pca, y], axis=1)
    
    # Save datasets and models
    df_final.to_csv(OUTPUT_FILE, index=False)
    joblib.dump(pca, PCA_MODEL_PATH)
    
    print(f"\nSaved PCA-reduced dataset to: {OUTPUT_FILE}")
    print(f"Saved PCA model to: {PCA_MODEL_PATH}")
    
    # -------------------------------------------------------------------------
    # 3. GENERATE PCA VARIANCE PLOT
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    plt.plot(range(1, n_components + 1), cumulative_variance * 100, marker='o', linestyle='-', markersize=4, color="#2b5c8f")
    plt.axhline(y=total_variance, color='r', linestyle='--', alpha=0.5, label=f"Total: {total_variance:.1f}%")
    plt.xlabel('Number of Principal Components', fontsize=12)
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=12)
    plt.title(f'PCA Explained Variance ({n_components} Components)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, "pca_variance_plot.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Generated PCA variance plot: {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
