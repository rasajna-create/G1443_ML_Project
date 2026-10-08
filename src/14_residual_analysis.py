import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

H_SET_FILE = os.path.join(DATA_DIR, "h_set.csv")
SVR_PATH = os.path.join(MODELS_DIR, "svr_model.pkl")

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
    print("BATTERY VOLTAGE PREDICTION - SVR RESIDUAL ANALYSIS")
    print("=" * 70)

    # 1. Load Data & Model
    print("Loading Holdout test set and SVR model...")
    df = pd.read_csv(H_SET_FILE, low_memory=False)
    svr_model = joblib.load(SVR_PATH)

    y_true = df[TARGET_COL].values
    available_metadata = [c for c in METADATA_COLS if c in df.columns]
    X = df.drop(columns=available_metadata + [TARGET_COL]).values

    # 2. Compute Predictions & Residuals
    y_pred = svr_model.predict(X)
    df['Predicted_Voltage_V'] = y_pred
    df['Residual_V'] = y_true - y_pred
    df['Absolute_Error_V'] = np.abs(df['Residual_V'])

    # 3. MAE Segmented by Working Ion
    print("\n--- MAE Segmented by Working Ion ---")
    ion_stats = df.groupby('Working Ion')['Absolute_Error_V'].agg(['mean', 'count']).sort_values('mean')
    ion_stats.columns = ['MAE (V)', 'Count']
    print(ion_stats)

    # 4. Top 5 Largest Outliers
    print("\n--- Top 5 Largest Prediction Outliers ---")
    outliers = df.sort_values(by='Absolute_Error_V', ascending=False).head(5)
    cols_to_print = ['Working Ion', 'Battery Formula', TARGET_COL, 'Predicted_Voltage_V', 'Absolute_Error_V']
    cols_to_print = [c for c in cols_to_print if c in df.columns]
    print(outliers[cols_to_print].to_string(index=False))

    # 5. Generate Dual-Panel Plot
    print("\nGenerating residual plots...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel 1: Histogram of Residuals
    axes[0].hist(df['Residual_V'], bins=40, color='#3498db', edgecolor='black', alpha=0.7)
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error')
    axes[0].set_title('Residual Error Distribution\n($V_{DFT} - V_{ML}$)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Residual Error (V)', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].grid(axis='y', linestyle='--', alpha=0.6)
    axes[0].legend()

    # Panel 2: Per-Ion Boxplot
    ions = list(ion_stats.index)
    plot_data = [df[df['Working Ion'] == ion]['Residual_V'].values for ion in ions]
    
    box = axes[1].boxplot(plot_data, tick_labels=ions, patch_artist=True, notch=True)
    
    # Color the boxes
    colors = ['#1abc9c', '#2ecc71', '#3498db', '#9b59b6', '#34495e', '#f1c40f']
    for patch, color in zip(box['boxes'], colors[:len(ions)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        
    axes[1].axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axes[1].set_title('Residual Distribution by Working Ion', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Working Ion', fontsize=12)
    axes[1].set_ylabel('Residual Error (V)', fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "residual_analysis.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Dual-panel residual plot saved to: {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    main()
