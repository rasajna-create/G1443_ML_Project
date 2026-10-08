import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

H_SET_FILE = os.path.join(DATA_DIR, "h_set.csv")

# Note: Adjusting extensions to match the files we generated in previous steps
DNN_PATH = os.path.join(MODELS_DIR, "dnn_model.h5")
SVR_PATH = os.path.join(MODELS_DIR, "svr_model.pkl")
KRR_PATH = os.path.join(MODELS_DIR, "krr_model.pkl")

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
    print("BATTERY VOLTAGE PREDICTION - PARITY PLOTS & BENCHMARKING")
    print("=" * 70)

    # 1. Load Data
    print(f"Loading Holdout test set from {H_SET_FILE}...")
    df_h = pd.read_csv(H_SET_FILE, low_memory=False)
    
    y_true = df_h[TARGET_COL].values
    available_metadata = [c for c in METADATA_COLS if c in df_h.columns]
    X = df_h.drop(columns=available_metadata + [TARGET_COL]).values

    # 2. Load Models
    print("Loading pre-trained models...")
    # Passing compile=False to avoid legacy keras metric deserialization bugs
    dnn_model = load_model(DNN_PATH, compile=False)
    svr_model = joblib.load(SVR_PATH)
    krr_model = joblib.load(KRR_PATH)

    # 3. Generate Predictions
    print("Generating predictions...")
    y_pred_dnn = dnn_model.predict(X, verbose=0).flatten()
    y_pred_svr = svr_model.predict(X)
    y_pred_krr = krr_model.predict(X)

    predictions = {
        "DNN": y_pred_dnn,
        "SVR": y_pred_svr,
        "KRR": y_pred_krr
    }

    # 4. Compute Metrics
    print("Computing evaluation metrics...")
    results = []
    for name, y_pred in predictions.items():
        mae = mean_absolute_error(y_true, y_pred)
        rmse = root_mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        results.append({
            "Model": name,
            "Holdout_MAE_V": round(mae, 4),
            "Holdout_RMSE_V": round(rmse, 4),
            "Holdout_R2": round(r2, 4)
        })

    # 5. Export Benchmark Table
    results_df = pd.DataFrame(results)
    benchmark_csv_path = os.path.join(RESULTS_DIR, "model_benchmark_comparison.csv")
    results_df.to_csv(benchmark_csv_path, index=False)
    print(f"\nConsolidated benchmark table exported to: {benchmark_csv_path}")
    print(results_df.to_string(index=False))

    # 6. Generate 3-Panel Parity Plot
    print("\nGenerating 3-panel parity plot...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    
    models_list = ["DNN", "SVR", "KRR"]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    
    # Calculate limits for the ideal line based on the data
    min_val = min(y_true.min(), min([p.min() for p in predictions.values()])) - 0.5
    max_val = max(y_true.max(), max([p.max() for p in predictions.values()])) + 0.5
    
    for i, model_name in enumerate(models_list):
        ax = axes[i]
        y_p = predictions[model_name]
        c = colors[i]
        
        # Scatter plot of actual vs predicted
        ax.scatter(y_true, y_p, alpha=0.5, color=c, edgecolor='k', s=25)
        
        # Ideal y=x line
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Ideal (y=x)")
        
        # +/- 0.5V error bands
        ax.plot([min_val, max_val], [min_val+0.5, max_val+0.5], 'gray', linestyle=':', lw=1.5, label="±0.5V Band")
        ax.plot([min_val, max_val], [min_val-0.5, max_val-0.5], 'gray', linestyle=':', lw=1.5)
        
        # Labels and formatting
        ax.set_title(f"{model_name} Parity Plot\nMAE = {results_df.loc[i, 'Holdout_MAE_V']} V", fontsize=14, fontweight='bold')
        ax.set_xlabel("DFT Calculated Voltage (V)", fontsize=12)
        if i == 0:
            ax.set_ylabel("ML Predicted Voltage (V)", fontsize=12)
        ax.set_xlim([min_val, max_val])
        ax.set_ylim([min_val, max_val])
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc="upper left")
        
    plt.tight_layout()
    plot_path = os.path.join(PLOTS_DIR, "parity_plot_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Publication-quality parity plot saved to: {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    # Silence TF warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
