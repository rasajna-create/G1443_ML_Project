import os
import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def main():
    print("Step 15: Explainability (SHAP & PCA Loadings)")
    svr = joblib.load(os.path.join(MODELS_DIR, "svr_model.joblib"))
    pca = joblib.load(os.path.join(MODELS_DIR, "pca.joblib"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    
    X_train_pca = pd.read_csv(os.path.join(DATA_DIR, "X_train_pca.csv")).values
    X_test_pca = pd.read_csv(os.path.join(DATA_DIR, "X_test_pca.csv")).values
    
    # SHAP on SVR using KernelExplainer (on a small background sample)
    background = shap.sample(X_train_pca, 50)
    explainer = shap.KernelExplainer(svr.predict, background)
    test_sample = X_test_pca[:100]
    shap_values = explainer.shap_values(test_sample)
    
    # Calculate absolute mean SHAP per PC
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0) # shape (80,)
    
    # Map back to original 237 features
    # SHAP_org = dot(mean_abs_shap, abs(pca.components_))
    # Note: feature names from scaler
    original_features = scaler.feature_names_in_
    mapped_importances = np.dot(mean_abs_shap, np.abs(pca.components_))
    
    imp_df = pd.DataFrame({
        "Feature": original_features,
        "Importance": mapped_importances
    }).sort_values(by="Importance", ascending=False).head(10)
    
    plt.figure(figsize=(10, 6))
    plt.barh(imp_df["Feature"][::-1], imp_df["Importance"][::-1], color='#3498db')
    plt.title("Top 10 Most Influential Original Features (via SHAP * PCA Loadings)")
    plt.xlabel("Derived SHAP Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shap_feature_importance.png"), dpi=300)
    plt.close()
    
    print("Explainability complete. Plot saved.")

if __name__ == '__main__':
    main()
