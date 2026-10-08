import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def main():
    print("Step 7 & 8: Normalization & PCA (Leak-Free)")
    
    X_train_raw = pd.read_csv(os.path.join(DATA_DIR, "X_train_raw.csv"))
    X_test_raw = pd.read_csv(os.path.join(DATA_DIR, "X_test_raw.csv"))
    
    print(f"X_train_raw shape: {X_train_raw.shape}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)
    
    pca = PCA(n_components=80, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    pca_cols = [f"PC{i+1}" for i in range(80)]
    
    pd.DataFrame(X_train_pca, columns=pca_cols).to_csv(os.path.join(DATA_DIR, "X_train_pca.csv"), index=False)
    pd.DataFrame(X_test_pca, columns=pca_cols).to_csv(os.path.join(DATA_DIR, "X_test_pca.csv"), index=False)
    
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca.joblib"))
    
    print("Preprocessing completed perfectly without data leakage.")

if __name__ == '__main__':
    main()
