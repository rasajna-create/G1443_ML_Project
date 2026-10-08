import os
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf

class BatteryVoltagePredictor:
    def __init__(self, models_dir):
        self.scaler = joblib.load(os.path.join(models_dir, "scaler.joblib"))
        self.pca = joblib.load(os.path.join(models_dir, "pca.joblib"))
        self.svr = joblib.load(os.path.join(models_dir, "svr_model.joblib"))
        self.krr = joblib.load(os.path.join(models_dir, "krr_model.joblib"))
        self.dnn = tf.keras.models.load_model(os.path.join(models_dir, "dnn_model.keras"), compile=False)
        
    def _preprocess(self, X_raw_df):
        # Align columns properly (assuming X_raw_df exactly matches the scaler's training features)
        if hasattr(self.scaler, 'feature_names_in_'):
            for col in self.scaler.feature_names_in_:
                if col not in X_raw_df.columns:
                    X_raw_df[col] = 0.0
            X_raw_df = X_raw_df[self.scaler.feature_names_in_]
        
        X_scaled = self.scaler.transform(X_raw_df)
        return self.pca.transform(X_scaled)
        
    def predict_single(self, X_raw_df, model_name="SVR"):
        X_pca = self._preprocess(X_raw_df)
        if model_name == "SVR":
            return self.svr.predict(X_pca)[0]
        elif model_name == "KRR":
            return self.krr.predict(X_pca)[0]
        else:
            return self.dnn.predict(X_pca, verbose=0).flatten()[0]

    def dynamic_profile(self, base_X_raw_df, fractions, model_name="SVR"):
        # For simulating multi-step intercalation.
        # It takes a base row, iterates over working-ion fraction, returns a curve
        df_sim = pd.concat([base_X_raw_df]*len(fractions), ignore_index=True)
        df_sim['Fraction_Charged'] = 0.0
        df_sim['Fraction_Discharged'] = fractions
        df_sim['Fraction_Interval'] = df_sim['Fraction_Discharged'] - df_sim['Fraction_Charged']
        X_pca = self._preprocess(df_sim)
        
        if model_name == "SVR":
            v = self.svr.predict(X_pca)
        elif model_name == "KRR":
            v = self.krr.predict(X_pca)
        else:
            v = self.dnn.predict(X_pca, verbose=0).flatten()
        return v
