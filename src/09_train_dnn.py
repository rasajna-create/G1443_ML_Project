import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

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
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

T_SET_FILE = os.path.join(DATA_DIR, "t_set.csv")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "dnn_model.h5")

# Metadata columns to drop from X
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
    print("BATTERY VOLTAGE PREDICTION - STEP 9: TRAIN DNN MODEL")
    print("=" * 70)

    if not os.path.exists(T_SET_FILE):
        print(f"Error: {T_SET_FILE} not found. Ensure Step 8 was run.")
        return

    print("Loading T-set (Training Data)...")
    df_train = pd.read_csv(T_SET_FILE)
    
    # Isolate X and y
    available_metadata = [c for c in METADATA_COLS if c in df_train.columns]
    y = df_train[TARGET_COL].values
    X = df_train.drop(columns=available_metadata + [TARGET_COL]).values
    
    input_dim = X.shape[1]
    print(f"Training features shape: {X.shape}")
    
    # 1. BUILD MODEL (Architecture from Joshi et al.)
    # "4 hidden layers consisting of 120, 80, 40 and 10 neurons, respectively. 
    # The nonlinear RELU function was applied on the output of all nodes..."
    print("\nBuilding DNN Architecture...")
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(120, activation='relu'),
        Dense(80, activation='relu'),
        Dense(40, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')  # "...final predicted voltage is a linear sum"
    ])
    
    # "We used the AMSGrad version of the Adam optimizer... learning rate starting at 0.005"
    # "...mean-absolute-error is used as the loss function"
    optimizer = Adam(learning_rate=0.005, amsgrad=True)
    model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])
    
    model.summary()
    
    # 2. CONFIGURE CALLBACKS
    # "Early-stopping is used to terminate training... to prevent overfitting"
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=50,          # Stop if no improvement after 50 epochs
        restore_best_weights=True,
        verbose=1
    )
    
    # 3. TRAIN MODEL
    print("\nStarting DNN Training...")
    # Using 10% of the T-set for validation during training to guide early stopping
    history = model.fit(
        X, y,
        validation_split=0.1,
        epochs=1000,          # High max limit, relying on early stopping
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # 4. SAVE MODEL
    model.save(MODEL_SAVE_PATH)
    print(f"\nSaved DNN model to {MODEL_SAVE_PATH}")
    
    # 5. PLOT TRAINING HISTORY
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train MAE', color='#2b5c8f')
    plt.plot(history.history['val_loss'], label='Validation MAE', color='#e67e22')
    plt.title('DNN Training History (AMSGrad + Early Stopping)')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error (V)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    plot_path = os.path.join(PLOTS_DIR, "dnn_training_history.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Saved training history plot to {plot_path}")
    print("=" * 70)

if __name__ == '__main__':
    # Silence TF info logs
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()
