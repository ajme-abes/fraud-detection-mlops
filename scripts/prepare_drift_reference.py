import pandas as pd
import os

# --- CONFIG ---
DATA_PROCESSED_DIR = "data/processed"   # Where your x_train.csv lives
REFERENCE_DIR = "data"                  # Where to save training_features.csv
REFERENCE_FILE = os.path.join(REFERENCE_DIR, "training_features.csv")
X_TRAIN_FILE = os.path.join(DATA_PROCESSED_DIR, "x_train.csv")

# --- STEP 1: Load x_train ---
if not os.path.exists(X_TRAIN_FILE):
    raise FileNotFoundError(f"{X_TRAIN_FILE} not found! Make sure x_train.csv exists.")

x_train = pd.read_csv(X_TRAIN_FILE)

# --- STEP 2: Verify columns match model input ---
EXPECTED_FEATURES = 31  # Your model expects 31 features
if x_train.shape[1] != EXPECTED_FEATURES:
    raise ValueError(f"x_train has {x_train.shape[1]} columns, but model expects {EXPECTED_FEATURES} features.")

# --- STEP 3: Optionally sample to reduce size ---
# For drift, you don’t need all rows. Let's take first 1000 rows (or all if smaller)
sample_size = min(1000, len(x_train))
reference_df = x_train.iloc[:sample_size].copy()

# --- STEP 4: Save to REFERENCE_DIR ---
os.makedirs(REFERENCE_DIR, exist_ok=True)
reference_df.to_csv(REFERENCE_FILE, index=False)

print(f"Reference dataset saved at {REFERENCE_FILE} ({len(reference_df)} rows, {len(reference_df.columns)} columns)")