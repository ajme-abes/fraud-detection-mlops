import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.config.config_loader import get_config


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale columns defined in config (Amount, Time by default).
    """
    cfg = get_config()
    scale_cols = cfg["preprocessing"]["scale_columns"]

    df = df.copy()
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    return df


def split_data(df: pd.DataFrame):
    """
    Split dataset with stratification using config parameters.
    """
    cfg = get_config()
    prep_cfg = cfg["preprocessing"]

    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=prep_cfg["test_size"],
        stratify=y,
        random_state=prep_cfg["random_state"],
    )

    return X_train, X_test, y_train, y_test


def save_data(X_train, X_test, y_train, y_test):
    """
    Save processed splits to the directory defined in config.
    """
    cfg = get_config()
    processed_dir = cfg["paths"]["processed_dir"]

    os.makedirs(processed_dir, exist_ok=True)

    X_train.to_csv(os.path.join(processed_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(processed_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(processed_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(processed_dir, "y_test.csv"), index=False)
