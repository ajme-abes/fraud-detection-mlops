import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering
    """
    df = df.copy()

    # Convert time to hours (simple feature)
    df["hour"] = (df["Time"] // 3600) % 24

    return df