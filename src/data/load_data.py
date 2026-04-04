import pandas as pd


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV
    """
    df = pd.read_csv(path)
    return df


def validate_data(df: pd.DataFrame) -> None:
    """
    Basic data validation checks
    """
    print("\n🔍 Data Info:")
    print(df.info())

    print("\n🧹 Missing Values:")
    print(df.isnull().sum())

    print("\n⚖️ Class Distribution:")
    print(df["Class"].value_counts(normalize=True))