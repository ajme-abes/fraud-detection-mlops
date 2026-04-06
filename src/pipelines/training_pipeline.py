from src.data.load_data import load_data, validate_data
from src.data.preprocess import clean_data, split_data, save_data
from src.features.build_features import build_features
from src.models.train import train_model, evaluate_model


def run_pipeline():
    print("🚀 Starting Training Pipeline...\n")

    my_path = r'data/raw/creditcard.csv'


    # 1. Load
    df = load_data(my_path)

    # 2. Validate
    validate_data(df)

    # 3. Clean
    df = clean_data(df)

    # 4. Feature Engineering
    df = build_features(df)

    # 5. Split
    X_train, X_test, y_train, y_test = split_data(df)

    # 6. Save processed data
    save_data(X_train, X_test, y_train, y_test)

    # 7. Train
    model = train_model(X_train, y_train, X_test, y_test)
    # 8. Evaluate
    evaluate_model(model, X_test, y_test)

    print("\n✅ Pipeline completed successfully!")


if __name__ == "__main__":
    run_pipeline()