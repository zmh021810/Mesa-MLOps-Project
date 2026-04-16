import argparse
import os
import pandas as pd
import joblib
from src.custom_package.processor import DataProcessor
from sklearn.ensemble import RandomForestClassifier


def main():
    # --- 1. Path Standardization ---
    # These are fixed paths defined by the SageMaker Training Toolkit
    input_dir = "/opt/ml/input/data/train"
    model_dir = "/opt/ml/model"

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # --- 2. Dynamic Data Discovery ---
    # Instead of hardcoding a filename, we scan the input directory.
    # This is critical for multi-tenancy where clients might upload 'data.csv' or 'client_v1.csv'
    files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]
    if not files:
        raise ValueError(
            f"No CSV files found in {input_dir}. Ensure your S3 'train/' prefix is not empty."
        )

    # Pick the first CSV found in the channel
    data_path = os.path.join(input_dir, files[0])
    print(f"📂 Loading training data from: {data_path}")

    # --- 3. Processing & Feature Engineering ---
    # Reading the dataset
    df = pd.read_csv(data_path)

    # Using your custom logic from the 'src' package
    processor = DataProcessor()
    df = processor.clean_data(df)
    X, y = processor.split_features_target(df)

    # --- 4. Optimized Model Training ---
    print("🧠 Training RandomForest model...")
    # n_jobs=-1 allows the model to use all available CPU cores on the m5.large instance
    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    model.fit(X, y)

    # --- 5. Model Export ---
    # The artifact must be saved directly to /opt/ml/model/ to be captured by SageMaker
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Training successful. Model artifact saved: {model_path}")


if __name__ == "__main__":
    main()
