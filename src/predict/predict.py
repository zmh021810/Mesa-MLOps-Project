import joblib
import tarfile
import os
import boto3
import pandas as pd
import io
from datetime import datetime
import glob


def run_inference():
    # --- 1. SageMaker Standard Paths ---
    # SageMaker automatically downloads and extracts model.tar.gz into this folder
    model_dir = "/opt/ml/model"
    # SageMaker mounts input data from S3 into this folder
    input_dir = "/opt/ml/input/data/batch"
    # Anything saved here is automatically uploaded to S3 after the job finishes
    output_dir = "/opt/ml/output"

    # --- 2. Load Model ---
    # Supporting both your previous 'output/model.pkl' and root 'model.pkl' structures
    model_path = os.path.join(model_dir, "output", "model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(model_dir, "model.pkl")

    print(f"📦 Loading model from: {model_path}")
    model = joblib.load(model_path)

    # --- 3. Process Files with Chunking ---
    # Batch Transform might send multiple files or one large split file
    input_files = glob.glob(os.path.join(input_dir, "*"))

    if not input_files:
        print("⚠️ No input files found in /opt/ml/input/data/batch")
        return

    for file_path in input_files:
        file_name = os.path.basename(file_path)
        output_file_path = os.path.join(output_dir, f"predictions_{file_name}")

        print(f"📖 Processing file: {file_name}")

        # 🌟 Use chunking to handle massive datasets (e.g., 5M+ rows) without OOM
        # 100,000 rows per chunk is a safe default for m5.large memory
        chunk_size = 100000
        first_chunk = True

        # Read CSV in chunks
        for chunk in pd.read_csv(file_path, chunksize=chunk_size):
            # Perform prediction
            # Ensure your model's .predict() expects the same feature set as your CSV
            predictions = model.predict(chunk)

            # Append predictions to the current chunk
            chunk["prediction"] = predictions

            # Write results back to CSV
            # 'w' (write) for the first chunk, 'a' (append) for subsequent ones
            write_mode = "w" if first_chunk else "a"
            write_header = True if first_chunk else False

            chunk.to_csv(
                output_file_path,
                mode=write_mode,
                index=False,
                header=write_header,
                lineterminator="\n",
            )
            first_chunk = False

        print(f"✅ Successfully processed and saved: {output_file_path}")


if __name__ == "__main__":
    run_inference()
