import boto3
import joblib
import pandas as pd
import io
import json
import os


class S3Handler:
    """Handles S3 data exchange with memory efficiency and modern serialization."""

    def __init__(self, bucket_name):
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name

    def load_csv(self, s3_key, chunksize=None):
        """
        Downloads data from S3. Supports chunking for large datasets (5M+ rows).
        """
        print(f"📥 Downloading data from S3: {s3_key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)

        # If chunksize is provided, it returns an iterator to prevent memory overflow
        return pd.read_csv(io.BytesIO(obj["Body"].read()), chunksize=chunksize)

    def save_model(self, model, s3_key):
        """
        Uploads model to S3 using Joblib for better efficiency with Scikit-learn.
        """
        print(f"📤 Uploading model to: {s3_key}")
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=buffer.getvalue())

    def load_model(self, s3_key):
        """
        Loads a Joblib-serialized model from S3.
        """
        print(f"🧠 Loading model from S3: {s3_key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return joblib.load(io.BytesIO(obj["Body"].read()))

    def save_json(self, data, s3_key):
        """
        Saves structured results to S3 as a JSON file.
        """
        print(f"📝 Saving results to: {s3_key}")
        json_data = json.dumps(data)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=json_data)

    @staticmethod
    def ensure_local_dir(path):
        """Ensures a local directory exists before saving files."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
