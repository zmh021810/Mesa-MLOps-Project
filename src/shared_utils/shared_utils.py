import boto3
import pickle
import pandas as pd
import io
import os


class S3Handler:
    """专门负责 S3 数据交换"""

    def __init__(self, bucket_name):
        self.s3 = boto3.client("s3")
        self.bucket = bucket_name

    def load_csv(self, s3_key):
        print(f"📥 正在从 S3 下载数据: {s3_key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))

    def save_model(self, model, s3_key):
        print(f"📤 正在上传模型至: {s3_key}")
        pickle_byte_obj = pickle.dumps(model)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=pickle_byte_obj)

    def load_model(self, s3_key):
        print(f"🧠 正在加载模型: {s3_key}")
        obj = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
        return pickle.loads(obj["Body"].read())

    def save_json(self, data, s3_key):
        print(f"📝 正在保存结果至: {s3_key}")
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=pd.io.json.dumps(data))
