import joblib
import tarfile
import os
import boto3
import pandas as pd
import io
from datetime import datetime

s3 = boto3.client('s3')

# 全局缓存
cached_model = None
cached_model_path = None

def handler(event, context):
    global cached_model, cached_model_path
    
    try:
        bucket = event.get('bucket')
        model_key = event.get('model_key')    
        data_key = event.get('data_key')      
        
        # --- 2. 环境准备 ---
        local_tar = '/tmp/model.tar.gz'
        extract_path = '/tmp/model/'
        # 🌟 指向解压后的 output 子目录
        model_file = os.path.join(extract_path, "output", "model.pkl")

        # --- 第一部分：智能模型加载 ---
        if cached_model and cached_model_path == model_key:
            print("🚀 命中内存缓存")
            model = cached_model
        else:
            # 如果磁盘也没解压过，或者 model_key 变了，重新下载解压
            if not os.path.exists(model_file):
                print(f"📥 开始下载并解压: {model_key}")
                if not os.path.exists(extract_path):
                    os.makedirs(extract_path)
                
                s3.download_file(bucket, model_key, local_tar)
                
                with tarfile.open(local_tar, "r:gz") as tar:
                    tar.extractall(path=extract_path)
                
                # 兼容性检查：如果 output/model.pkl 不存在，尝试找根目录的 model.pkl
                if not os.path.exists(model_file):
                    model_file = os.path.join(extract_path, "model.pkl")
            
            model = joblib.load(model_file)
            cached_model = model
            cached_model_path = model_key

        # --- 第二部分：读取预测数据 ---
        response = s3.get_object(Bucket=bucket, Key=data_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))

        # --- 第三部分：执行预测 ---
        predictions = model.predict(df)

        # --- 第四部分：保存结果到 s3/output ---
        df['prediction'] = predictions
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base_name = os.path.basename(data_key).split('.')[0]
        output_key = f"output/{file_base_name}_results_{timestamp}.csv"
        
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=csv_buffer.getvalue()
        )

        return {
            "statusCode": 200,
            "body": {
                "message": "Success",
                "output_s3": f"s3://{bucket}/{output_key}",
                "rows": len(df)
            }
        }

    except Exception as e:
        print(f"❌ 异常: {str(e)}")
        return {"statusCode": 500, "body": str(e)}