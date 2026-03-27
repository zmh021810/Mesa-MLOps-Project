import argparse
import os
import pandas as pd
import joblib
from src.custom_package.processor import DataProcessor
from sklearn.ensemble import RandomForestClassifier

def main():
    parser = argparse.ArgumentParser()
    # SageMaker 会自动传入这些环境变量，或者通过 ContainerArguments 传入
    parser.add_argument('--bucket', type=str) 
    parser.add_argument('--data_key', type=str)
    args, _ = parser.parse_known_args()

    # 🌟 1. 定位数据：SageMaker 默认会将 S3 数据下载到这个本地目录
    # 如果你在 run.py 里的 ChannelName 叫 'train'，路径就是下面这个
    input_dir = "/opt/ml/input/data/train"
    
    # 获取该目录下第一个 csv 文件（兼容不同文件名的好办法）
    files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not files:
        raise ValueError(f"No CSV files found in {input_dir}")
    
    data_path = os.path.join(input_dir, files[0])
    print(f"📂 正在读取本地数据: {data_path}")
    
    # 2. 加载并处理
    df = pd.read_csv(data_path)
    processor = DataProcessor() # 假设你的自定义包已经写好
    df = processor.clean_data(df)
    X, y = processor.split_features_target(df)
    
    # 3. 训练
    print("🧠 正在训练 RandomForest 模型...")
    model = RandomForestClassifier()
    model.fit(X, y)
    
    # 🌟 4. 关键保存：必须存到 /opt/ml/model/ 目录下
    model_dir = "/opt/ml/model"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)
    
    print(f"✅ 训练完成，模型已保存至: {model_path}")

if __name__ == "__main__":
    main()