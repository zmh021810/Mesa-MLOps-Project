import pandas as pd

class DataProcessor:
    """负责将原始 DataFrame 转换为模型可用的 X 和 y"""
    def __init__(self, target_column=None):
        self.target_column = target_column

    def split_features_target(self, df):
        """
        根据目标列名切分 X 和 y。
        如果没指定 target_column，默认最后一列为 y。
        """
        print(f"📊 正在处理数据，目标列: {self.target_column if self.target_column else '最后一列'}")
        
        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1]
            
        return X, y

    def clean_data(self, df):
        """
        以后你可以在这里添加复杂的清洗逻辑（如填充缺失值、归一化等）
        """
        return df.dropna()