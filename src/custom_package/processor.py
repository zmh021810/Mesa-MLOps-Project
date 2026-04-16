import pandas as pd


class DataProcessor:
    """Responsible for transforming raw DataFrames into model-ready X and y."""

    def __init__(self, target_column=None):
        self.target_column = target_column

    def split_features_target(self, df):
        """
        Splits the DataFrame into features (X) and target (y).
        Explicitly drops the first column (assumed to be ID) to avoid noise.
        """
        # --- 1. Drop the first column (ID) ---
        # Since the first column is an ID, it provides no predictive value.
        # We slice from index 1 to the end.
        id_col_name = df.columns[0]
        print(f"🗑️ Dropping the first column (ID): {id_col_name}")
        df_processed = df.iloc[:, 1:]

        # --- 2. Determine Target Column ---
        target_name = self.target_column if self.target_column else "Last Column"
        print(f"📊 Processing features. Target selected: {target_name}")

        if self.target_column and self.target_column in df_processed.columns:
            X = df_processed.drop(columns=[self.target_column])
            y = df_processed[self.target_column]
        else:
            # Features are everything except the last column
            X = df_processed.iloc[:, :-1]
            # Target is the final column
            y = df_processed.iloc[:, -1]

        return X, y

    def clean_data(self, df):
        """
        Basic cleaning logic. Currently drops any rows containing null values.
        Can be extended for normalization or missing value imputation.
        """
        return df.dropna()
