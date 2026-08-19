import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import save_dataframe


def normalize_data(df):
    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    normalized_feats = scaler.fit_transform(df)
    normalized_df = pd.DataFrame(normalized_feats, columns=df.columns)
    return normalized_df


if __name__ == "__main__":
    # Load the split data
    X_train = pd.read_csv("data/processed_data/X_train.csv")
    X_test = pd.read_csv("data/processed_data/X_test.csv")

    # Normalize the features
    X_train_scaled = normalize_data(X_train)
    X_test_scaled = normalize_data(X_test)

    # Save the normalized data to CSV files
    save_dataframe(X_train_scaled, "X_train_scaled.csv", "data/processed_data")
    save_dataframe(X_test_scaled, "X_test_scaled.csv", "data/processed_data")
