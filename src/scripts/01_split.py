import pandas as pd
from sklearn.model_selection import train_test_split
import os

from utils.utils import save_dataframe


def split_data(df):
    # Split data into training and testing sets
    target = df["silica_concentrate"]
    feats = df.drop(["silica_concentrate"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(
        feats, target, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Load the cleaned data
    df = pd.read_csv("data/raw_data/raw.csv")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(df)

    # Save the split data to CSV files
    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test],
    ):
        save_dataframe(data, f"{name}.csv", "data/processed_data")
