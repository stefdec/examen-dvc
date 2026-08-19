import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
import os

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.utils import save_dataframe


def process_date(df):
    # Convert the 'date' column to datetime format
    df["date"] = pd.to_datetime(df["date"])
    # Extract year, month, and day from the 'date' column
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    # Drop the original 'date' column
    df.drop("date", axis=1, inplace=True)
    return df


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
    df = process_date(df)
    X_train, X_test, y_train, y_test = split_data(df)

    # Save the split data to CSV files
    for name, data in zip(
        ["X_train", "X_test", "y_train", "y_test"],
        [X_train, X_test, y_train, y_test],
    ):
        save_dataframe(data, f"{name}.csv", "data/processed_data")
