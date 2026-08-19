import pandas as pd
import joblib

from lightgbm import LGBMRegressor

PARAMS_PATH = "models/grid_search_lgbm_best_params.pkl"


def get_model_params():
    # Load the best hyperparameters from the pickle file
    best_params = joblib.load(PARAMS_PATH)
    return best_params


def train_full_ds(X_train, y_train):
    # Train the model on the full dataset
    model = LGBMRegressor(**get_model_params())
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":
    # Load the training data
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    # Train the model on the full dataset
    trained_model = train_full_ds(X_train, y_train)

    # save the model with pkl extension
    joblib.dump(trained_model, "models/lgbm_model.pkl")
