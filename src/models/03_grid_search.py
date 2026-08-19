import pandas as pd
import joblib

from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMRegressor


def grid_search(X_train, y_train):
    # Define the model
    model = LGBMRegressor()

    # Define the hyperparameter grid
    param_grid = {
        "num_leaves": [15, 31],
        "learning_rate": [0.05, 0.1],
        "n_estimators": [100, 200],
    }

    # Set up the grid search
    grid_search = GridSearchCV(
        estimator=model,
        scoring="neg_mean_squared_error",
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        verbose=2,
    )

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


if __name__ == "__main__":
    # Load the training data
    X_train = pd.read_csv("data/processed_data/X_train_scaled.csv")
    y_train = pd.read_csv("data/processed_data/y_train.csv")

    # Perform grid search to find the best model and hyperparameters
    best_model, best_params, best_score = grid_search(X_train, y_train)

    # save the best params in pkl format
    joblib.dump(best_params, "models/grid_search_lgbm_best_params.pkl")
