import random
import pandas as pd
import sklearn
import sklearn.tree
import sklearn.impute
import sklearn.ensemble
import sklearn.pipeline
from sklearn.metrics import (
    explained_variance_score,
    mean_squared_error,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV

# Fix the random seed to have repeatable experiments
random.seed(1337)

# THOSE LOADERS NEED TO BE IMPLEMENTED FOR EACH DATASET!!!
from preprocessing import train_loader, val_loader, test_loader

import numpy as np

train_data = train_loader()
x_train = np.vstack([x[-1, :] for x in train_data["x"]])
y_train = np.vstack(train_data["y"])

test_data = test_loader()
x_test = np.vstack([x[-1, :] for x in test_data["x"]])
y_test = np.vstack(test_data["y"])

metrics = {
    "explained_variance": explained_variance_score,
    "mse": mean_squared_error,
    "mae": mean_absolute_error,
}

models = [
    ("Linear", sklearn.linear_model.LinearRegression),
    ("ElasticNet", sklearn.linear_model.ElasticNet),
    ("Tweedie", sklearn.linear_model.TweedieRegressor),
    ("SVR", sklearn.svm.SVR),
    ("NearestNeighbors", sklearn.neighbors.KNeighborsRegressor),
    ("Tree", sklearn.tree.DecisionTreeRegressor),
]

grids = {
    "Linear": None,
    "ElasticNet": {
        "alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 3],
        "l1_ratio": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    },
    "Tweedie": {"power": [0, 1, 2, 3], "alpha": [0, 1e-2, 1, 2, 3]},
    "SVR": {
        "kernel": ["linear", "poly", "rbf", "sigmoid"],
        "degree": [2, 3, 4, 5, 6],
    },
    "NearestNeighbors": {
        "n_neighbors": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "weights": ["uniform", "distance"],
        "algorithm": ["auto", "ball_tree", "kd_tree"],
    },
    "Tree": {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 3, 4, 5, 6],
        "min_samples_leaf": [1, 2, 3, 4, 5],
    },
}

results = {"model": [], "explained_variance": [], "mse": [], "mae": []}
for model in models:
    print(f"Fitting {model[0]}.")
    if grids[model[0]]:
        grid_instance = GridSearchCV(model[1](), grids[model[0]])
    else:
        grid_instance = model[1]()

    grid_instance.fit(x_train, np.squeeze(y_train))
    y_hat = grid_instance.predict(x_test)

    results["model"].append(model[0])
    results["explained_variance"].append(metrics["explained_variance"](y_test, y_hat))
    results["mae"].append(metrics["mae"](y_test, y_hat))
    results["mse"].append(metrics["mse"](y_test, y_hat))

# store results
pd.DataFrame(results).to_csv("results.csv", index=None)
