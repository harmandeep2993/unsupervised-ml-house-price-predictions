import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import xgboost as xgb

def single_model_results(X_train, y_train, X_test, y_test, model: str):
    
        # Train
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        # MAE
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_test_pred)

        # RMSE
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))

        print(f"Model Evaluation - {model}")
        print("\nR2 Score")
        print(f"Train: {r2_train:.4f}")
        print(f"Test : {r2_test:.4f}")

        print("\nMAE")
        print(f"Train: {mae_train:.4f}")
        print(f"Test : {mae_test:.4f}")

        # print("\nRMSE")
        # print(f"Train: {rmse_train:.4f}")
        # print(f"Test : {rmse_test:.4f}")

        print("\nOverfit Gap")
        print(f"Gap: {round((r2_test-r2_train),4)*100}%")

def compare_multiple_models(X_train, y_train, X_test, y_test, steps= str):
    """
    Train and evaluate multiple regression models.
    Returns a comparison DataFrame.
    """

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(),
        "Lasso Regression": Lasso(),
        "KNN Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": xgb.XGBRFRegressor()
    }

    results = []

    for name, model in models.items():

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        # Metrics
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_test_pred)

        results.append({
            "model_name": name,
            "data_steps": steps,
            "r2_train": round(r2_train, 4),
            "r2_test": round(r2_test, 4),
            "mae_train": round(mean_absolute_error(y_train, y_train_pred), 2),
            "mae_test": round(mean_absolute_error(y_test, y_test_pred), 2),
            # "rmse_train": round(np.sqrt(mean_squared_error(y_train, y_train_pred)), 4),
            # "rmse_test": round(np.sqrt(mean_squared_error(y_test, y_test_pred)), 4),
            "overfit_gap": round(r2_train - r2_test, 4)*100
        })

    return pd.DataFrame(results)
