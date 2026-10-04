import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from preprocessing import make_preprocessor, ALL_INPUT_FEATURES, TARGET_FEATURE


def train_and_evaluate_models(data, test_size=0.20, random_state=42):
    """
    Trains Linear Regression and Random Forest models on the dataset.
    Uses 80/20 train/test split with fixed random_state.
    Calculates actual MAE, RMSE, and R² scores dynamically.
    """
    X = data[ALL_INPUT_FEATURES]
    y = data[TARGET_FEATURE]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    models = {
        "Linear Regression": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", LinearRegression())
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", make_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            ))
        ])
    }

    metrics_list = []
    predictions = {}

    for name, pipeline in models.items():
        # Fit pipeline
        pipeline.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = pipeline.predict(X_test)
        predictions[name] = y_pred

        # Compute dynamic evaluation metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics_list.append({
            "Model": name,
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R² Score": round(r2, 3)
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Determine best model dynamically based on R² Score
    best_row = metrics_df.loc[metrics_df["R² Score"].idxmax()]
    best_model_name = best_row["Model"]

    test_results = {
        "X_test": X_test,
        "y_test": y_test,
        "predictions": predictions
    }

    return models, metrics_df, test_results, best_model_name
