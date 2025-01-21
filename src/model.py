from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
from typing import Tuple, Dict, Any
import mlflow
import mlflow.sklearn
from utils import get_next_version

def perform_grid_search(features_train: pd.DataFrame,
                       target_train: pd.Series,
                       param_grid: Dict[str, Any],
                       random_state: int = 42) -> Tuple[Dict[str, Any], RandomForestRegressor]:
    """Perform grid search to find optimal hyperparameters for the Random Forest model."""
    print("grid search tuning started")
    # param_grid = {
    #     'n_estimators': [10, 30, 50, 100],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4]
    # }
    rf_model = RandomForestRegressor(random_state=random_state)
    grid_search = GridSearchCV(
        estimator=rf_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    grid_search.fit(features_train, target_train)
    print("grid search tuning ended")
    return grid_search.best_params_, grid_search.best_estimator_

def train_model_with_tuning(features_train: pd.DataFrame, 
                           features_test: pd.DataFrame,
                           target_train: pd.Series,
                           target_test: pd.Series,
                           param_grid: Dict[str, Any],
                           random_state: int = 42) -> Tuple[RandomForestRegressor, float, float, Dict[str, Any]]:
    """Train a Random Forest model with hyperparameter tuning and calculate performance metrics."""
    print("model training started")
    best_params, best_model = perform_grid_search(features_train, target_train, param_grid, random_state)
    predictions = best_model.predict(features_test)
    mse_score = mean_squared_error(target_test, predictions)
    r2_score_val = r2_score(target_test, predictions)
    print("model training ended")
    return best_model, mse_score, r2_score_val, best_params

def retune_model(features_train: pd.DataFrame,
                  features_test: pd.DataFrame,
                  target_train: pd.Series,
                  target_test: pd.Series,
                  hyperparameters: Dict[str, Any]) -> Tuple[RandomForestRegressor, float, float]:
    # Retune an existing model with new hyperparameters.
    
    print("model retuning started")
    rf_model = RandomForestRegressor(**hyperparameters)
    rf_model.fit(features_train, target_train)
    predictions = rf_model.predict(features_test)
    mse = mean_squared_error(target_test, predictions)
    r2e = r2_score(target_test, predictions)
    print("model retuning ended")
    return rf_model, mse, r2e

def train_model_without_tuning(features_train: pd.DataFrame,
                             target_train: pd.Series,
                             features_test: pd.DataFrame,
                             target_test: pd.Series,
                             n_estimators: int,
                             random_state: int) -> Tuple[RandomForestRegressor, float, float]:
    """Train a Random Forest model without hyperparameter tuning."""
    print("model training started")
    rf_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
    rf_model.fit(features_train, target_train)
    predictions = rf_model.predict(features_test)
    mse = mean_squared_error(target_test, predictions)
    r2e = r2_score(target_test, predictions)
    print("model training ended")
    return rf_model, mse, r2e