from flask import Flask, request, jsonify
from utils import init_azure_ml, load_and_split_data, get_next_version
from model import train_model_with_tuning, train_model_without_tuning
from config import Config
import mlflow
from mlflow.client import MlflowClient
import pandas as pd
import traceback
from typing import Tuple, Dict, Any

app = Flask(__name__)
  
@app.route('/health', methods=['GET'])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return jsonify({"status": "healthy"})

@app.route('/trainWithTuning', methods=['POST'])
def train_with_tuning() -> Dict[str, Any]:
    """Train model endpoint with hyperparameter tuning."""
    print("train started")
    try:
        data = request.get_json()
        data_path = data.get('data_path', Config.DEFAULT_DATA_PATH)
        random_state = data.get('random_state', Config.DEFAULT_RANDOM_STATE)
        experiment_name = data.get('experiment_name', Config.DEFAULT_EXPERIMENT_NAME)

        if not init_azure_ml():
            return jsonify({"error": "Failed to initialize Azure ML"}), 500

        X_train, X_test, y_train, y_test = load_and_split_data(data_path)
        current_version = get_next_version(experiment_name)
        
        print("mlflow setting experiment - ",experiment_name)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"my_model_v{current_version}"):
            model, mse, r2e, best_params = train_model_with_tuning(
                X_train, X_test, y_train, y_test, random_state=random_state
            )
            
            for param_name, param_value in best_params.items():
                mlflow.log_param(param_name, param_value)
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2e", r2e)
            mlflow.sklearn.log_model(model, "random_forest_housing")
            mlflow.set_tag("version", current_version)
            print("new model version-",current_version)
            return jsonify({
                "status": "success",
                "version": current_version,
                "best_parameters": best_params,
                "metrics": {
                    "mse": mse,
                    "r2e": r2e
                }
            })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/trainWithoutTuning', methods=['POST'])
def train_without_tuning() -> Dict[str, Any]:
    """Train model endpoint without hyperparameter tuning."""
    print("train started")
    try:
        data = request.get_json()
        data_path = data.get('data_path', Config.DEFAULT_DATA_PATH)
        random_state = data.get('random_state', Config.DEFAULT_RANDOM_STATE)
        experiment_name = data.get('experiment_name', Config.DEFAULT_EXPERIMENT_NAME)
        n_estimators = data.get('n_estimators', Config.DEFAULT_N_ESTIMATORS)

        if not init_azure_ml():
            return jsonify({"error": "Failed to initialize Azure ML"}), 500

        X_train, X_test, y_train, y_test = load_and_split_data(data_path)
        current_version = get_next_version(experiment_name)
        print("mlflow setting experiment - ",experiment_name)
        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(run_name=f"my_model_v{current_version}"):
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("random_state", random_state)

            model, mse, r2e = train_model_without_tuning(
                X_train, y_train, X_test, y_test, n_estimators, random_state
            )
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2e", r2e)
            mlflow.sklearn.log_model(model, "random_forest_housing")
            mlflow.set_tag("version", current_version)
            print("new model version-",current_version)
            
            return jsonify({
                "status": "success",
                "version": current_version,
                "metrics": {
                    "mse": mse,
                    "r2e": r2e
                }
            })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

@app.route('/predict', methods=['POST'])
def predict() -> Dict[str, Any]:
    """Prediction endpoint."""
    try:
        if not init_azure_ml():
            return jsonify({"error": "Failed to initialize Azure ML"}), 500
        
        data = request.get_json()
        features = pd.DataFrame([data['features']])
        experiment_name = data.get('experiment_name', Config.DEFAULT_EXPERIMENT_NAME)
        
        client = MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            return jsonify({"error": "Experiment not found"}), 404
        
        runs = client.search_runs(experiment.experiment_id)
        if not runs:
            return jsonify({"error": "No trained models found"}), 404
            
        latest_run = sorted(runs, key=lambda x: x.info.start_time or 0, reverse=True)[0]
        model_uri = f"runs:/{latest_run.info.run_id}/random_forest_housing"
        model = mlflow.sklearn.load_model(model_uri)
        
        prediction = model.predict(features)[0]
        
        return jsonify({
            "prediction": prediction,
            "model_version": latest_run.data.tags.get("version", "unknown")
        })

    except Exception as e:
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)