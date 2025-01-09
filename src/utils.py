import pandas as pd
import mlflow
from mlflow.client import MlflowClient
from azure.ai.ml import MLClient
from azure.identity import ClientSecretCredential
from sklearn.model_selection import train_test_split
from packaging import version
import traceback
from typing import Tuple, Dict, Any
from config import Config

def init_azure_ml() -> bool:
    """Initialize Azure ML client"""
    try:
        credential = ClientSecretCredential(
            tenant_id=Config.TENANT_ID,
            client_id=Config.CLIENT_ID,
            client_secret=Config.CLIENT_SECRET
        )
        
        ml_client = MLClient(
            credential,
            subscription_id=Config.SUBSCRIPTION_ID,
            resource_group_name=Config.RESOURCE_GROUP,
            workspace_name=Config.WORKSPACE_NAME
        )
        tracking_uri = ml_client.workspaces.get(Config.WORKSPACE_NAME).mlflow_tracking_uri
        # tracking_uri='http://localhost:5000'
        mlflow.set_tracking_uri(tracking_uri)
        return True
    except Exception as e:
        print(f"Failed to initialize Azure ML: {traceback.format_exc()}")
        return False

def load_and_split_data(data_path: str, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Load housing data and split it into training and test sets."""
    print("data load initiated")
    data = pd.read_csv(data_path)
    features = data.drop(['Address'], axis=1)
    target = data.Price.copy()
    print("data load completed")
    return train_test_split(features, target, test_size=test_size, random_state=random_state)
    
def get_next_version(experiment_name: str) -> str:
    """Get the next version number for model versioning."""
    print("getting version")
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        mlflow.create_experiment(experiment_name)
        return "0.0.0"
    
    client = MlflowClient()
    experiment_id = experiment.experiment_id
    runs = client.search_runs(experiment_id)
    versions = []
    
    for run in runs:
        try:
            ver = run.data.tags.get("version", "0.0.0")
            versions.append(ver)
        except ValueError as e:
            print(f"Error parsing version: {e}")
            continue

    if not versions:
        return "0.0.1"

    latest = max(versions, key=version.parse)
    v = version.parse(latest)
    print("current verision",v)
    return f"{v.major}.{v.minor}.{v.micro + 1}"