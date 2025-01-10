from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

class Config:
    TENANT_ID = os.environ.get('AZURE_TENANT_ID')
    CLIENT_ID = os.environ.get('AZURE_CLIENT_ID')
    CLIENT_SECRET = os.environ.get('AZURE_CLIENT_SECRET')
    SUBSCRIPTION_ID = os.environ.get('AZURE_SUBSCRIPTION_ID')
    RESOURCE_GROUP = os.environ.get('AZURE_RESOURCE_GROUP')
    WORKSPACE_NAME = os.environ.get('AZURE_WORKSPACE_NAME')
    AZURE_STORAGE_CONNECTION_STRING = os.environ.get('AZURE_STORAGE_CONNECTION_STRING')
    DEFAULT_EXPERIMENT_NAME = 'Random Forest Housing'
    DEFAULT_DATA_PATH = 'src/data/housing.csv'
    DEFAULT_RANDOM_STATE = 42
    DEFAULT_N_ESTIMATORS = 20