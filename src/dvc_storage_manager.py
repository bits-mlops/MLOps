# storage_manager.py
from azure.storage.blob import BlobServiceClient
from typing import Dict, List, Optional
import json
import os
import shutil
from datetime import datetime
from config import Config
from typing import Tuple, Dict, Any

class StorageManager:
    def __init__(self):
        """Initialize Azure Blob Storage manager."""
        self.connection_string = Config.AZURE_STORAGE_CONNECTION_STRING
        if not self.connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")
            
        self.container_name = 'data'
        self.versions_file = 'versions.json'
        self.local_data_path = 'src/data'
        
        try:
            self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
            
            # Ensure container exists
            container_client = self.blob_service_client.get_container_client(self.container_name)
            if not container_client.exists():
                container_client.create_container()
                
        except Exception as e:
            raise Exception(f"Failed to initialize Azure Blob Storage: {str(e)}")

    def _download_blob(self, blob_name: str, destination: str) -> None:
        """Download a blob to local file."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with open(destination, "wb") as file:
                data = blob_client.download_blob()
                file.write(data.readall())
        except Exception as e:
            raise Exception(f"Failed to download blob {blob_name}: {str(e)}")

    def _upload_blob(self, source: str, blob_name: str) -> None:
        """Upload a local file to blob."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )
            with open(source, "rb") as file:
                blob_client.upload_blob(file, overwrite=True)
        except Exception as e:
            raise Exception(f"Failed to upload blob {blob_name}: {str(e)}")

    def _get_versions(self) -> Dict:
        """Get versions info from blob storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.versions_file
            )
            try:
                versions_str = blob_client.download_blob().readall()
                return json.loads(versions_str)
            except Exception:
                # If versions file doesn't exist or is invalid
                return {"versions": [], "latest_version": None}
        except Exception as e:
            raise Exception(f"Failed to get versions: {str(e)}")

    def _save_versions(self, versions: Dict) -> None:
        """Save versions info to blob storage."""
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=self.versions_file
            )
            blob_client.upload_blob(json.dumps(versions, indent=2), overwrite=True)
        except Exception as e:
            raise Exception(f"Failed to save versions: {str(e)}")

    def initialize_storage(self) -> Dict[str, str]:
        """Initialize storage with local dataset if no versions exist."""
        try:
            versions = self._get_versions()
            
            if not versions["versions"]:
                # Check if local file exists
                local_file = os.path.join(self.local_data_path, 'housing.csv')
                if not os.path.exists(local_file):
                    raise FileNotFoundError(f"Initial dataset not found at {local_file}")
                
                # Upload initial dataset
                initial_version = "0.0.0"
                blob_name = f"housing_{initial_version}.csv"
                
                self._upload_blob(local_file, blob_name)
                
                # Create initial version info
                versions["versions"].append({
                    "version": initial_version,
                    "filename": blob_name,
                    "created_at": datetime.now().isoformat(),
                    "current": True
                })
                versions["latest_version"] = initial_version
                self._save_versions(versions)
                
                return {"status": "success", "initialized_version": initial_version}
            else:
                # Download latest version to local
                latest_version = versions["latest_version"]
                latest_file = next(v["filename"] for v in versions["versions"] 
                                 if v["version"] == latest_version)
                
                # Ensure local data directory exists
                os.makedirs(self.local_data_path, exist_ok=True)
                
                self._download_blob(
                    latest_file,
                    os.path.join(self.local_data_path, 'housing.csv')
                )
                
                return {"status": "success", "current_version": latest_version}
                
        except Exception as e:
            raise Exception(f"Failed to initialize storage: {str(e)}")
        
    def upload_dataset(self, file_path: str) -> Dict[str, str]:
        """Upload new dataset version."""
        versions = self._get_versions()
        current_version = versions["latest_version"] or "0.0.0"
        
        # Calculate next version
        v_parts = [int(x) for x in current_version.split('.')]
        next_version = f"{v_parts[0]}.{v_parts[1]}.{v_parts[2] + 1}"
        
        # Upload new version
        blob_name = f"housing_{next_version}.csv"
        self._upload_blob(file_path, blob_name)
        
        # Update versions
        for v in versions["versions"]:
            v["current"] = False
            
        versions["versions"].append({
            "version": next_version,
            "filename": blob_name,
            "created_at": datetime.now().isoformat(),
            "current": True
        })
        versions["latest_version"] = next_version
        self._save_versions(versions)
        
        # Update local file
        shutil.copy2(file_path, os.path.join(self.local_data_path, 'housing.csv'))
        
        return {
            "status": "success",
            "version": next_version,
            "filename": blob_name
        }

    def get_versions(self) -> Dict[str, Any]:
        """Get all dataset versions."""
        versions = self._get_versions()
        return {
            "versions": versions["versions"],
            "latest_version": versions["latest_version"]
        }

    def load_version(self, version: str) -> Dict[str, str]:
        """Load specific version to local storage."""
        versions = self._get_versions()
        version_info = next((v for v in versions["versions"] 
                           if v["version"] == version), None)
        
        if not version_info:
            raise ValueError(f"Version {version} not found")
        
        # Download specified version
        self._download_blob(
            version_info["filename"],
            os.path.join(self.local_data_path, 'housing.csv')
        )
        
        # Update current version
        for v in versions["versions"]:
            v["current"] = (v["version"] == version)
            
        self._save_versions(versions)
        print(versions)
        
        return {
            "status": "success",
            "loaded_version": version
        }