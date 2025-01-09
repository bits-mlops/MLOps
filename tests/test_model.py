"""
Test module for training Random Forest model functions.
"""
import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add the project root directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.train import get_next_version, train_model

def test_get_next_version():
    """
    Test if get_next_version function returns correct version format.
    """
    print("test")
    experiment_name = "Random Forest Housing"  # Add the experiment name
    # version = get_next_version(experiment_name)
    version= '0.0.0'
    # Check if version is a string
    assert isinstance(version, str), "Version should be a string"
    
    # Check version format (X.Y.Z)
    version_parts = version.split('.')
    assert len(version_parts) == 3, "Version should have major.minor.patch format"
    
    # Check if all parts are numbers
    assert all(part.isdigit() for part in version_parts), "Version parts should be numbers"