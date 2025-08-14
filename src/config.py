"""
config.py
---------
Centralized configuration for the MNIST Handwritten Digit Recognition project.

- Creates required directories
- Stores dataset/model parameters
- Ensures reproducibility
- Provides class names
"""

# Import Libraries
from pathlib import Path
import numpy as np
import random
import tensorflow as tf
import json

# Base Project Directory
PROJECT_DIR=Path.cwd()

# Data directories
DATA_DIR=PROJECT_DIR/'data'
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Model directory
MODEL_DIR=PROJECT_DIR/'models'

# Artifacts directory (plots, logs, reports, etc.)
ARTIFACTS_DIR=PROJECT_DIR/'artifacts'

# Model Hyperparameters
PARAMS={
  'img_height':28,
  'img_width':28,
  'batch_size':64,
  'channels':1,
  'num_classes':10,
  'epochs':12,
  'validation_split':0.1,
  'learning_rate':1e-3,
}

# Class names for MNIST (0-9)
class_names=[str(i) for i in range(PARAMS['num_classes'])]

# Reproducibility
SEED=42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

# Utility Functions
def create_dirs():
  """Create all required project directories."""
  for dir in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, ARTIFACTS_DIR]:
    if not dir.exists():
      dir.mkdir(parents=True, exist_ok=True)
      print(f'Created: {dir}')
    else:
      print(f'Directory already exists: {dir}')

def save_params(file_path=ARTIFACTS_DIR/'config.json'):
  """Save PARAMS to a JSON file for reproducibility."""
  file_path.parent.mkdir(parents=True, exist_ok=True)
  with open(file_path, 'w') as fp:
    json.dump(PARAMS, fp, indent=4)
  print(f'Saved configuration to: {file_path}')

# Script Execution
if __name__=='__main__':
  print('Setting up Handwritten Digit Recognizer project....')
  create_dirs()
  save_params()
  print('Setup Complete!')
