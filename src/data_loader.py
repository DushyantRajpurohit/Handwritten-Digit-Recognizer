"""
data_loader.py
--------------
Handles downloading, saving, processing, and loading the MNIST dataset.
Auto-detects when re-download or reprocessing is needed.
"""

# Import Libraries
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PARAMS

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _print_shapes(X_train, y_train, X_test, y_test, prefix=""):
    """Helper function to print dataset shapes."""
    print(f"{prefix} Training: {X_train.shape}, Labels: {y_train.shape}")
    print(f"{prefix} Testing:  {X_test.shape}, Labels: {y_test.shape}")

def raw_data_exists():
  """Check if raw MNIST files exist."""
  return all((RAW_DATA_DIR / fname).exists() for fname in [
    "X_train_raw.npy", "y_train_raw.npy", "X_test_raw.npy", "y_test_raw.npy"
  ])

def processed_data_exists():
    """Check if processed MNIST files exist."""
    return all((PROCESSED_DATA_DIR / fname).exists() for fname in [
      "X_train.npy", "y_train.npy", "X_test.npy", "y_test.npy"
    ])

def save_raw_data():
  """Download MNIST dataset and save as raw numpy arrays."""

  print("Downloading MNIST dataset...")
  (X_train, y_train), (X_test, y_test) = mnist.load_data()

  # Save raw data
  np.save(RAW_DATA_DIR/'X_train_raw.npy', X_train)
  np.save(RAW_DATA_DIR/'y_train_raw.npy', y_train)
  np.save(RAW_DATA_DIR/'X_test_raw.npy', X_test)
  np.save(RAW_DATA_DIR/'y_test_raw.npy', y_test)
  print('Raw data saved')

  # Check the shape of the dataset
  _print_shapes(X_train, y_train, X_test, y_test, prefix="Raw")

def process_and_save():
  """Normalize, reshape, one-hot encode labels, and save processed data."""
  print("Processing MNIST data...")

  # Load raw data
  X_train=np.load(RAW_DATA_DIR/'X_train_raw.npy')
  y_train=np.load(RAW_DATA_DIR/'y_train_raw.npy')
  X_test=np.load(RAW_DATA_DIR/'X_test_raw.npy')
  y_test=np.load(RAW_DATA_DIR/'y_test_raw.npy')

  # Normalize to range [0, 1]
  X_train=X_train.astype('float32')/255.0
  X_test=X_test.astype('float32')/255.0

  # Reshape the data to add a channel dimension (for grayscale)
  X_train=X_train.reshape(-1, PARAMS['img_height'], PARAMS['img_width'], PARAMS['channels'])
  X_test=X_test.reshape(-1, PARAMS['img_height'], PARAMS['img_width'], PARAMS['channels'])

  # One-hot encode labels
  y_train=to_categorical(y_train, PARAMS['num_classes'])
  y_test=to_categorical(y_test, PARAMS['num_classes'])

  # Save processed data
  np.save(PROCESSED_DATA_DIR/'X_train.npy', X_train)
  np.save(PROCESSED_DATA_DIR/'y_train.npy', y_train)
  np.save(PROCESSED_DATA_DIR/'X_test.npy', X_test)
  np.save(PROCESSED_DATA_DIR/'y_test.npy', y_test)
  print('Processed data saved')

  # Check the shape of the dataset
  _print_shapes(X_train, y_train, X_test, y_test, prefix="Processed")

def load_processed_data(force_reload=False):
  """
  Load MNIST dataset.
  - If missing raw data, downloads it.
  - If missing processed data, processes it.
  """

  # Load and process the data
  if not raw_data_exists():
    save_raw_data()
  else:
    print("Raw data already exists")
  print('-'*100)

  if not processed_data_exists():
    process_and_save()
  else:
    print("Processed data already exists")
  print('-'*100)

  """Load processed MNIST dataset."""
  X_train=np.load(PROCESSED_DATA_DIR/'X_train.npy')
  y_train=np.load(PROCESSED_DATA_DIR/'y_train.npy')
  X_test=np.load(PROCESSED_DATA_DIR/'X_test.npy')
  y_test=np.load(PROCESSED_DATA_DIR/'y_test.npy')

  return X_train, y_train, X_test, y_test

if __name__=="__main__":
  print("Preparing MNIST data...")
  load_processed_data()
  print("Data preparation complete!")
