"""
utils.py
--------------------------------------------------------------------
Utility functions for plotting, evaluation, and loading training history.
"""

# Import libraries
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from config import ARTIFACTS_DIR
from config import class_names as default_class_names

# Ensure artifacts directory exists
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

def save_history(history, filename='history.json'):
  """Save training history to a JSON file."""
  history_path=ARTIFACTS_DIR/filename
  with open(history_path, 'w') as f:
    json.dump(history.history, f)
  print(f"Training history saved to {history_path}")

def plot_history(history, filename='training_history.png'):
  """Plot training and validation accuracy/loss over epochs."""

  plt.figure(figsize=(12, 4))

  # Accuracy
  plt.subplot(1, 2, 1)
  plt.plot(history.history['accuracy'], label='Training Accuracy')
  plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend()

  # Loss
  plt.subplot(1, 2, 2)
  plt.plot(history.history['loss'], label='Training Loss')
  plt.plot(history.history['val_loss'], label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend()

  plt.tight_layout()
  plt.savefig(ARTIFACTS_DIR/filename)
  plt.show()
  plt.close()
  print(f"Training history plot saved to {ARTIFACTS_DIR/filename}")

def plot_confusion_matrix(y_true, y_pred, class_names=None, filename='confusion_matrix.png'):
  """
  Plot confusion matrix with heatmap visualization.

  Args:
    y_true (list or np.ndarray): True labels
    y_pred (list or np.ndarray): Predicted labels
    class_names (list): Class names
  """

  if class_names is None:
    class_names=default_class_names

  cm=confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(8, 6))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
  plt.xlabel('Predicted')
  plt.ylabel('True')
  plt.title('Confusion Matrix')
  plt.savefig(ARTIFACTS_DIR/filename)
  plt.show()
  plt.close()
  print(f"Confusion matrix plot saved to {ARTIFACTS_DIR/filename}")

def print_classification_report(y_true, y_pred, class_names=None):
  """Print classification report."""

  if class_names is None:
    class_names=default_class_names

  print("Classification Report:")
  report=classification_report(y_true, y_pred, target_names=class_names)
  print(report)

def plot_history_from_file(filename='history.json'):
  """Plot training history from a JSON file."""

  # Load JSON file
  with open(ARTIFACTS_DIR/filename, 'r') as f:
    history=json.load(f)

  plt.figure(figsize=(12, 4))

  # Accuracy
  plt.subplot(1, 2, 1)
  plt.plot(history['accuracy'], label='Training Accuracy')
  plt.plot(history['val_accuracy'], label='Validation Accuracy')
  plt.xlabel('Epochs')
  plt.ylabel('Accuracy')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend()

  # Loss
  plt.subplot(1, 2, 2)
  plt.plot(history['loss'], label='Training Loss')
  plt.plot(history['val_loss'], label='Validation Loss')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.grid(True, linestyle='--', alpha=0.6)
  plt.legend()

  # Save plot
  plt.tight_layout()
  plt.savefig(ARTIFACTS_DIR/'training_history_from_file.png')
  plt.show()
  plt.close()
  print(f"Training history plot saved to {ARTIFACTS_DIR/'training_history_from_file.png'}")
