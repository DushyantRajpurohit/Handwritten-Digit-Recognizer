"""
train.py
-----------
Train a deep learning model for handwritten digit recognition.
"""

# Import libraries
import json
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from config import PARAMS, MODEL_DIR, SEED, ARTIFACTS_DIR
from data_loader import load_processed_data
from model import build_cnn_model
from utils import save_history, plot_history, plot_confusion_matrix, plot_history_from_file
from evaluate import evaluate_model
from predict import predict_single_image

# Ensure dirs exist
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Set random seeds for reproducibility
np.random.seed(SEED)

def train_model():
  # Load processed data
  X_train, y_train, X_test, y_test=load_processed_data()

  # Data augmentation
  datagen=ImageDataGenerator(
   rotation_range=12,
   width_shift_range=0.08,
   height_shift_range=0.08,
   zoom_range=0.08,
   shear_range=0.06,
   validation_split=PARAMS['validation_split']
  )

  # Build CNN model
  model=build_cnn_model()

  print('-'*100)

  # Callbacks
  checkpoint_path=MODEL_DIR/'best_model.h5'

  cb=[
   EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
   ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
   ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True)
  ]

  # Generators
  train_gen=datagen.flow(X_train, y_train, batch_size=PARAMS['batch_size'], subset='training', seed=SEED)
  val_gen=datagen.flow(X_train, y_train, batch_size=PARAMS['batch_size'], subset='validation', seed=SEED)

  steps_per_epoch=len(train_gen)
  validation_steps=len(val_gen)

  # Train the model
  print('Training the model')
  history=model.fit(
   train_gen,
   validation_data=val_gen,
   epochs=PARAMS['epochs'],
   callbacks=cb,
   steps_per_epoch=steps_per_epoch,
   validation_steps=validation_steps
  )

  print('-'*100)

  # Save history
  print('Saving history')
  save_history(history)
  print('-'*100)

  print('Training complete. Best model saved at:', checkpoint_path)
  print('-'*100)

  # Evaluating the model on test dataset
  evaluate_model()
  
  # Prediction on a test image
  print('Prediction on a test image:')
  predict_single_image()

if __name__ == '__main__':
  train_model()
