"""
model_builder.py
----------------
Defines the Convolutional Neural Network (CNN) architecture for MNIST classification.
"""

# Import Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from config import PARAMS

def build_cnn_model(input_shape=None, num_classes=None, learning_rate=None, show_summary=True):
  """
  Build and compile a Convolutional Neural Network (CNN) for MNIST digit classification.

  Args:
    input_shape (tuple, optional): Shape of the input images (H, W, C).
    num_classes (int, optional): Number of output classes.
    learning_rate (float, optional): Learning rate for optimizer.
    show_summary (bool, optional): If True, prints model summary.

  Returns:
    model (Sequential): Compiled CNN model.
  """

  if input_shape is None:
    input_shape=(PARAMS['img_height'],
                 PARAMS['img_width'],
                 PARAMS['channels'])

  if num_classes is None:
    num_classes=PARAMS['num_classes']

  if learning_rate is None:
    learning_rate=PARAMS['learning_rate']

  model=Sequential([
    Conv2D(32, (3, 3), activation='relu', padding='same',
           kernel_regularizer=regularizers.l2(1e-4), input_shape=input_shape),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu', padding='same',
           kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu', padding='same',
           kernel_regularizer=regularizers.l2(1e-4)),
    BatchNormalization(),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),

    Dense(num_classes, activation='softmax')
  ])

  # Compile model
  model.compile(optimizer=Adam(learning_rate=learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy'])

  if show_summary:
    print("CNN Model Summary:")
    model.summary()

  return model

if __name__ == "__main__":
  # Quick test build
  build_cnn_model()
