# Import libraries
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_processed_data
from utils import plot_confusion_matrix, print_classification_report, plot_history_from_file
from config import MODEL_DIR, class_names

def evaluate_model():
  """Evaluate the saved model on test data"""

  # Load processed data
  _, _, X_test, y_test = load_processed_data()

  # Load best saved model
  model_path=MODEL_DIR/'best_model.h5'
  model=load_model(model_path)
  print(f'Loaded model from {model_path}')
  print('-'*100)

  # Evaluate
  loss, accuracy=model.evaluate(X_test, y_test, verbose=0)
  print(f'Test loss: {loss:.4f}')
  print(f'Test accuracy: {accuracy*100:.2f}%')
  print('-'*100)

  # Predictions
  y_pred=np.argmax(model.predict(X_test), axis=1)
  y_true=np.argmax(y_test, axis=1)

  # Confusion matrix
  print('Confusion matrix:')
  plot_confusion_matrix(y_true, y_pred, class_names)
  print('-'*100)

  # Classification report
  print('Classification report:')
  print_classification_report(y_true, y_pred, class_names)
  print('-'*100)

  # Also plot hisotry from saved JSON
  print('History from JSON:')
  plot_history_from_file()


if __name__=="__main__":
  evaluate_model()
