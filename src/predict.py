# predict.py
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from data_loader import load_processed_data
from config import MODEL_DIR, class_names, SEED, ARTIFACTS_DIR

def predict_single_image():
    # Load processed data
    _, _, X_test, y_test=load_processed_data()

    # Load best saved model
    model_path=MODEL_DIR / 'best_model.h5'
    model=load_model(model_path)
    print(f"Loaded model from {model_path}")

    # Pick a random image from the test set
    np.random.seed(SEED)
    idx=np.random.randint(0, len(X_test))
    image=X_test[idx]
    true_label = np.argmax(y_test[idx])

    # Predict
    pred_probs=model.predict(image[np.newaxis, ...])
    pred_label=np.argmax(pred_probs)

    print(f"True Label: {class_names[true_label]}")
    print(f"Predicted Label: {class_names[pred_label]}")
    print(f"Prediction Probabilities: {pred_probs}")

    # Plot the image
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title(f"True: {class_names[true_label]}, Pred: {class_names[pred_label]}")
    plt.axis('off')
    save_path=ARTIFACTS_DIR/'prediction_result.png'
    plt.savefig(save_path, dpi=300)
    plt.show()

if __name__ == "__main__":
    predict_single_image()
