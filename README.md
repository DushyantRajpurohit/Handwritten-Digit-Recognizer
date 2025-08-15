# Handwritten Digit Recognizer

A complete pipeline for training, evaluating, and deploying a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. Includes a **Streamlit** web app with a drawing canvas and image upload support for real‑time digit prediction.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Features](#features)
* [Installation](#-installation)
* [Training](#-training)
* [Evaluation](#-evaluation)
* [Streamlit App](#-streamlit-app)
* [CLI Prediction](#-cli-prediction)
* [Requirements](#requirements)
* [Example Results](#-example-results)
* [Troubleshooting & Tips](#troubleshooting--tips)
* [License](#-license)

---

## Project Structure

```
handwritten-digit-recognizer/
│
├── data/
│   ├── raw/                # Original unprocessed data (e.g., MNIST raw files)
│   └── processed/          # Preprocessed datasets (train/test split)
│
├── notebooks/
│   └── Handwritten_Digit_Recognizer.ipynb
│
├── src/
│   ├── config.py           # Configurations (paths, hyperparameters, constants)
│   ├── data_loader.py      # Functions to load and preprocess data
│   ├── model.py            # Model architecture definition (CNN)
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Model evaluation
│   ├── predict.py          # Prediction utilities + CLI
│   └── utils.py            # Helper functions (plotting, metrics, etc.)
│
├── app/
│   └── app.py              # Streamlit interface
│   
├── models/
│   └── best_model.h5       # Saved trained model (tracked with Git LFS)
│
├── artifacts/
│   ├── augmented_sample.png
│   ├── class_distribution.png
│   ├── confusion_matrix.png
│   ├── history.json
│   ├── misclassified_example.png
│   ├── prediction_result.png
│   ├── sample_images.png
│   └── training_history.png
│
├── requirements.txt        # All dependencies
├── README.md               # Project overview, usage, and results
└── LICENSE
```

---

## Features

* **Data Handling**

  * Automatic MNIST download & preprocessing
  * Raw and processed data caching for faster reloads
* **Model**

  * CNN with `Conv2D → BatchNorm → ReLU → MaxPool → Dropout` blocks
  * Adam optimizer with learning‑rate scheduling
* **Training**

  * Data augmentation (`ImageDataGenerator`)
  * Early stopping, `ReduceLROnPlateau`, and model checkpoint
  * History saved to JSON + training curves
* **Evaluation**

  * Accuracy & loss, Confusion Matrix, Classification Report
* **Deployment**

  * **Streamlit app** with: drawing canvas, image upload, real‑time prediction + probability chart

---

## 🚀 Installation

```bash
# 1) Clone
git clone https://github.com/your-username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer

# 2) Create & activate virtual environment
python -m venv venv
# Mac/Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1

# 3) Install dependencies
pip install -r requirements.txt
```

Optional but recommended:

```bash
# track large model files with Git LFS (prevents >100MB push errors)
git lfs install
git lfs track "models/*.h5"
```

---

## 📊 Training

```bash
python -m src.train
```

This will:

* Download & preprocess MNIST (if not already cached)
* Train the CNN with data augmentation
* Save the best model to `models/best_model.h5`
* Save plots and metrics to `artifacts/`

**Tip:** Set seeds for reproducibility via `src/config.py`.

---

## 🧪 Evaluation

```bash
python -m src.evaluate
```

This will:

* Load the best saved model
* Evaluate on the test set
* Save/Display accuracy, confusion matrix, and classification report

---

## 🌐 Streamlit App

```bash
streamlit run app/app.py
```

Features:

* Draw digits in a canvas and get predictions
* Upload PNG/JPG for recognition
* View prediction probabilities in a bar chart

---

## 🖥️ CLI Prediction

Use the included CLI to predict a single image from the terminal.

```bash
python -m src.predict path/to/image.png
# or
python src/predict.py path/to/image.png
```

**Supported inputs:** grayscale or RGB images; the script will convert, center‑pad, resize to 28×28, and normalize to `[0,1]`.

---

## Requirements

Pinned to stable versions known to work on CPU and most GPUs. Adjust if you need newer CUDA/cuDNN.

```
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.10.0
pillow>=8.0.0
streamlit-drawable-canvas>=0.8.0
```

> If you face binary issues on Linux with GPU, consider installing the matching `tensorflow` build and CUDA/cuDNN versions from the official install guide.

---

## 📈 Example Results

* Test Accuracy: \~**99%** (depends on hyperparameters)
* Confusion Matrix: shows strong diagonals with sparse off‑diagonals
* Example prediction:

| Input Digit | Prediction | Confidence |
| ----------- | ---------- | ---------- |
| 🖌️ 5       | 5          | 0.998      |

---

## Troubleshooting & Tips

* **Large repo pushes (>100MB)**: use Git LFS for `models/*.h5` and `artifacts/*.png`.
* **Matplotlib not showing in scripts**: save figures to `artifacts/` instead of `plt.show()` in non‑interactive runs.
* **VS Code & notebooks**: install the Python + Jupyter extensions; select the `venv` interpreter.
* **Determinism**: set NumPy/TensorFlow seeds in `config.py` and disable GPU nondeterminism if needed.
* **Docker**: create an image with `requirements.txt` to ensure reproducibility.

---

## 📜 License

MIT License – Free to use and modify.

---

## `src/predict.py` (drop‑in ready)

> CLI + importable functions. Handles grayscale/RGB, auto‑centering, 28×28 resizing, normalization, and outputs both class and probabilities.

```python
# src/predict.py
import argparse
import os
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Optional: lazy import for speed when used as a library
try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = tf.keras.models.load_model

DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.h5")


def load_digit_model(model_path: str = DEFAULT_MODEL_PATH):
    """Load a trained Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return load_model(model_path)


def preprocess_image(img_path: str) -> np.ndarray:
    """Load image, convert to 28x28 grayscale, normalize to [0,1], shape (1, 28, 28, 1)."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image file not found: {img_path}")

    img = Image.open(img_path).convert("L")  # grayscale

    # Invert if background is dark and digit is light (heuristic):
    # Many digit images have white background and black digit (MNIST-like).
    # We normalize to that convention.
    if np.mean(img) < 128:
        img = ImageOps.invert(img)

    # Resize with keeping aspect ratio, then center-pad to 28x28
    img.thumbnail((26, 26), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), color=0)
    left = (28 - img.width) // 2
    top = (28 - img.height) // 2
    canvas.paste(img, (left, top))

    arr = np.asarray(canvas, dtype=np.float32) / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


def predict_digit(model, processed_img: np.ndarray):
    """Return (pred_class:int, probabilities:np.ndarray[10])."""
    probs = model.predict(processed_img, verbose=0)[0]
    pred = int(np.argmax(probs))
    return pred, probs


def save_barplot(probs: np.ndarray, out_path: str):
    """Save a simple probability bar plot to artifacts/prediction_result.png."""
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure()
    plt.bar(range(10), probs)
    plt.xlabel("Digit")
    plt.ylabel("Probability")
    plt.title("Prediction Probabilities")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Predict a handwritten digit from an image")
    parser.add_argument("image", help="Path to input image (PNG/JPG)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to Keras .h5 model")
    parser.add_argument("--save-plot", action="store_true", help="Save probability bar plot to artifacts/")
    args = parser.parse_args()

    model = load_digit_model(args.model)
    x = preprocess_image(args.image)
    pred, probs = predict_digit(model, x)

    print({"prediction": pred, "probabilities": probs.tolist()})

    if args.save_plot:
        out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts", "prediction_result.png")
        save_barplot(probs, out_path)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
```

**Usage examples**

```bash
python -m src.predict samples/my_digit.png
python -m src.predict samples/my_digit.png --save-plot
python -m src.predict samples/my_digit.png --model models/best_model.h5
```

---

### Optional: `requirements.txt` (drop‑in)

```text
tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.10.0
pillow>=8.0.0
streamlit-drawable-canvas>=0.8.0
```

> Pin exact versions for full reproducibility if needed (e.g., `tensorflow==2.13.0`, etc.).

