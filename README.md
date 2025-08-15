# Handwritten Digit Recognizer

A complete pipeline for training, evaluating, and deploying a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset. Includes a **Streamlit** web app with a drawing canvas and image upload support for real‑time digit prediction.

---

## Table of Contents

* [Project Structure](#project-structure)
* [Features](#features)
* [Installation](#-installation)
* [Training](#-training)
* [Streamlit App](#-streamlit-app)
* [Requirements](#requirements)
* [Example Results](#-example-results)
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
* Evaluate on the test set
* Save/Display accuracy, confusion matrix, and classification report

**Tip:** Set seeds for reproducibility via `src/config.py`.

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

## Requirements

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

---

## 📈 Example Results

* Test Accuracy: \~**99%** (depends on hyperparameters)
* Confusion Matrix: shows strong diagonals with sparse off‑diagonals
* Example prediction:

| Input Digit | Prediction | Confidence |
| ----------- | ---------- | ---------- |
| 🖌️ 5       | 5          | 0.998      |

---

## 📜 License

MIT License – Free to use and modify.
