# 🖊️ Handwritten Digit Recognizer

A complete pipeline for training, evaluating, and deploying a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset.  
Includes a **Streamlit** web app with a drawing canvas and image upload support for real-time digit prediction.

---

## 📂 Project Structure

.
├── app.py # Streamlit web app for digit prediction
├── config.py # Centralized configuration & directory setup
├── data_loader.py # MNIST dataset download, preprocessing, saving/loading
├── evaluate.py # Model evaluation on test data
├── model.py # CNN model architecture
├── train.py # Training pipeline with augmentation & callbacks
├── utils.py # Utility functions (plots, metrics, history saving)
├── data/
│ ├── raw/ # Raw dataset (.npy files)
│ ├── processed/ # Processed dataset (.npy files)
├── models/ # Saved trained models (best_model.h5)
├── artifacts/ # Plots, metrics reports, training history


---

## ⚙️ Features

- **Data Handling**
  - Automatic MNIST download & preprocessing.
  - Raw and processed data caching for faster reloads.
- **Model**
  - CNN with Conv2D, BatchNorm, MaxPooling, Dropout.
  - Adam optimizer with learning rate scheduling.
- **Training**
  - Data augmentation (`ImageDataGenerator`).
  - Early stopping, learning rate reduction, model checkpoint.
  - History saving to JSON + training curves.
- **Evaluation**
  - Accuracy, loss, confusion matrix, classification report.
- **Deployment**
  - **Streamlit app** with:
    - Drawing canvas to write digits.
    - Image upload for digit recognition.
    - Real-time prediction + probability chart.

---

## 🚀 Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/handwritten-digit-recognizer.git
cd handwritten-digit-recognizer

2️⃣ Create and activate virtual environment

python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

3️⃣ Install dependencies

pip install -r requirements.txt

📦 requirements.txt
<details> <summary>Click to expand</summary>

tensorflow>=2.8.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
streamlit>=1.10.0
pillow>=8.0.0
streamlit-drawable-canvas>=0.8.0

</details>
📊 Training the Model

Run:

python train.py

This will:

    Download & preprocess MNIST (if not already cached).

    Train the CNN with data augmentation.

    Save the best model in models/best_model.h5.

    Save training plots and metrics in artifacts/.

🧪 Evaluating the Model

Run:

python evaluate.py

This will:

    Load the best saved model.

    Evaluate on the test set.

    Display accuracy, confusion matrix, and classification report.

🌐 Running the Streamlit App

Run:

streamlit run app.py

Features:

    Draw digits in a canvas and get predictions.

    Upload a PNG/JPG image for recognition.

    View prediction probabilities in a bar chart.

📈 Example Results

    Test Accuracy: ~99% (depending on hyperparameters)

    Confusion Matrix: Shows correct classification rates.

    Prediction Example:

Input Digit	Prediction	Confidence
🖌️ 5	5	0.998
📜 License

MIT License – Free to use and modify.
