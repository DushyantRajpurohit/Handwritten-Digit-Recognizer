import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

# Load model once
@st.cache_resource
def load_digit_model():
  return load_model("models/best_model.h5")

model=load_digit_model()

st.set_page_config(page_title="Handwritten Digit Recognizer")
st.title("Handwritten Digit Recognizer with Drawing Canvas")
st.write("Draw a digit (0-9) or upload an image to predict.")

# Drawing canvas
canvas_result=st_canvas(
    fill_color="#000000",           # Black background
    stroke_width=10,
    stroke_color="#FFFFFF",         # White drawing color
    background_color="#000000",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

# Prediction from canvas
if canvas_result.image_data is not None:
    img=Image.fromarray((canvas_result.image_data[:, :, 0]).astype(np.uint8))
    img=img.resize((28, 28))
    arr=np.array(img).astype("float32") / 255.0
    arr=arr.reshape(1, 28, 28, 1)

    if np.sum(arr) > 0:  # Only predict if something is drawn
        predictions = model.predict(arr)
        pred_digit = np.argmax(predictions)

        st.subheader(f"Prediction: **{pred_digit}**")
        st.bar_chart(predictions[0])

# File uploader as fallback
uploaded=st.file_uploader("Or upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    img=Image.open(uploaded).convert('L')
    img=ImageOps.invert(img)
    img=img.resize((28, 28))
    arr=np.array(img).astype("float32") / 255.0
    arr=arr.reshape(1, 28, 28, 1)

    predictions=model.predict(arr)
    pred_digit=np.argmax(predictions)

    st.subheader(f"Prediction: **{pred_digit}**")
    st.bar_chart(predictions[0])
