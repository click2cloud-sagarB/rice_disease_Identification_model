import os
import json
from PIL import Image

import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras import backend as K


working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = "D:/rice_disease_detection/models/rice_model_new.h5"
# Load the pre-trained model
model = tf.keras.models.load_model(model_path)

# loading the class names
class_indices = json.load(open("class_indices.json"))


# Function to Load and Preprocess the Image using Pillow
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = Image.open(image_path)
    # Resize the image
    img = img.resize(target_size)
    # Convert the image to a numpy array
    img_array = np.array(img)
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0, 1]
    img_array = img_array.astype('float32') / 255.
    return img_array


# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices, threshold=0.7):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    
    # Get the sorted indices of predictions
    sorted_indices = np.argsort(predictions, axis=1)[0][::-1]
    
    # Get the top prediction and its probability
    predicted_class_index = sorted_indices[0]
    predicted_probability = predictions[0][predicted_class_index]
    
    # If the highest probability is below the threshold, consider it unknown
    if predicted_probability < threshold:
        predicted_class_name = "This Plant is not healthy and is affected by some Unknown Disease."
    else:
        predicted_class_name = class_indices[str(predicted_class_index)]
    
    return predicted_class_name


# Streamlit App
st.title('Rice Disease Detector')

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((150, 150))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the uploaded image and predict the class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f'Prediction: {str(prediction)}')