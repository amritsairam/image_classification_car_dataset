import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pandas as pd 
from keras.applications.efficientnet import preprocess_input, decode_predictions

# Load the trained model
model = tf.keras.models.load_model('/Users/sairam/Desktop/desktop/computer vision projects/shreyas_car_images/cnn_model_runs/model_weights_efficient_best.keras')  # Replace with your model path

df=pd.read_csv('car_models.csv',header=None)
df.columns = ['class_name']
print(df)

# Define the Streamlit App
st.title("Car Model Classification")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to match EfficientNet input size
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    # Inference
    st.write("Classifying...")
    predictions = model.predict(img_array)
    print(len(predictions[0]))
    predicted_class = np.argmax(predictions, axis=1)
    print(predicted_class)

    predicted_class_name = df['class_name'].iloc[predicted_class]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class_name}")
