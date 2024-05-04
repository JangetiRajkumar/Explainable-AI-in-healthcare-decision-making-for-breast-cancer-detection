import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import grad_cam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the input with the correct shape
input_shape = (512, 512, 3)  inputs = Input(shape=input_shape)

# Rebuild the model architecture
x = Conv2D(16, 3, padding='same', activation='relu')(inputs)
x = MaxPooling2D()(x)
x = Conv2D(32, 3, padding='same', activation='relu')(x)
x = MaxPooling2D()(x)
x = Dropout(0.2)(x)
x = Conv2D(64, 3, padding='same', activation='relu')(x)
x = MaxPooling2D()(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(2, activation='softmax')(x)  # Assuming softmax for probabilities

# Create the model
functional_model = Model(inputs=inputs, outputs=outputs)

# Load weights into the new model
functional_model.load_weights("cancer_detection_model.h5")

def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name

st.title('Mammogram Prediction App')


uploaded_file = st.file_uploader("Upload a mammogram image in PNG format", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in the correct format

    # Resize the image to match the model's expected input shape
    image = image.resize((512, 512))

    # Convert the image to a numpy array and normalize it if necessary
    image_array = np.array(image) / 255.0  # Normalize to 0-1 if your model expects that

    # Expand dimensions to match the model's input format
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = functional_model.predict(image_array)

    cols = st.columns(2)
    cols[0].image(image)
    cols[0].write("Uploaded image")
    heatmap = grad_cam.make_gradcam_heatmap(image_array, functional_model,find_last_conv_layer(functional_model))
    original_img = np.array(tf.squeeze(image)) * 255
    fig = grad_cam.display_gradcam(original_img, heatmap)
    cols[1].pyplot(fig)
    cols[1].write("Explainability image")
    print(prediction)
    if round(prediction[0][0],0) == 0.0:
        st.success("Positive and Bright areas on right image influence decision")
    else:
        st.error("Negative and Bright areas on right image influence decision")




