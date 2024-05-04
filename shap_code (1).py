import shap
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense

# Define the input with the correct shape
input_shape = (512, 512, 3)
inputs = Input(shape=input_shape)

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
outputs = Dense(2)(x)

# Create the model
functional_model = Model(inputs=inputs, outputs=outputs)
functional_model.load_weights("cancer_detection_model.h5")

img = tf.keras.preprocessing.image.load_img('images/5_640805896.png', target_size=(512, 512))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Model expects batches

# Set up SHAP
background = img_array  # Using a single image as a simple background dataset
explainer = shap.GradientExplainer(functional_model, background)

# Explain a prediction
shap_values = explainer.shap_values(img_array)

# Plot the SHAP values
shap.image_plot(shap_values, img_array)
