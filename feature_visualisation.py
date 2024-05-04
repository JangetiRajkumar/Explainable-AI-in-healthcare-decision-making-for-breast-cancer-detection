import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
import os

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

# Load weights into the new model
functional_model.load_weights("cancer_detection_model.h5")

# Assuming the dataset directory and preprocessing are the same as in  original model training script
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "input_transformed/",
    color_mode='rgb',
    image_size=(512, 512),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=2023,
    batch_size=32)  # Adjust the batch size as needed

# Take one batch of images for visualization
images, labels = next(iter(train_ds))

def visualize_activations(model, layer_name, input_image, save_dir):
    # Extracts the outputs of the specified layer
    layer_output = model.get_layer(layer_name).output
    # Creates a model that will return these outputs, given the model input
    activation_model = Model(inputs=model.input, outputs=layer_output)
    # Returns a list of Numpy arrays: one array per layer activation
    activations = activation_model.predict(input_image[np.newaxis, ...])

    # For layers with multiple filters, visualize each filter activation
    if len(activations.shape) == 4:
        n_features = activations.shape[-1]  # Number of features in the feature map
        size = activations.shape[1]  # The feature map has shape (1, size, size, n_features)

        n_cols = 8
        n_rows = n_features // n_cols + (n_features % n_cols > 0)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        for i in range(n_rows * n_cols):
            ax = axes.flat[i]
            if i < n_features:
                ax.imshow(activations[0, :, :, i], cmap='viridis')
            ax.axis('off')
        plt.savefig(os.path.join(save_dir, f"{layer_name}.png"))
        plt.close()

save_dir = "activation_visualizations"
os.makedirs(save_dir, exist_ok=True)

# Fetch an image from your dataset
image = images[1]  # Selecting the second image in the batch for visualization

# Iterate through all the layers of the model
for layer in functional_model.layers:
    # Filter layers; adjust the condition based on your requirements
    if 'conv' in layer.name:
        visualize_activations(functional_model, layer.name, image, save_dir)






