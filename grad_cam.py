
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
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
outputs = Dense(2)(x)  # Adjust this 

# Create the model
functional_model = Model(inputs=inputs, outputs=outputs)

# Load weights into the new model
functional_model.load_weights("cancer_detection_model.h5")
# Assuming the dataset directory and preprocessing are the same as in your original model training script
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "input_transformed/",
    color_mode='rgb',
    image_size=(512, 512),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=2023,
    batch_size=32)  # Adjust the batch size as needed

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # Adjusted to pass model.inputs directly without extra list wrapping
    grad_model = Model(inputs=model.inputs,
                       outputs=[model.get_layer(last_conv_layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Prepare the image (Ensure img_array is correctly preprocessed as required by  model)
for images, _ in train_ds.take(1):
    img_array = images[0:1]  # Taking a single image from the batch
    # Using 'conv2d' as an example for the last conv layer name, replace it with the actual name
    heatmap = make_gradcam_heatmap(img_array, functional_model, 'conv2d_2')  # Adjust the layer name

# Display the heatmap
plt.matshow(heatmap)
plt.show()

# Optionally, superimpose the heatmap over the original image
def display_gradcam(img, heatmap, alpha=0.4):
    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = plt.cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)

    # Display the superimposed image
    plt.imshow(superimposed_img)
    plt.axis('off')
    plt.show()

# Assuming the images in train_ds are normalized in the [0, 1] range
original_img = np.array(tf.squeeze(images[0:1])) * 255
display_gradcam(original_img, heatmap)






