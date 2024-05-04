import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
# Define the input with the correct shape
input_shape = (512, 512, 3)  # Update this if  model uses a different input shape
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

def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(512, 512))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        out.append(x)
    return np.vstack(out)

# Create a LIME explainer object
explainer = lime_image.LimeImageExplainer()

# Load an image to explain
image_path = 'images/5_640805896.png'
images = transform_img_fn([image_path])

# Explanation with LIME
explanation = explainer.explain_instance(images[0].astype('double'),
                                         functional_model.predict,
                                         top_labels=5,
                                         hide_color=0,
                                         num_samples=1000) # Adjust num_samples for speed vs accuracy trade-off

# Display the image and the mask for the top class
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
plt.imshow(mark_boundaries(temp / 255.0, mask))
plt.show()

