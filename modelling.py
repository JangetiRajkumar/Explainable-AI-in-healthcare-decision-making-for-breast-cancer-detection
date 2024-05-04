import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# Load the dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "input_transformed/",
    color_mode='rgb',
    image_size=(512, 512),
    shuffle=True,
    validation_split=0.2,
    subset="training",
    seed=2023)

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "input_transformed/",
    color_mode='rgb',
    image_size=(512, 512),
    shuffle=True,
    validation_split=0.2,
    subset="validation",
    seed=2023)

# Define the number of classes
num_classes = 2

# Build the model
model = Sequential([
    layers.Rescaling(1./255, input_shape=(512, 512, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Print the model summary
print(model.summary())

# Train the model
model.fit(train_ds, validation_data=valid_ds, epochs=10)

# Save the model
model.save("cancer_detection_model.h5")


