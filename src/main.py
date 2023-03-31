# -------------------------------------------------------------------------------
# main.py
# -------------------------------------------------------------------------------
#
# Author: David Maiolo
# Date: 03/30/2023
#
# This file serves as the main entry point for the Mars Terrain Analysis project.
# It orchestrates the entire pipeline, from loading data, training the model,
# and evaluating its performance, to visualizing results.

import numpy as np
import tensorflow as tf
from model import create_model
from data_loader import get_data_generators

# Define model parameters
input_shape = (260, 260, 1)

# Create the CNN model
model = create_model(input_shape)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Load data generators
data_dir = "data"
images_dir = "msl/images/edr"
masks_dir = "msl/labels/train"
x_train, x_val, y_train, y_val = get_data_generators(data_dir, images_dir, masks_dir)

# Convert the lists to numpy arrays
x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)
#x_val = tf.stack(x_val)
#y_val = tf.stack(y_val)

# Reshape the training data
x_train = x_train.reshape(-1, 260, 260, 1)
x_val = x_val.reshape(-1, 260, 260, 1)

y_train = y_train.reshape(-1, 260, 260, 1)
y_val = y_val.reshape(-1, 260, 260, 1)

# Train the model
epochs = 10
batch_size = 32
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Save the trained model
model.save("mars_terrain_model.h5")

print("Model trained and saved successfully.")
