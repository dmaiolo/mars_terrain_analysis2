# -------------------------------------------------------------------------------
# main.py
# -------------------------------------------------------------------------------
#
# Author: David Maiolo
# Date: 03/30/2023
# Last Modified: 04/04/2023
#
# This is the main script for the Mars Terrain Analysis project. The script
# trains a CNN model to segment images of the Mars terrain into different
# geological features. The trained model can be used to analyze new images
# and provide insights into the geological features present in the images.


import numpy as np
import tensorflow as tf
import cv2
from model import create_model
from data_loader import get_data_generators
from utils import plot_training_history, display_processed_image_and_masks
from config import IMAGE_WIDTH, IMAGE_HEIGHT

def print_welcome_banner():
    print("------------------------------------------------")
    print("Mars Terrain Analysis Project")
    print("------------------------------------------------")

# Print the welcome banner
print_welcome_banner()

# Define model parameters
input_shape = (IMAGE_WIDTH, IMAGE_HEIGHT, 1)

# Create the CNN model
print("Creating the CNN model...")
model = create_model(input_shape)

# Compile the model
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# Load data generators
print("Loading data generators...")
data_dir = "data"
images_dir = "msl/images/edr"
masks_dir = "msl/labels/train"
x_train, x_val, y_train, y_val = get_data_generators(data_dir, images_dir, masks_dir)

# Convert the lists to numpy arrays
x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# Reshape the training data
x_train = x_train.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
x_val = x_val.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

y_train = y_train.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)
y_val = y_val.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)

# Train the model
print("Training the model...")
epochs = 10
batch_size = 32
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val))

# Plot the training history
print("Plotting the training history...")
plot_training_history(history)

# Select two random indices from the validation set
num_samples = 2
selected_indices = np.random.choice(len(x_val), num_samples, replace=False)

# Display the comparisons for the selected indices
sample_images = x_val[selected_indices]
sample_true_masks = y_val[selected_indices]
sample_predictions = model.predict(sample_images)
sample_predicted_masks = sample_predictions.reshape(-1, IMAGE_WIDTH, IMAGE_HEIGHT)
display_processed_image_and_masks(sample_images, sample_true_masks, sample_predicted_masks, title="Random Samples Comparison")

# Prompt the user for the H5 file location and filename
h5_file = input("Enter the filename and location to save the H5 file (e.g., 'models/mars_terrain_model.h5') or press Enter for the default: ")
if not h5_file.strip():
    h5_file = "mars_terrain_model.h5"

# Save the trained model
print("Saving the trained model...")
model.save(h5_file)

print("Model trained and saved successfully.")