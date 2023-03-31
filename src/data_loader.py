import os
import cv2
import numpy as np
from skimage import io
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

IMG_WIDTH = 260
IMG_HEIGHT = 260

import matplotlib.pyplot as plt

def load_images_and_masks(image_folder, mask_folder, img_height, img_width):
    images = []
    masks = []
    image_files = sorted(os.listdir(image_folder))
    mask_files = sorted(os.listdir(mask_folder))

    # Filter image and mask filenames
    image_files = [f for f in image_files if f[:-3] + "png" in mask_files]
    mask_files = [f for f in mask_files if f[:-3] + "JPG" in image_files]

    for img_file, mask_file in zip(image_files, mask_files):

        img = cv2.imread(os.path.join(image_folder, img_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Missing image for mask {mask_file}")
            continue

        if mask is None:
            print(f"Warning: Missing mask for image {img_file}")
            continue

        img = cv2.resize(img, (img_width, img_height))
        mask = cv2.resize(mask, (img_width, img_height))

        images.append(img)
        masks.append(mask)

        print(f"Processed: {img_file} with mask {mask_file}")

    images = np.array(images, dtype=np.float32) / 255.0
    masks = np.array(masks, dtype=np.float32) / 255.0

    return images, masks

def preprocess_data(images, masks):
    # Convert RGB masks to class labels
    masks = np.array([np.where(mask == 0, 0, 1).astype(np.float32) for mask in masks])

    return images, masks



def get_data_generators(data_dir, images_dir, masks_dir):
    # Create full paths for image and mask directories
    image_folder = os.path.join(data_dir, images_dir)
    mask_folder = os.path.join(data_dir, masks_dir)

    # Load images and masks
    images, masks = load_images_and_masks(image_folder, mask_folder, IMG_HEIGHT, IMG_WIDTH)
    print(f"Loaded {len(images)} images and {len(masks)} masks")

    # Display a side-by-side comparison of the images and masks
    num_images = len(images)
    fig, ax = plt.subplots(num_images, 2, figsize=(10, 20))

    for i in range(num_images):
        ax[i][0].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        #ax[i][0].imshow(images[i])
        ax[i][0].set_title("Image")
        ax[i][1].imshow(masks[i], cmap='gray')
        ax[i][1].set_title("Mask")

    plt.show()

    # Preprocess images and masks
    images, masks = preprocess_data(images, masks)
    print(f"Preprocessed {len(images)} images and {len(masks)} masks")

    # Split data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)
    print(f"x_train shape: {x_train.shape}")
    print(f"x_val shape: {x_val.shape}")
    print(f"y_train size: {len(y_train)}")
    print(f"y_val size: {len(y_val)}")

    # Convert y_train and y_val to NumPy arrays
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    return x_train, x_val, y_train, y_val
