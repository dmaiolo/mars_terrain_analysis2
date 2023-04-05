# -------------------------------------------------------------------------------
# utils.py
# -------------------------------------------------------------------------------
#
# Author: David Maiolo
# Date: 03/30/2023
# Last Modified: 04/04/2023
#
# This file contains various utility functions for the Mars Terrain Analysis
# project, including functions for visualizing the training history and
# displaying images and masks.


import cv2
import matplotlib.pyplot as plt
import numpy as np

# Define the color code for different terrain types
color_dict = {
    0: (91, 155, 213),  # Soil
    1: (255, 255, 0),   # Bedrock
    2: (112, 48, 160),  # Sand
    3: (255, 0, 0),     # Big rock
    255: (255, 255, 255)  # No Label
}

def plot_training_history(history):
    plt.figure()

    # Plot training accuracy
    plt.plot(history.history['accuracy'], 'b-', label='train_accuracy')
    
    # Plot validation accuracy
    plt.plot(history.history['val_accuracy'], 'r-', label='val_accuracy')
    
    # Plot training loss
    plt.plot(history.history['loss'], 'b--', label='train_loss')
    
    # Plot validation loss
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')

    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.title("Training and Validation Accuracy & Loss")
    plt.legend(loc='upper right')

    plt.show()

def display_images_and_masks(images, masks):
    num_images = len(images)
    fig, ax = plt.subplots(num_images, 2, figsize=(12, 5 * num_images))

    for i in range(num_images):
        ax[i][0].imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        ax[i][0].axis('off')
        if i == 0:
            ax[i][0].set_title("Image")
        ax[i][1].imshow(masks[i], cmap='gray')
        ax[i][1].axis('off')
        if i == 0:
            ax[i][1].set_title("Mask")

    plt.tight_layout()
    plt.show()

def display_processed_image_and_masks(images, true_masks, predicted_masks, title=None):
    num_samples = len(images)
    fig, axs = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))

    if title:
        fig.suptitle(title, fontsize=16)

    for i in range(num_samples):
        axs[i, 0].imshow(images[i].squeeze(), cmap='gray')
        axs[i, 0].set_title(f"Processed Image {i+1}")
        axs[i, 0].axis('off')

        true_color_coded_mask = create_color_coded_mask(true_masks[i])
        axs[i, 1].imshow(true_color_coded_mask)
        axs[i, 1].set_title(f"True Mask {i+1}")
        axs[i, 1].axis('off')

        # Threshold the predicted masks to create binary masks
        threshold = 0.5
        binary_predicted_masks = (predicted_masks > threshold).astype('uint8') * 255

        predicted_color_coded_mask = create_color_coded_mask(binary_predicted_masks[i])
        axs[i, 2].imshow(predicted_color_coded_mask)
        axs[i, 2].set_title(f"Predicted Mask {i+1}")
        axs[i, 2].axis('off')

        legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Soil', markersize=10, markerfacecolor=tuple(c/255 for c in color_dict[0])),
            plt.Line2D([0], [0], marker='o', color='w', label='Bedrock', markersize=10, markerfacecolor=tuple(c/255 for c in color_dict[1])),
            plt.Line2D([0], [0], marker='o', color='w', label='Sand', markersize=10, markerfacecolor=tuple(c/255 for c in color_dict[2])),
            plt.Line2D([0], [0], marker='o', color='w', label='Big rock', markersize=10, markerfacecolor=tuple(c/255 for c in color_dict[3])),
            plt.Line2D([0], [0], marker='o', color='w', label='No Label', markersize=10, markerfacecolor=tuple(c/255 for c in color_dict[255]))]

    # Add legend to the plot
    axs[-1, -1].legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.show()

def display_color_coded_mask(image, mask, title=None):

    # Create an empty color-coded mask with the same size as the original image
    color_coded_mask = np.zeros((*image.shape[:2], 3), dtype=np.uint8)

    # Replace grayscale values in the mask with the corresponding RGB values
    for value, color in color_dict.items():
        color_coded_mask[mask == value] = color

    # Overlay the color-coded mask on the original image
    image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    color_coded_mask = color_coded_mask.astype(np.uint8)
    overlayed_image = cv2.addWeighted(image, 0.7, color_coded_mask, 0.3, 0)

    # Display the overlaid image
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    ax.set_title(title if title else "Color-coded Mask")
    ax.axis('off')

    # Define legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Soil', markersize=10, markerfacecolor=color_dict[0]),
                    plt.Line2D([0], [0], marker='o', color='w', label='Bedrock', markersize=10, markerfacecolor=color_dict[1]),
                    plt.Line2D([0], [0], marker='o', color='w', label='Sand', markersize=10, markerfacecolor=color_dict[2]),
                    plt.Line2D([0], [0], marker='o', color='w', label='Big rock', markersize=10, markerfacecolor=color_dict[3]),
                    plt.Line2D([0], [0], marker='o', color='w', label='No Label', markersize=10, markerfacecolor=color_dict[255])]

    # Add legend to the plot
    ax.legend(handles=legend_elements, loc='upper right')

    plt.show()

def create_color_coded_mask(mask):

    mask_shape = mask.shape
    if len(mask_shape) > 2 and mask_shape[2] > 1:
        mask = mask[:, :, 0]

    # Squeeze the input mask to remove the extra dimension
    mask = mask.squeeze()

    # Create an empty color-coded mask with the same size as the original mask
    color_coded_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)

    # Replace grayscale values in the mask with the corresponding RGB values
    for value, color in color_dict.items():
        color_coded_mask[mask == value] = color

    return color_coded_mask





