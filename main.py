import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import cv2

from subprocess import check_output

# List files in the directory
print(check_output(["ls", "./data-science-bowl-2018"]).decode("utf8"))

# Suppress warnings
warnings.filterwarnings('ignore')

# Stephen Bailey solution
# https://www.kaggle.com/stkbailey/teaching-notebook-for-total-imaging-newbies

# Define the path to the images directory
images_dir = pathlib.Path('./data-science-bowl-2018/stage1_train')

# Get paths to all image files
training_paths = images_dir.glob('*/images/*.png')
training_sorted = sorted([x for x in training_paths])

# Select an image path
im_path = training_sorted[45]

# Read the image
bgrimg = cv2.imread(str(im_path))

# Create a figure and subplots
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Plot the RGB image
axs[0].imshow(cv2.cvtColor(bgrimg, cv2.COLOR_BGR2RGB))
axs[0].set_title('RGB Image')
axs[0].axis('off')

# Convert the image to grayscale
grayimg = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)

# Plot the grayscale image
axs[1].imshow(grayimg, cmap='gray')
axs[1].set_title('Grayscale Image')
axs[1].axis('off')

# Show the plot
plt.show()

# Print shapes of the original and grayscale images
print('Original Image Shape:', bgrimg.shape)
print('New Image Shape:', grayimg.shape)

# reduced a dimension when transformed from BGR to grayscale.
# Original Image Shape: (520, 696, 3)
# New Image Shape: (520, 696)
# grayscale is a range of  shades from black to white. 
# contains only shades of gray and no color, not RGB

# Plot the distribution of intensity values
plt.subplot(1, 2, 1)
plt.hist(grayimg.flatten(), bins=50, color='blue', alpha=0.7)
plt.title('Distribution of intensity values')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

# to check if the peak in prev histogram we see is actually present, zoom in
plt.subplot(1, 2, 2)
plt.hist(grayimg.flatten(), bins=50, color='orange', alpha=0.7)
plt.ylim(0, 30000)
plt.title('Distribution of intensity values (Zoomed In)')
plt.xlabel('Intensity')
plt.ylabel('Frequency')

plt.show()

