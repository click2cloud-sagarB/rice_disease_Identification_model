import cv2
import numpy as np

def is_image_bgr(image):
    # Select a pixel in the middle of the image
    height, width, _ = image.shape
    middle_pixel = image[height // 2, width // 2]

    # Check the values of the middle pixel
    blue, green, red = middle_pixel

    # If blue is the highest value, the image is likely in BGR format
    if blue > red and blue > green:
        return True
    else:
        return False

# Load an image using OpenCV
image = cv2.imread('dataset/Augmented_dataset/Leaf Blast_Augmented/grid_distortion\Leaf_blast  (102)_grid_distortion_transformed.jpg')

# Check if the image is in BGR format
if is_image_bgr(image):
    print("The image is in BGR format.")
else:
    print("The image is in RGB format.")

