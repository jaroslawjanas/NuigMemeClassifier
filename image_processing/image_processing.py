import cv2
import math
import os

def image_processing(img_path):
    # Read image
    image = cv2.imread(img_path)
    height, width = image.shape[:2]
    dimensions = (width, height)

    # Display image
    cv2.imshow('Image Processing', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate new dimensions (multiple of 32, required by EAST)
    new_width = math.floor(width/32)*32
    new_height = math.floor(height/32)*32
    new_dimensions = (new_width, new_height)

    # Display new image
    image = cv2.resize(image, dimensions)
    cv2.imshow('Image Processing', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
