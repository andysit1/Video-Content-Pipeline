


"""
BRAIN STORM

This file has methods that can take ina numpy array which represent an image and count the number of cells which are white or black
return precentages that can later be used to give points which rank frames within a clip

"""

import numpy as np



def percentage_of_white_pixels(image):
    """
    Calculate the percentage of white pixels in a given image.

    Parameters:
    image (numpy.ndarray): Input image as a NumPy array. The image should be in grayscale or binary format.

    Returns:
    float: Percentage of white pixels in the image.
    """

    if len(image.shape) != 2:
        raise ValueError("Input image should be a 2D array representing a grayscale or binary image.")

    # Count the number of white pixels (assuming white is represented by the maximum value, i.e., 255)
    white_pixel_count = np.sum(image == 255)
    total_pixel_count = image.size

    # Calculate the percentage of white pixels
    percentage = (white_pixel_count / total_pixel_count) * 100

    return percentage