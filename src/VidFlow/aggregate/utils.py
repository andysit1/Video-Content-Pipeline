import numpy as np


def luminosity_method(x_list):
  """
    I forgot the exact reason I created this method.
    Pretty sure it just averages the frame to get the gray scale in a more advanced way that opencv offers. Not quite sure.
  """
  if len(x_list) != 3:
    raise ValueError("List type is wrong, not enough values (3) found {}".format(len(x_list)))

  return 0.3 * x_list[0] + 0.59 * x_list[1] + 0.11 * x_list[2]


def percentage_of_white_pixels(image: np.ndarray):
    """
    Calculate the percentage of white pixels in a given image.

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
