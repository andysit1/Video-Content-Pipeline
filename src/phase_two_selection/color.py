import numpy as np


img = None
#this file should implement domaniant colors and average colors

# find average per row, assuming image is already in the RGB format.
# np.average() takes in an axis argument which finds the average across that axis.
average_color_per_row = np.average(img, axis=0)
# find average across average per row
average_color = np.average(average_color_per_row, axis=0)
# convert back to uint8
average_color = np.uint8(average_color)
print(average_color)