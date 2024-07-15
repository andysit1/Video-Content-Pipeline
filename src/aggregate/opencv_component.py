import cv2
import numpy as np

"""
Handles the OpenCV/Numpy aspect of the code
  -> mostly frame extraction information
"""
import cv2
import numpy as np
import glob
import numpy as np
import os

from utils import percentage_of_white_pixels


class OpenCVAggregate:
    def __init__(self):
        self.crosshair_offset = 75   #distance from center to left, right, top, down
        self.gaus_blue_strength : tuple = (5, 5)

    def is_gray(self, img) -> bool:
        if len(img.shape) != 2:
            return False
        return True

    #given a set of points  example region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)
    def crop_viewable_region(self, img : np.ndarray, region=None) -> np.ndarray:
        if region:
            mask = np.zeros_like(img)
            cv2.fillPoly(mask, region, (255, 255, 255))
            masked = cv2.bitwise_and(img, mask)
            return masked
        return False

    #gets center then offets to get viewable region of the crosshair
    def crop_image_crosshair(self, img : np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        if width > self.crosshair_offset and height > self.crosshair_offset:
            x_mid, y_mid = int(width[0] / 2), int(height[1] / 2)
            offset = self.crosshair_offset
            cropped_img = img[x_mid-offset:x_mid+offset , y_mid-offset:y_mid+offset]
            return cropped_img
        return False

    def change_color_to_BRG2GRAY(self, img : np.ndarray) -> np.ndarray:
        return cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    def do_gaussian_blur(self, img : np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(
            img,
            self.gaus_blue_strength,
            0
        )

    def do_canny_edge_detection(self, img : np.ndarray) -> np.ndarray:
        return cv2.Canny(
            img,
            threshold1 = 200, #TODO/CHANGE make these values init vars
            threshold2=300
        )

    def get_gaussian_gaussian_blur_white_percentage(self, img : np.ndarray) -> float:
        if self.is_gray(img):
           return percentage_of_white_pixels(self.do_gaussian_blur(img))

        gray_img = self.change_color_to_BRG2GRAY(img)
        return percentage_of_white_pixels(self.do_gaussian_blur(gray_img))

    def get_canny_edge_detection_white_percentage(self, img) -> float:
        if self.is_gray(img):
           return percentage_of_white_pixels(self.do_canny_edge_detection(img))

        gray_img = self.change_color_to_BRG2GRAY(img)
        return percentage_of_white_pixels(self.do_canny_edge_detection(gray_img))




