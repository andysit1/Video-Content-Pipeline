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

from .utils import percentage_of_white_pixels


"""

Features to implement
    - killfeed detection
    - better cropping of images
    - silence mode processing -> requires more thought on approach

"""

class OpenCVAggregate:
    def __init__(self):
        self.crosshair_offset = 75   #distance from center to left, right, top, down
        self.kill_feed_offset_region = np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)
        self.gaus_blue_strength : tuple = (5, 5)

    def is_gray(self, img) -> bool:
        if len(img.shape) != 2:
            return False
        return True

    #GAME SPECIFICS (1920 , 1080) -> (480, 270) -> kill feed(3,4), (0, 1)
    def generate_region(self, img : np.ndarray, width_start : int, width_end : int, height_start : int, height_end : int):
        h, w = img.shape[:2]
        quarter_screen_width = (w / 4)
        quarter_screen_height = (h / 4)

        top_left_corner = [width_start * quarter_screen_width, height_start * quarter_screen_height]
        top_right_corner = [width_end * quarter_screen_width, height_start * quarter_screen_height]
        bot_right_corner = [width_end * quarter_screen_width, height_end * quarter_screen_height]
        bot_left_corner = [width_start * quarter_screen_width, height_end * quarter_screen_height]

        #implement a better way of cropping images by region
        # x = int(width_start * quarter_screen_width)
        # x_ = int(width_end * quarter_screen_width)
        # y = int(height_start * quarter_screen_height)
        # y_ = int(height_end * quarter_screen_height)

        return np.array([top_left_corner, top_right_corner, bot_right_corner, bot_left_corner], np.int32)


    #given a set of points  example region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)
    def crop_viewable_region(self, img : np.ndarray, region=None) -> np.ndarray:
        mask = np.zeros_like(img, dtype='uint8')
        cv2.fillPoly(mask, [region], (255, 255, 255))
        masked = cv2.bitwise_and(img, mask)
        return masked


    #gets center then offets to get viewable region of the crosshair
    def crop_image_crosshair(self, img : np.ndarray) -> np.ndarray:
        height, width = img.shape[:2]
        if width > self.crosshair_offset and height > self.crosshair_offset:
            x_mid, y_mid = int(width[0] / 2), int(height[1] / 2)
            offset = self.crosshair_offset
            cropped_img = img[x_mid-offset:x_mid+offset , y_mid-offset:y_mid+offset]
            return cropped_img
        return False

    def convert_to_gray(self, img : np.ndarray) -> np.ndarray:
        return cv2.cvtColor(
            img,
            cv2.COLOR_BGR2GRAY
        )

    def do_binary_threshold(self, img):
        _, thresh = cv2.threshold(img, 170, 255, cv2.THRESH_BINARY)
        return percentage_of_white_pixels(thresh)

    def color_threshold_processing(self, img, lower, higher):
        # processedImage => crosshair focus

        lower_bound = np.array(lower)
        upper_bound = np.array(higher)
        # Threshold the HSV image to get only desired colors
        mask = cv2.inRange(img, lower_bound, upper_bound)
        return mask


    def get_sith_vector(self, img):
        key_points = cv2.SIFT.detectAndCompute(image=img)
        #get the keypoints, then make them a vector
        #similar to boid function from a game, we can probably average the vectors into a single vector.
            #hopefully returning the strongest point
        #then each video will have a vector strength that will identify it's energy/vibe

        pass


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




if __name__ == "__main__":
    path = "E:\Projects/2024\Video-Content-Pipeline\src\__archive/frame_extraction\in_frame\demo3.jpg"
    img = cv2.imread(path)
    cv = OpenCVAggregate()
    region = cv.generate_region(img, 3,4,0,1)
    kill_feed = cv.crop_viewable_region(img, region)
    red_thresh_hold = cv.color_threshold_processing(kill_feed, (84, 36, 36), (250, 0, 0))
    cv2.imwrite("red_threshold_kill_feed.png", red_thresh_hold)
