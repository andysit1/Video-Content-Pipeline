import cv2
import numpy as np
from math import sqrt


class VideoLoader:
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_number = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame_image(self, frame_index):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()
        if ret:
            return frame
        else:
            return None

    #releases the resourses
    def close(self):
        self.cap.release()
