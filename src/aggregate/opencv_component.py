import cv2
import numpy as np

"""
Handles the OpenCV/Numpy aspect of the code
  -> mostly frame extraction information
"""
import cv2
import numpy as np
import glob
import os

class OpenCVAggregate:

    def crop_viewable_region(self, img, region=None):
        self.logger.debug(f"Cropping viewable region with input: {img}, region: {region}")
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, region, (255, 255, 255))
        masked = cv2.bitwise_and(img, mask)
        self.logger.debug(f"Masking output: {masked}")
        return masked

    def crop_image_crosshair(self, img):
        self.logger.debug("Cropping image around crosshair")
        shape = img.shape
        x_mid = int(shape[0] / 2)
        y_mid = int(shape[1] / 2)
        offset = 75
        cropped_img = img[x_mid-offset:x_mid+offset , y_mid-offset:y_mid+offset]
        self.logger.debug(f"Cropped image: {cropped_img}")
        return cropped_img


    def get_filed_info(self) -> list:
        files = self.load_frames()
        info = []

        for file in files:
            image = cv2.imread(file)
            processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processedImage = cv2.Canny(processedImage, threshold1 = 200, threshold2=300)
            processedImage = cv2.GaussianBlur(processedImage,(5,5),0)
            info.append(processedImage)
            self.logger.debug(f"Processed image: {processedImage}")

        return info

    def get_cropped(self):
        self.logger.debug("Starting to crop images")
        data = self.get_filed_info()
        region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)
        cropped = []
        self.logger.debug(f"Processed data for cropping: {data}")

        for arr in data:
            img = self.crop_viewable_region(arr, region=[region])
            self.logger.debug(f"Cropped image: {img}")
            cropped.append(img)

        self.logger.debug(f"All cropped images: {cropped}")
        return cropped

    def save_out_frames(self, images, pattern: str):
        self.logger.debug("Saving out frames")
        current_dir = os.path.dirname(os.path.dirname(__file__))
        c = 0

        self.remove_all_contents_output_frame()
        for image in images:
            out_file = os.path.join(current_dir, "frame_extraction", "out_frame", f"{pattern}{c}.png")
            if not cv2.imwrite(out_file, image):
                raise Exception("Could not write image")
            self.logger.debug(f"Saved image: {out_file}")
            c += 1

    def process_frames(self):
        self.logger.debug("Processing frames")
        files = self.load_frames()
        current_dir = os.path.dirname(os.path.dirname(__file__))
        c = 0

        for file in files:
            image = cv2.imread(file)
            processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            processedImage = cv2.Canny(processedImage, threshold1 = 200, threshold2=300)
            processedImage = cv2.GaussianBlur(processedImage,(5,5),0)
            out_file = os.path.join(current_dir, "frame_extraction", "out_frame", f"demo_gaus{c}.png")

            cv2.imshow("Processed", processedImage)
            self.logger.debug(f"Processed image: {processedImage}")
            self.logger.debug(f"Output file: {out_file}")

            if not cv2.imwrite(out_file, processedImage):
                raise Exception("Could not write image")
            c += 1
