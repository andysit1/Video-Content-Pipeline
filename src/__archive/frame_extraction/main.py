
"""
  The idea of frame extract is to pull information from an image/frame


  #PROCESSING
    #Canny we want to

  #Colors
    #Domaniant Colors


"""
import cv2
import numpy as np
import logging

import glob
import os
import sys

"""
This file is more utils to analyze a frame/image -> shove into the utils


Brain Storm
  1.) We should also create a new region that cuts 70% of the screen


  2.) Take Array and throw it into a histrogram to see if we can see any patterns
  3.) Create some sort of function to find the focus of the screen? or the highlight at a given moment
  4.) Look into parrel looping or way to speed up the process since this will take so long

"""



logger = logging.getLogger(__name__)

#crops a based on an array of list
def crop_viewable_region(img, region=None):
  print("input {} output{}".format(img, region))
  mask = np.zeros_like(img)
  cv2.fillPoly(mask, region, (255, 255, 255))
  masked = cv2.bitwise_and(img, mask)
  print("Masking output", masked)
  return masked


#crops its around the middle offset so we can see action near the crosshair
def crop_image_crosshair(img):
  shape = img.shape
  x_mid = int(shape[0] / 2)
  y_mid = int(shape[1] / 2)
  offset = 75
  return img[x_mid-offset:x_mid+offset , y_mid-offset:y_mid+offset]

def load_frames():
  files = glob.glob("./frame_extraction/in_frame/*")
  return files

def get_filed_info() -> list:
  files = load_frames()
  info = []

  for file in files:
    image = cv2.imread(file)
    processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processedImage =  cv2.Canny(processedImage, threshold1 = 200, threshold2=300)
    processedImage = cv2.GaussianBlur(processedImage,(5,5),0)
    info.append(processedImage)
  return info

def get_cropped():
  logger.debug("CROPPING IMAGES")

  data = get_filed_info()
  region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)
  cropped = []
  logger.debug("Procecesed INFO {}".format(data))
  for arr in data:
    img = crop_viewable_region(arr, region=[region])
    logger.debug("CROPPED IMG {}".format(img))

    cropped.append(img)
  logger.debug("CROPPED IMGS {}".format(cropped))
  return cropped



def remove_all_contents_output_frame():
  input_video_path = './frame_extraction/out_frame/'

  if os.path.exists(path=input_video_path):
    frames = glob.glob(os.path.join(input_video_path, "*png"))

    for frame in frames:
      os.remove(frame)
  else:
    raise TypeError("Input file not found.")

#takes numpy array of images...
def save_out_frames(images, pattern : str):

  current_dir = os.path.dirname(os.path.dirname(__file__))
  c = 0

  remove_all_contents_output_frame()
  for image in images:
    out_file = os.path.join(current_dir, "frame_extraction", "out_frame", "{}{}.png".format(pattern, str(c)))
    if not cv2.imwrite(out_file, image):
      raise Exception("Could not write image")

    c += 1

def process_frames():
  logger.debug("PROCESSING FRAMES")


  files = load_frames()
  current_dir = os.path.dirname(os.path.dirname(__file__))

  print(current_dir)

  c = 0
  for file in files:
    print(file)

    image = cv2.imread(file)
    processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processedImage =  cv2.Canny(processedImage, threshold1 = 200, threshold2=300)
    processedImage = cv2.GaussianBlur(processedImage,(5,5),0)

    print("process", processedImage)
    if not os.path.exists(os.path.join(current_dir, "frame_extraction", "out_frame")):
      print("Starting dir is incorrect, please cd into src")

    out_file = os.path.join(current_dir, "frame_extraction", "out_frame", "demo_gaus{}.png".format(str(c)))

    w = 100
    h = 100
    crop_img = processedImage[x:x+w, y:y+h]

    cv2.imshow("cropped", crop_img)


    print("outfile", out_file)
    if not cv2.imwrite(out_file, processedImage):
      raise Exception("Could not write image")

    c += 1
  # # cv2.imread()



if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  logging.debug("starting debugging")
  print("Testing Frame Detections")
  crop_imgs = get_cropped()
  save_out_frames(crop_imgs)



  # image = No
  # processedImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  # processedImage =  cv2.Canny(processedImage, threshold1 = 100, threshold2=300)
  # cv2.imwrite('1.png', processedImage)