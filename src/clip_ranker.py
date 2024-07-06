import cv2
from base.clip_object import Clip

from phase_two_selection.points import percentage_of_white_pixels
from frame_extraction.main import crop_viewable_region, crop_image_crosshair, save_out_frames
from frame_extraction.utils import average_method, luminosity_method, combine_output_frames_into_video
import numpy as np
#taks a clip objet and starts processing the rank
from sklearn.cluster import KMeans
from collections import Counter
import os




class Ranker:
  def __init__(self):
    self.frames = []
    self.processed_frames = []
    self.processed_frames_edge = []
    self.processed_frames_threshold = []

    self.processed_frames_color = []

    self.processed_frames_color_threshold = []
    self.processed_frames_color_dom = []

    #constant
    self.constant_fps = None
    self.constant_frames = None


    self.action_weight : int = 1
    self.action_points : int = 0

    self.data = None

  #this should be in the clip object
  def load_frame(self, frame_path : str):
    if os.path.exists(path=frame_path):
      print("Successfully Loaded {}".format(frame_path))
      img = cv2.imread(frame_path)
      self.frames.append(img)
    else:
      print("Frame not found at {}".format(frame_path))

  def clear(self) -> None:
    self.frames = []
    self.processed_frames = []
    self.processed_frames_edge = []
    self.processed_frames_threshold = []

    self.processed_frames_color = []

    self.processed_frames_color_threshold = []
    self.processed_frames_color_dom = []

    self.action_weight : int = 1
    self.action_points : int = 0

  #will split the video into frames every .5 second assuming a 60 fps video
  def load_clip(self, clip : Clip):
    self.clear()

    cap = cv2.VideoCapture(clip.get_path())
    self.constant_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    self.constant_fps = cap.get(cv2.CAP_PROP_FPS)

    # seconds = round(self.constant_frames / self.constant_fps)

    c = 0
    while cap.isOpened():
      ret, frame = cap.read()
      if ret:
        if c % (self.constant_fps * 2):
          self.frames.append(frame)
      else:
        print("Video ended break...")
        cap.release()
      c += 1




  def simplify_frames(self):
    # region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)

    for frame in self.frames:
      # processedImage = crop_viewable_region(img=frame, region=[region])
      processedImage = crop_image_crosshair(img=frame)
      processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)

      #Note, its better to have a lower blur inorder to see more details, at 11,11 we loss orb details and it starts not noticing players
      processedImage = cv2.GaussianBlur(processedImage, (5, 5),0)

      #after applying and removing excess screen we can begin processing the frames
      self.processed_frames.append(processedImage)


  """

  Thoughts, the most eye popping/action pack scene will likely have a large
  contracts.

  """

  #
  # def color_dom_processing(self):
  #   for frame in self.frames:
  #     # Resize frame to speed up processing
  #     small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
  #     # Reshape the image to be a list of pixels
  #     pixels = small_frame.reshape((-1, 3))
  #     # Cluster the pixel intensities
  #     kmeans = KMeans(n_clusters=5)
  #     kmeans.fit(pixels)
  #     # Get the number of pixels in each cluster
  #     counts = Counter(kmeans.labels_)
  #     # Sort to find the most popular colors
  #     center_colors = kmeans.cluster_centers_
  #     ordered_colors = [center_colors[i] for i in counts.keys()]
  #     self.processed_frames_color_dom.append(ordered_colors)

  #     return ordered_colors[0], ordered_colors[-1]







  #if we calculate the dom colors within the range of the screen, we can determine when we see players or is looking at interesting objects
  def color_dom_processing(self, frame):
    small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
    # Reshape the image to be a list of pixels
    pixels = small_frame.reshape((-1, 3))
    # Cluster the pixel intensities
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    # Get the number of pixels in each cluster
    counts = Counter(kmeans.labels_)
    # Sort to find the most popular colors
    center_colors = kmeans.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    self.processed_frames_color_dom.append(ordered_colors)

    return ordered_colors


  #the domaniant colors will almost 90% be the colors of the envoirnment, hence I believe its resonable to take the first and last as the lower bounds
  #this way we can exclude a good about of the terrain of the start
  def color_threshold_processing(self):

    for frame in self.frames:
      process_frame = cv2.GaussianBlur(frame, (5,5), 0)
      dom_colors = self.color_dom_processing(frame=process_frame)

      hsv_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2HSV)
      # Define range of a color in HSV
      lower_bound = np.array(dom_colors[0])
      upper_bound = np.array(dom_colors[-1])
      # Threshold the HSV image to get only desired colors
      mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
      inverse_mask = cv2.bitwise_not(mask)

      self.processed_frames_color_threshold.append(inverse_mask)


  #wanted to see if this was possible to try and threshold the domaniant colors ina  gray image, the output wasn't soo good but it's okay

  def color_blacknwhite_threshold_processing(self):
    for frame in self.processed_frames:
      dom_colors = self.color_dom_processing(frame=frame)

      _, thresh = cv2.threshold(frame, luminosity_method(dom_colors[0]), luminosity_method(dom_colors[1]), cv2.THRESH_BINARY)
      self.processed_frames_threshold.append(thresh)

  def color_processing(self):
    self.color_threshold_processing()

  def edge(self):
    for frame in self.processed_frames:
      edges = cv2.Canny(frame, 100, 200)
      self.processed_frames_edge.append(edges)


  #after experimentating 200 is a pretty good threshold since it cuts out a lot noise and effects
  # @ 170 we still see effects such as brim smokes, likely need to tune per game but we will focus on Valorant for now
  def threshold(self):
    for frame in self.processed_frames:
        _, thresh = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
        self.processed_frames_threshold.append(thresh)

        #in the future we can change the action weight var to change the type of video we produce...
        self.action_points += percentage_of_white_pixels(thresh) * self.action_weight

  def run(self):
    from itertools import zip_longest

    self.simplify_frames()
    self.threshold()
    self.color_processing()
    
    self.data = zip_longest(
      self.processed_frames,
      self.processed_frames_color,
      self.processed_frames_color_dom,
      self.processed_frames_threshold,
      self.processed_frames_color_threshold,
      fillvalue="?"
    )






  def get_points(self):
    return self.action_points


    # self.color_blacknwhite_threshold_processing()
    # Add more processing steps as needed

if __name__ == "__main__":
  print("Testing Rank Class")

  #Testing Clipping
  clip_ranker = Ranker()
  # clip1 = Clip("../output-video/video49.mp4")
  # clip_ranker.load_clip(clip=clip1)
  # clip_ranker.run()

  # print(clip_ranker.processed_frames)
  # save_out_frames(clip_ranker.processed_frames, "process_frames")


  #Testing Frame
  # frame_path = "./frame_extraction/in_frame/demo2.png"
  # frame_path1 ="./frame_extraction/in_frame/demo3.jpg"
  # clip_ranker.load_frame(frame_path=frame_path)
  # clip_ranker.load_frame(frame_path=frame_path1)


  #video 78 = 3386.9422222222233
  #video 25 = 2291.168888888889
  #video 54 = 114.755

  video_file = "../output-video/video015.mp4"
  clip1 = Clip(path=video_file)
  clip_ranker.load_clip(clip1)
  clip_ranker.run()

  save_out_frames(clip_ranker.processed_frames_threshold, "process_frames_threshold")
  combine_output_frames_into_video()

