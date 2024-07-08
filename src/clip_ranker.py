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

from icecream import ic
from numba import jit, cuda

from base.video_stream import VideoStream


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
    self.total_action_points : int = 0
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


  #will split the video into frames every .5 second assuming a 60 fps video
  #largest contributor to time issues
  #change to numba function -> parr + gpu function for speed?


  def load_clip_opti(self, clip : Clip):
    self.clear()

    cv_color = cv2.COLOR_YUV2RGB_I420       # OpenCV converts YUV frames to RGB

    if os.path.exists(clip.get_path()):

      try:
        video = VideoStream(path=clip.get_path())
        video.open_stream()
        frames = 0
        ic(clip.get_path())
        while True:
            eof, frame = video.read()
            if eof: break
            frames += 1
            #only proccess every 30 frames to reduce work
            # ic(frame)
            if frames % 30:
              self.frames.append(frame)

              arr = np.frombuffer(frame, np.uint8).reshape(video.shape()[1] * 3 // 2, video.shape()[0])
              processedImage = cv2.GaussianBlur(arr, (5, 5),0)
              processedImage = crop_image_crosshair(img=processedImage)
              self.by_frame_threshold(processedImage)
              # color_image = cv2.cvtColor(processedImage, cv_color)
        self.run()
      except:
        pass


  def load_clip(self, clip : Clip):
    cv_color = cv2.COLOR_YUV2RGB_I420       # OpenCV converts YUV frames to RGB

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
          processedImage = cv2.GaussianBlur(frame, (5, 5),0)
          processedImage = crop_image_crosshair(img=processedImage)
          self.processed_frames_color.append(processedImage)

          #to gray
          uncolor_image = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)
          self.processed_frames.append(uncolor_image)

      else:
        print("Video ended break...")
        cap.release()
      c += 1




  #Note, its better to have a lower blur inorder to see more details, at 11,11 we loss orb details and it starts not noticing players
  def simplify_frames(self):
    # region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)

    for frame in self.frames:
      processedImage = cv2.GaussianBlur(frame, (5, 5),0)
      processedImage = crop_image_crosshair(img=frame)
      self.processed_frames_color.append(processedImage)
      processedImageNoColor = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)
      self.processed_frames.append(processedImageNoColor)

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
    pixels = frame.reshape((-1, 3))
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(pixels)
    counts = Counter(kmeans.labels_)
    center_colors = kmeans.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    self.processed_frames_color_dom.append(ordered_colors)

    return ordered_colors

  #wanted to see if this was possible to try and threshold the domaniant colors ina  gray image, the output wasn't soo good but it's okay
  def color_blacknwhite_threshold_processing(self):
    for frame in self.processed_frames:
      dom_colors = self.color_dom_processing(frame=frame)

      _, thresh = cv2.threshold(frame, luminosity_method(dom_colors[0]), luminosity_method(dom_colors[1]), cv2.THRESH_BINARY)
      self.processed_frames_threshold.append(thresh)


  #the domaniant colors will almost 90% be the colors of the envoirnment, hence I believe its resonable to take the first and last as the lower bounds
  #this way we can exclude a good about of the terrain of the start
  def color_threshold_processing(self):
    # processedImage => crosshair focus
    for frame in self.processed_frames_color:
      dom_colors = self.color_dom_processing(frame=frame)

      # Define range of a color in HSV
      lower_bound = np.array(dom_colors[0])
      upper_bound = np.array(dom_colors[-1])
      # Threshold the HSV image to get only desired colors
      mask = cv2.inRange(frame, lower_bound, upper_bound)
      # inverse_mask = cv2.bitwise_not(mask)

      p = percentage_of_white_pixels(mask) * self.action_weight

      self.action_points += p
      self.processed_frames_color_threshold.append(mask)

  def color_processing(self):
    self.color_threshold_processing()

  def edge(self):
    for frame in self.processed_frames:
      edges = cv2.Canny(frame, 100, 200)
      self.processed_frames_edge.append(edges)



  def by_frame_threshold(self, frame):
    _, thresh = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
    self.action_points = percentage_of_white_pixels(thresh) * self.action_weight
    self.total_action_points += self.action_points

  #after experimentating 200 is a pretty good threshold since it cuts out a lot noise and effects
  # @ 170 we still see effects such as brim smokes, likely need to tune per game but we will focus on Valorant for now
  def threshold(self):
    for frame in self.processed_frames:
        _, thresh = cv2.threshold(frame, 170, 255, cv2.THRESH_BINARY)
        self.action_points = percentage_of_white_pixels(thresh) * self.action_weight
        self.total_action_points += self.action_points

  def run(self):
    from itertools import zip_longest
    self.threshold()



    # self.color_processing()

    # self.data = zip_longest(
    #   self.processed_frames,
    #   self.processed_frames_color,
    #   self.processed_frames_color_dom,
    #   self.processed_frames_threshold,
    #   self.processed_frames_color_threshold,
    #   fillvalue="?"
    # )

  def get_total_points(self):
    return self.total_action_points

  def get_points(self):
    return self.action_points



import cProfile
import pstats
import glob

def test_opti():
    with cProfile.Profile() as profile:
      clip_file = "../output-video/"

      if not os.path.exists(clip_file):
          raise TypeError("Path not found")

      clips_filename = sorted(glob.glob(os.path.join(clip_file, "*mp4")), key=os.path.getmtime)
      clips_points = []
      ranker = Ranker()

      ic(clips_filename[0])


      for clip in clips_filename:
        clip_obj = Clip(path=clip)
        ranker.load_clip_opti(clip_obj)
        clips_points.append((clip, ranker.get_total_points()))

      ic(clips_points)
      avg = ranker.get_total_points() / len(clips_filename)
      ic(avg)
      selected_clips = list(filter(lambda x: (x[1] > avg), clips_points))
      ic(selected_clips)

    save_out_frames(images=ranker.processed_frames, pattern="opti")
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(20)

    print(len(ranker.processed_frames))

def test():
    with cProfile.Profile() as profile:
      silence_threshold=-10,
      silence_duration=3,
      clip_file = "../output-video/"
      out_pattern = "../output-video/video{}.mp4"

      if not os.path.exists(clip_file):
          raise TypeError("Path not found")


      clips_filename = sorted(glob.glob(os.path.join(clip_file, "*mp4")), key=os.path.getmtime)
      clips_points = []
      ranker = Ranker()

      ic(clips_filename[0])


      clip_obj = Clip(path=clips_filename[0])
      ranker.load_clip(clip_obj)
      ranker.run()
      clips_points.append((clips_filename[0], ranker.get_points()))

    save_out_frames(images=ranker.processed_frames, pattern="opti")
    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.print_stats(20)

    print(len(ranker.processed_frames))



if __name__ == "__main__":

  test_opti()