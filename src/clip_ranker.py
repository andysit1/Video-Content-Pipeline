import cv2
from base.clip_object import Clip
from frame_extraction.main import crop_viewable_region, save_out_frames
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

  def load_frame(self, frame_path : str):
    if os.path.exists(path=frame_path):
      print("Successfully Loaded {}".format(frame_path))
      img = cv2.imread(frame_path)
      self.frames.append(img)
    else:
      print("Frame not found at {}".format(frame_path))

  #will split the video into frames every .5 second assuming a 60 fps video
  def load_clip(self, clip : Clip):
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
    region =  np.array([[100, 100],[1600, 100],[1600,800],[100, 800]], np.int32)

    for frame in self.frames:
      #crop, gray scale, blur
      print('frame', frame)


      processedImage = crop_viewable_region(frame, [region])
      processedImage = cv2.cvtColor(processedImage, cv2.COLOR_BGR2GRAY)
      processedImage = cv2.GaussianBlur(processedImage,(5,5),0)

      #after applying and removing excess screen we can begin processing the frames
      self.processed_frames.append(processedImage)


  def color_dom_processing(self):
    for frame in self.frames:
      # Resize frame to speed up processing
      small_frame = cv2.resize(frame, (0, 0), fx=0.1, fy=0.1)
      # Reshape the image to be a list of pixels
      pixels = small_frame.reshape((-1, 3))
      # Cluster the pixel intensities
      kmeans = KMeans(n_clusters=5)
      kmeans.fit(pixels)
      # Get the number of pixels in each cluster
      counts = Counter(kmeans.labels_)
      # Sort to find the most popular colors
      center_colors = kmeans.cluster_centers_
      ordered_colors = [center_colors[i] for i in counts.keys()]
      self.processed_frames_color_dom.append(ordered_colors)

  def color_threshold_processing(self):
    for frame in self.frames:
      hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
      # Define range of a color in HSV
      lower_bound = np.array([30, 150, 50])
      upper_bound = np.array([255, 255, 180])
      # Threshold the HSV image to get only desired colors
      mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)
      self.processed_frames_color_threshold.append(mask)

  def color_processing(self):
    self.color_dom_processing()
    self.color_threshold_processing()

  def edge(self):
    for frame in self.processed_frames:
      edges = cv2.Canny(frame, 100, 200)
      self.processed_frames_edge.append(edges)

  def threshold(self):
    for frame in self.processed_frames:
        _, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)
        self.processed_frames_threshold.append(thresh)


  def run(self):
    self.simplify_frames()
    self.color_processing()
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

  frame_path = "./frame_extraction/in_frame/demo2.png"
  clip_ranker.load_frame(frame_path=frame_path)
  clip_ranker.run()
  save_out_frames(clip_ranker.processed_frames, "process_frames")
  # save_out_frames(clip_ranker.processed_frames_color_dom, "process_frames_color_dom")
  save_out_frames(clip_ranker.processed_frames_color_threshold, "process_frames_color_thresh")


