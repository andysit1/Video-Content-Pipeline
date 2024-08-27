
"""
  this file will handle more advance image processing and storing that data with averages and total points
"""

import os
from icecream import ic
from VidFlow.modules.pipeline_builder import Pipe
from VidFlow.modules.video_stream import VideoStream
from VidFlow.aggregate.opencv_component import OpenCVAggregate, cv2 as cv
from VidFlow.aggregate.filehandler_component import FileHandleComponent
import glob
import cv2
import numpy as np
from .compile_stage import CompileVideoPipe
import logging
logger = logging.getLogger(__name__)

class AnalyzeClipsPipe(Pipe, OpenCVAggregate, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)
    self.crosshair_offset = 75  # Initialize the attribute here

    self.low_piority_weight : int = 1
    self.analyze_data = os.path.join(self.engine.payload['cache_txt_out'], "analyze_data.txt")
    self.chunk_path = os.path.join(self.engine.payload['cache_txt_out'], 'chunks.txt')

    self.score = []

  def get_duration_data(self):
    lines = self.read_lines(self.chunk_path)
    chunks = []

    for line in lines:
      cleaned = line[1:-2].split(', ')
      chunks.append((float(cleaned[0]) - float(cleaned[-1])))
    ic(chunks)
    return chunks

  def get_clips(self) -> list[str]:
    return sorted(glob.glob(os.path.join(self.engine.payload['clips_out'], "*")), key=os.path.getmtime)

  def on_done(self):
    #compile -> make into video
    #ELSE
    #end program
    if self.engine.compile:
      self.engine.machine.next_state = CompileVideoPipe(self.engine)
    else:
      self.engine.machine.next_state = None


  def is_analyze_cache(self):
    if self.file_exists(self.analyze_data):
      return True
    return False

  def cache_analyze_data(self):
    self.write_lines(path=self.analyze_data, lines=self.score)

  def sort_video_order(self):
    di = sorted(self.score, key="points")
    ic(di)


  """
    To implement, I want to process certain parts of the frame, the facts is that it's the same algos but different
    regions of the screen
  """

  def on_run(self):

    if not self.is_analyze_cache():
      ic.enable()
      clips = self.get_clips()

      for clip in clips:
        #per clip vars
        gaus_white_percentage = 0
        canny_white_percentage = 0

        cap = cv.VideoCapture(os.path.abspath(clip))
        frame_count = 0
        while cap.isOpened():
            frame_count += 1
            ret, frame = cap.read()

            if not frame_count % 30:
                continue

            # if frame is read correctly ret is True
            if not ret:
                cap.release()
                break


            if len(frame.shape) == 2:  # Single channel image
                gray_frame = frame  # Already grayscale
            else:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
              crosshair = self.crop_image_crosshair(gray_frame)
              ic(crosshair)
            except Exception as e:
              ic(e)
            else:
              gaus_white_percentage += self.do_binary_threshold(img=crosshair)
              canny_white_percentage += self.get_canny_edge_detection_white_percentage(img=crosshair)
              # ic(canny_white_percentage)
              # focus = self.get_focus_point(gray_frame)

              #add a new data row entry for the stats picked up
              data_obj = {
                'name' : os.path.abspath(clip),
                'gaus_white_percentage' : gaus_white_percentage,
                'canny_white_percentage': canny_white_percentage,
                'points' : gaus_white_percentage * 2 + canny_white_percentage * 0.5,
              }

        self.score.append(data_obj)
      self.cache_analyze_data()

    self.on_done()




"""
  Video stream class is not working, going to switch back to opencv
"""


  # #add focus point as a stat
  # def on_run(self):
  #   if not self.is_analyze_cache():
  #     ic.enable()
  #     ic("Analyzing Clips....")
  #     clips = self.get_clips()

  #     for clip in clips:
  #       gaus_white_percentage = 0
  #       canny_white_percentage = 0

  #       if os.path.exists(clip):
  #         try:
  #           video = VideoStream(path=clip) #load yuv by default
  #           video.open_stream()

  #           frames = 0
  #           ic()
  #           while True:
  #             eof, frame = video.read()
  #             ic("erroring...")

  #             if eof:
  #               ic("closing...")
  #               video.close()
  #               break
  #             frames += 1

  #             if (frames % 30) == 1:
  #               ic(frames)
  #               arr = np.frombuffer(frame, np.uint8).reshape(video.shape()[1] * 3 // 2, video.shape()[0]) #Why does this work

  #               if self.crop_image_crosshair(arr):
  #                 gaus_white_percentage += self.do_binary_threshold(img=arr)
  #                 canny_white_percentage += self.get_canny_edge_detection_white_percentage(img=arr)
  #               # self.get_focus_point(arr)


  #           data_obj = {
  #             'name' : os.path.abspath(clip),
  #             'gaus_white_percentage' : gaus_white_percentage,
  #             'canny_white_percentage': canny_white_percentage,
  #             'points' : gaus_white_percentage * 2 + canny_white_percentage * 0.5,
  #           }

  #           self.score.append(data_obj)
  #         except:
  #           pass
  #     else:
  #       ic("Clip does not exist")
  #     self.cache_analyze_data() #cache the data we got from processing

  #   self.on_done()


