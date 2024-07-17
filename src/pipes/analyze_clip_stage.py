
"""
  this file will handle more advance image processing and storing that data with averages and total points
"""

import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.modules.video_stream import VideoStream
from src.aggregate.opencv_component import OpenCVAggregate
from src.aggregate.filehandler_component import FileHandleComponent
import glob
import cv2
import numpy as np
from .compile_stage import CompileVideoPipe


class AnalyzeClipsPipe(Pipe, OpenCVAggregate, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)

    self.low_piority_weight : int = 1
    self.analyze_data = os.path.join(self.engine.payload['cache_txt_out'], "analyze_data.txt")

    self.score = []

  def get_clips(self) -> list[str]:
    return sorted(glob.glob(os.path.join(self.engine.payload['clips_out'], "*")), key=os.path.getmtime)

  def on_done(self):
    self.engine.machine.current = CompileVideoPipe(self.engine)

  def is_analyze_cache(self):
    if self.file_exists(self.analyze_data):
      return True
    return False

  def cache_analyze_data(self):
    self.write_lines(path=self.analyze_data, lines=self.score)

  def sort_video_order(self):
    di = sorted(self.score, key="points")
    ic(di)

  def on_run(self):
    if not self.is_analyze_cache():
      ic.disable()
      ic("Analyzing Clips....")
      clips = self.get_clips()
      cv_color = cv2.COLOR_YUV2RGB_I420       # OpenCV converts YUV frames to RGB
      gaus_white_percentage = 0
      canny_white_percentage = 0
      for clip in clips:
        if os.path.exists(clip):
          try:
            video = VideoStream(path=clip) #load yuv by default
            video.open_stream()

            frames = 0
            ic()
            while True:
              eof, frame = video.read()
              if eof:
                ic("closing...")
                video.close()
                break
              frames += 1

              if (frames % 30) == 1:
                ic(frames)
                arr = np.frombuffer(frame, np.uint8).reshape(video.shape()[1] * 3 // 2, video.shape()[0]) #Why does this work
                gaus_white_percentage += self.do_binary_threshold(img=arr)
                canny_white_percentage += self.get_canny_edge_detection_white_percentage(img=arr)


            data_obj = {
              'name' : clip,
              'gaus_white_percentage' : gaus_white_percentage,
              'canny_white_percentage': canny_white_percentage,
              'points' : gaus_white_percentage * 2 + canny_white_percentage * 0.5
            }

            self.score.append(data_obj)
          except:
            pass
      else:
        ic("Clip does not exist")
      self.cache_analyze_data() #cache the data we got from processing

    self.on_done()


