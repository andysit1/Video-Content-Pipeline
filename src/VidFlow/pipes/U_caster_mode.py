

"""
This Stage handles the actual splicing of the video and is the most time consuming part of the program

  place the video splitting occurs
  ideally if we ever find a better way of splicing videos it will go here. Probably a lot of min maxing memory and making sure
  we aren't holding resources we dont need

  cuda/gpu rendering? idk we will figure that out later
"""
import os
from icecream import ic
from src.VidFlow.modules.pipeline_builder import Pipe
from src.VidFlow.aggregate.ffmpeg_component import FFMPEGAggregate
from src.VidFlow.modules.video_stream import VideoStream

import logging
logger = logging.getLogger(__name__)

"""
  This is pretty much the Analyze Clips Pipe but for the the entire video

"""


class U_CasterPipe(Pipe, FFMPEGAggregate):
    def __init__(self, engine):
      super().__init__(engine)

      if not self.path_exists(self.engine.payload['in_filename']):
        logger.error("FILE ERROR: in_filename does not exist in payload. Could be due to ActionPipe or __main__.py")
        self.on_error()

      self.video = VideoStream(path=self.engine.payload['in_filename']) #load yuv by default

      #ex how to config for reduced data.
      self.video.config(output_resolution=(1280, 720))
      self.video.config(crop_rect=(0,0,720,480))

    def is_not_time_stampable(seof) -> bool:
      """

      """
      #if screen block

      #if screen did not change compared to previous

        #binary threshold?
      pass

    def is_time_stampable(self) -> bool:
      """

      """
      pass

    def on_run(self):
      self.video.open_stream()