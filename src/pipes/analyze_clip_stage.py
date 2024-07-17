
"""
  this file will handle more advance image processing
"""

import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.modules.video_stream import VideoStream
from src.aggregate.opencv_component import OpenCVAggregate
import glob


class AnalyzeClips(Pipe, OpenCVAggregate):
  def __init__(self, engine):
    super().__init__(engine)

  def get_clips(self) -> list[str]:
    return sorted(glob.glob(os.path.join(self.engine.payload['clips_out'], "*")), key=os.path.getmtime)

  def on_done(self):
    self.engine.machine.current = None

  def on_run(self):
    ic(self.get_clips())
    self.on_done()















