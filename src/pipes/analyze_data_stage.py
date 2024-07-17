"""
After the processing stage we take our data that we deemed important in processing and start graphing the information.
Ideally, here is where I will make big decisions of the video ie stuff like thresholding db, location of region to cut,
where the facecam, and other stuff yet to implement

  Things I want for sure is
    -> average db
    -> the histogram in volume detect
"""

import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.aggregate.filehandler_component import FileHandleComponent

class AnalyzeDataFiles(Pipe, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)

  def on_done(self):
    print("Done pipeline for now")
    self.engine.machine.current = None

  def get_data(self):
    volume_detect = os.path.join(self.engine.payload['cache_txt_out'], 'volume_detect.txt')
    silence_detect = os.path.join(self.engine.payload['cache_txt_out'], 'silence_detect.txt')

    volume_data = self.read_lines(volume_detect)
    silence_data = self.read_lines(silence_detect)

    ic(volume_data, len(volume_data))
    ic(silence_data, len(silence_detect))

    return volume_data, silence_detect

  def on_run(self):
    print("Running AnalyzeDataFiles")
    self.get_data()
    self.on_done()


