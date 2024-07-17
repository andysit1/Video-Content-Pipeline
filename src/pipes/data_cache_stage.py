"""
This Stage handles the init read of the mp4 file and saving a cache representation to txt. Since, video formats are huge
we do not want to hold onto each frame. This causes an huge amount of mem to be used.
"""


from src.modules.pipeline_builder import Pipe
from src.pipes.analyze_data_stage import AnalyzeDataFiles
from src.aggregate.ffmpeg_component import FFMPEGAggregate
from src.aggregate.filehandler_component import FileHandleComponent
import os


"""
  Outputs the data into the text files and saves them
"""
class DataCachePipe(Pipe, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)
    self.ffmpeg = FFMPEGAggregate(engine=engine)

    #init the cache files locations
    self.volume_detect = os.path.join(self.engine.payload['cache_txt_out'], 'volume_detect.txt')
    self.silence_detect = os.path.join(self.engine.payload['cache_txt_out'], 'silence_detect.txt')


  #check if cache exist else, just move onto the next stage (some times we want to re compile videos or try new algos)
  def check_data_exist(self):
    if not self.file_exists(self.volume_detect):
      return False

    if not self.file_exists(self.silence_detect):
      return False
    
    #we should also check if the files had data inside
    return True

  def on_run(self):
    if not self.check_data_exist():
      mean_max_lines = self.ffmpeg.get_mean_max(self.engine.payload['in_filename'])
      silence_lines = self.ffmpeg.silence_detect(self.engine.payload['in_filename'],
                                                silence_threshold=-13,
                                                silence_duration=0.5)

      self.write_lines(self.volume_detect, mean_max_lines)
      self.write_lines(self.silence_detect, silence_lines)

    self.on_done()

  def on_done(self):
    self.engine.machine.next_state = AnalyzeDataFiles(self.engine)


    return super().on_done()

  def on_error(self):
    #gets more complicated later on
    raise TypeError("Missing input file in filename")


