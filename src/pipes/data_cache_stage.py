"""
This Stage handles the init read of the mp4 file and saving a cache representation to txt. Since, video formats are huge
we do not want to hold onto each frame. This causes an huge amount of mem to be used.
"""


from src.modules.pipeline_builder import Pipe
from src.pipes.analyze_data_stage import AnalyzeDataFiles
from src.aggregate.ffmpeg_component import FFMPEGAggregate, ic
from src.aggregate.filehandler_component import FileHandleComponent
import os
import re
#global regex
histogram_re = re.compile(r'histogram_(?P<db_level>\d+)db: (?P<count>\d+)')
import logging
logger = logging.getLogger(__name__)
"""
  Outputs the data into the text files and saves them
"""
class DataCachePipe(Pipe, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)
    self.ffmpeg = FFMPEGAggregate(engine=engine)
    #handle db
    self.silence_db_range : int = 10500

    #init the cache files locations
    self.volume_detect = os.path.join(self.engine.payload['cache_txt_out'], 'volume_detect.txt')
    self.silence_detect = os.path.join(self.engine.payload['cache_txt_out'], 'silence_detect.txt')


  #check if cache exist else, just move onto the next stage (some times we want to re compile videos or try new algos)
  def check_data_exist(self):
    if not self.file_exists(self.volume_detect):
      mean_max_lines = self.ffmpeg.get_mean_max(self.engine.payload['in_filename'])
      self.write_lines(self.volume_detect, mean_max_lines)

    if not self.file_exists(self.silence_detect):
      silence_db = self.get_silence_db()

      if silence_db == -1:
          self.on_error()
          return False

      ic("selected db", silence_db)
      silence_lines = self.ffmpeg.silence_detect(self.engine.payload['in_filename'],
                                                silence_threshold=silence_db,
                                                silence_duration=0.5)
      self.write_lines(self.silence_detect, silence_lines)

    return True

  #TODO/FEATURE
  def get_silence_db(self) -> int:
    lines = self.read_lines(path=self.volume_detect)
    histogram_matches = []
    for line in lines:
      histogram_matches = histogram_re.findall(line)

      if histogram_matches and int(histogram_matches[0][1]) > self.silence_db_range:
        return int(histogram_matches[0][0]) * -1
    return -1

  def on_run(self):
    if self.check_data_exist():
      self.on_done()
    else:
      return

  def on_done(self):
    self.engine.machine.next_state = AnalyzeDataFiles(self.engine)

  def on_error(self):
    #send to Download if there's any error
    from src.pipes.dl_stage import DownloadPipe
    self.engine.machine.current = DownloadPipe(self.engine)


