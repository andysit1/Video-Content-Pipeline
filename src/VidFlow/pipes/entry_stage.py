

from VidFlow.modules.pipeline_builder import Pipe
from VidFlow.pipes.data_cache_stage import DataCachePipe
from VidFlow.aggregate.filehandler_component import FileMaster
from icecream import ic

import logging
logger = logging.getLogger(__name__)


"""
  handles requirements and setting up folder setup
"""
class EntryPipe(Pipe):
  def __init__(self, engine):
    ic("Starting Entry")
    super().__init__(engine)
    self.file_master = FileMaster(self.engine)

  def debug_payload_variables(self):
    ic(self.engine.payload['in_filename'])
    ic(self.engine.payload['video_name'])
    ic(self.engine.payload['cache_txt_out'])
    ic(self.engine.payload['clips_out'])

  def check_payload(self):
    # check if all variables are init
    if not self.engine.payload['in_filename']:
      self.debug_payload_variables()
      raise ValueError("Payload Value Error")

    if not self.engine.payload['cache_txt_out']:
      self.debug_payload_variables()
      raise ValueError("Payload Value Error")

    if not self.engine.payload['clips_out']:
      self.debug_payload_variables()
      raise ValueError("Payload Value Error")

    if not self.engine.payload['video_name']:
      self.debug_payload_variables()
      raise ValueError("Payload Value Error")

    # check if path exist for input filename, cache txt_out, clips_outs
    if not self.file_master.path_exists(self.engine.payload['in_filename']):
      raise FileExistsError("File in_filename does not exists")

    if not self.file_master.path_exists(self.engine.payload['cache_txt_out']):
      raise FileExistsError("File cache_txt_out does not exists")

    if not self.file_master.path_exists(self.engine.payload['clips_out']):
      raise FileExistsError("File clips_out does not exists")


  #checks for requirements and variables incase
  def on_run(self):
    ic.configureOutput(includeContext=True)
    if not self.engine.payload['in_filename']:
      raise FileExistsError("This video does no exist")

    if not self.engine.payload['video_name']:
      raise ValueError("Required Video Name")

    self.file_master.setup() #starts the directories we need base on if we are private or community
    self.check_payload() #if passes we know that all dirs exist and variables have been started correctly
    self.on_done()

  def on_done(self):
    self.engine.machine.next_state = DataCachePipe(self.engine)

  #should do nothing cause its the start. there should never be an error else something went terribly wrong
  def on_error(self):
    pass
