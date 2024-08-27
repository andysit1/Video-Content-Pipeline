from src.VidFlow.modules.pipeline_builder import Pipe
from icecream import ic

class FillerPipe(Pipe):
  def __init__(self, engine):
    super().__init__(engine)
    self.text = "Display Text"

  def on_done(self):
    ic("Filler pipe is finished")

  def on_run(self):
    self.on_done()

  def on_error(self):
    print("ERROR")



