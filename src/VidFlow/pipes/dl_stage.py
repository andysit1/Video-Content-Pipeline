"""
This Stage handles downloading from various APIS can be from youtube, twitch, medal, and so on...
  -> Not too important to worry about since for our needs this is more end game once we finished the rest up
  -> testing does not require constant download either so we can ignore for now
  -> subprocess the downloading of videos (requires a API connection some day)
"""


from icecream import ic
from VidFlow.modules.pipeline_builder import Pipe
from VidFlow.pipes.entry_stage import EntryPipe
import subprocess
import os


#give streamers to handle.

"""
  check for recent streams on twitch
    if true, return stream ID and title.
      then queue download streams

    on finish creation
      -> delete vod
      or we can do manually.
"""

# 2199720069
# cmd "../TD/TwitchDownloaderCLI.exe videodownload --id 612942303 -o video.mp4"
import logging
logger = logging.getLogger(__name__)


class DownloadPipe(Pipe):
  def __init__(self, engine):
    super().__init__(engine)

  def on_done(self):
    print("Done Download all clips!")
    self.engine.machine.current = None

  def on_error(self):
    return super().on_error()

  def on_run(self):
    while self.engine.q.qsize() > 0:
      payload = self.engine.q.get()
      in_filename = "E:/Projects/2024/Video-Content-Pipeline/TD/VODS/{}.mp4".format(payload['video_name'])
      payload['in_filename'] = in_filename
      print(in_filename)
      if not os.path.exists(in_filename):
        command = ["E:/Projects/2024/Video-Content-Pipeline/TD/TwitchDownloaderCLI.exe", "videodownload", "--id", str(payload['stream_id']), "-o" , in_filename]
        ic(command)
        subprocess.run(command, capture_output=True, text=True)

      ic()
      self.engine.load_payload(payload)
      self.engine.machine.current = EntryPipe(self.engine)
      return
    self.on_done()

