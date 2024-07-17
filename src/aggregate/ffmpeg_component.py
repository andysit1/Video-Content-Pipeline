


import sys
import subprocess
from icecream import ic
from filehandler_component import FileHandleComponent
import ffmpeg
import re
import os
import glob


"""
  File Standards - files names that are universal for this projects
    community -> this file will be cleared and emptied for any clips
    the-rest -> will be whenever we want to save specific videos for later comparison
                  -> needs a file system to handle the files
"""




#FFMPEG uses a lot of files from find instances of vides to files we need it
class FFMPEGAggregate(FileHandleComponent):
  def __init__(self, engine) -> None:
      self.engine = engine

      if self.engine:
         raise ValueError("Engine does not exist")

  def _logged_popen(cmd_line, *args, **kwargs):
      ic('Running command: {}'.format(subprocess.list2cmdline(cmd_line)))
      return subprocess.Popen(cmd_line, *args, **kwargs)

  def get_freeze_frames(self, in_filename):
      #TODO/FEATURE: call the freeze frames to check bad
      pass

  def load_frames(self):
      files = glob.glob("./frame_extraction/in_frame/*")
      return files


  def get_mean_max(self, in_filename) -> tuple:

    if not os.path.exists(in_filename):
        raise FileExistsError("path not found")

    p = self._logged_popen(
      (ffmpeg
          .input(in_filename)
          .filter('volumedetect')
          .output('-', format='null')
          .compile()
      ) + ['-nostats'],
      stderr=subprocess.PIPE
    )
    output = p.communicate()[1].decode('utf-8')
    lines = output.splitlines()
    return lines


  def silence_detect(self, in_filename, silence_threshold, silence_duration, start_time=None, end_time=None):

      input_kwargs = {}
      if start_time is not None:
          input_kwargs['ss'] = start_time
      else:
          start_time = 0.
      if end_time is not None:
          input_kwargs['t'] = end_time - start_time

      p = self._logged_popen(
        (ffmpeg
            .input(in_filename, **input_kwargs)
            .filter('silencedetect', n='{}dB'.format(silence_threshold), d=silence_duration)
            .output('-', format='null')
            .compile()
        ) + ['-nostats'],
        stderr=subprocess.PIPE
      )

      output = p.communicate()[1].decode('utf-8')

      if p.returncode != 0:
          sys.stderr.write(output)
          sys.exit(1)
      lines = output.splitlines()

      ic(lines)
      return lines

  def combine_videos_demuxer_method(self):
      #cmd: ffmpeg -f concat -safe 0 -i tmp_file.txt -c copy output.mp4

      try:
        self._logged_popen(
          (
            ffmpeg
              .input("tmp_file.txt", safe=0, f="concat")
              .output("test_imaqt.mp4", vcodec="copy")
              .overwrite_output()
              .compile()
          )
        ).communicate()

      except Exception as e:
        print("Error", e)
