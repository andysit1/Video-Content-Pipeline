


import sys
import subprocess
import ffmpeg
from icecream import ic
import ffmpeg


from .filehandler_component import FileHandleComponent

import re
import os
import glob


"""
  File Standards - files names that are universal for this projects
    community -> this file will be cleared and emptied for any clips
    the-rest -> will be whenever we want to save specific videos for later comparison
                  -> needs a file system to handle the files
"""


def _logged_popen(cmd_line, *args, **kwargs):
    ic('Running command: {}'.format(subprocess.list2cmdline(cmd_line)))
    return subprocess.Popen(cmd_line, *args, **kwargs)

class FFMPEGAggregate(FileHandleComponent):
  def __init__(self, engine, debug=False) -> None:
      self.engine = engine
      if not self.engine and not debug:
         raise ValueError("Engine does not exist")

  # def get_freeze_frames(self, in_filename):
  #     #TODO/FEATURE: call the freeze frames to check bad
  #     pass

  def load_frames(self):
      files = glob.glob("./frame_extraction/in_frame/*")
      return files

  def get_mean_max(self, in_filename) -> tuple:

    if not os.path.exists(in_filename):
        raise FileExistsError("path not found")

    cmd = (ffmpeg
          .input(in_filename)
          .filter('volumedetect')
          .output('-', format='null')
          .compile()
        ) + ['-nostats']

    p = _logged_popen(
      cmd_line=cmd,
      stderr=subprocess.PIPE
    )

    output = p.communicate()[1].decode('utf-8')
    lines = output.splitlines()
    return lines

  def split_video(self, in_filename, out_filename, start, time):
      _logged_popen(
            (ffmpeg
                .input(in_filename, ss=start, t=time)
                .output(out_filename)
                .overwrite_output()
                .compile()
            ),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).communicate()


  def silence_detect(self, in_filename, silence_threshold, silence_duration, start_time=None, end_time=None):

      input_kwargs = {}
      if start_time is not None:
          input_kwargs['ss'] = start_time
      else:
          start_time = 0.
      if end_time is not None:
          input_kwargs['t'] = end_time - start_time

      p = _logged_popen(
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
      ic(type(lines), lines)

      return lines

  def combine_videos_demuxer_method(self, out_filename):
      print(out_filename)
      #this place does not compile... whyyyyyyy

      #cmd: ffmpeg -f concat -safe 0 -i tmp_file.txt -c copy output.mp4
      try:
        _logged_popen(
          (
            ffmpeg
              .input('tmp_file.txt', safe=0, f="concat")
              .output(out_filename, vcodec="copy")
              .overwrite_output()
              .compile()
          )
        ).communicate()
      except Exception as e:
        print("Error", e)


if __name__ == "__main__":
    #Test if the functions works
    in_filename = "E:/Projects/2024/Video-Content-Pipeline/input-video/demo_valorant.mov"
    ffmpeg_hand = FFMPEGAggregate(engine=None, debug=True)
    volume_detect_lines = ffmpeg_hand.get_mean_max(in_filename=in_filename)
    silence_detect_lines = ffmpeg_hand.silence_detect(
                                                    in_filename=in_filename,
                                                      silence_threshold=-13,
                                                      silence_duration=0.5,
                                                    )



