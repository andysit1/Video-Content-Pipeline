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
from .action_stage import ActionPipe


import re
silence_start_re = re.compile(r' silence_start: (?P<start>[0-9]+(\.?[0-9]*))$')
silence_end_re = re.compile(r' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')
total_duration_re = re.compile(
    r'size=[^ ]+ time=(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9\.]{5}) bitrate=')



class AnalyzeDataFiles(Pipe, FileHandleComponent):
  def __init__(self, engine):
    super().__init__(engine)

    #handle chunking
    self.chunk_path = os.path.join(self.engine.payload['cache_txt_out'], 'chunks.txt')

  def on_done(self):
    ic()
    self.engine.machine.current = ActionPipe(self.engine)

  def get_data(self):
    volume_detect = os.path.join(self.engine.payload['cache_txt_out'], 'volume_detect.txt')
    silence_detect = os.path.join(self.engine.payload['cache_txt_out'], 'silence_detect.txt')

    volume_data = self.read_lines(volume_detect)
    silence_data = self.read_lines(silence_detect)

    ic(volume_data[-1])
    ic(silence_data[-1])

    return volume_data, silence_data


  def if_chunks_txt(self) -> bool:
     return self.path_exists(self.chunk_path)

  def analyze_silence(self):
    if not self.if_chunks_txt():
      silence_detect_path = os.path.join(self.engine.payload['cache_txt_out'], 'silence_detect.txt')
      lines = self.read_lines(silence_detect_path)

      start_time = None
      end_time = None
      chunk_starts = []
      chunk_ends = []
      seconds_between_clips_varriance = 3

      for line in lines:
          ic(line)
          ic()
          silence_start_match = silence_start_re.search(line)
          silence_end_match = silence_end_re.search(line)
          total_duration_match = total_duration_re.search(line)
          ic(silence_start_match, silence_end_match, total_duration_match)

          if silence_start_match:
              chunk_ends.append(float(silence_start_match.group('start')))
              if len(chunk_starts) == 0:
                  # Started with non-silence.
                  chunk_starts.append(start_time or 0.)
          elif silence_end_match:
              chunk_starts.append(float(silence_end_match.group('end')))
          elif total_duration_match:
              hours = int(total_duration_match.group('hours'))
              minutes = int(total_duration_match.group('minutes'))
              seconds = float(total_duration_match.group('seconds'))
              end_time = hours * 3600 + minutes * 60 + seconds

      if len(chunk_starts) == 0:
          # No silence found.
          chunk_starts.append(start_time)

      if len(chunk_starts) > len(chunk_ends):
          # Finished with non-silence.
          chunk_ends.append(end_time or 10000000.)

      chunks = list(zip(chunk_starts, chunk_ends))


      previous_end = 0
      #merges clips intervals together if within 3 second intervals of each other
      for i, (start_time, end_time) in enumerate(chunks):
          if i == 0:
              previous_end = end_time
              continue

          if start_time - previous_end <= seconds_between_clips_varriance:
              chunks[i - 1] = chunks[i - 1] + chunks[i]
              chunks.remove(chunks[i])
          previous_end = end_time

      #clips should be atless 1.5 inlength
      cleaned_chunks = [interval for interval in chunks if round(interval[1] - interval[0], 3) >= 1.5]

      ic(cleaned_chunks)
      self.write_lines(self.chunk_path, cleaned_chunks)

  def get_chunk_data(self):
    lines = self.read_lines(self.chunk_path)
    chunks = []

    for line in lines:
      cleaned = line[1:-2].split(', ')
      chunks.append((float(cleaned[0]), float(cleaned[-1])))

    ic(chunks)



  def on_run(self):
    print("Running AnalyzeDataFiles")
    ic.disable()
    self.analyze_silence()
    self.on_done()


