

"""
This Stage handles the actual splicing of the video and is the most time consuming part of the program

  place the video splitting occurs
  ideally if we ever find a better way of splicing videos it will go here. Probably a lot of min maxing memory and making sure
  we aren't holding resources we dont need

  cuda/gpu rendering? idk we will figure that out later
"""


import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.aggregate.ffmpeg_component import FFMPEGAggregate
from .analyze_clip_stage import AnalyzeClipsPipe

class ActionPipe(Pipe, FFMPEGAggregate):
    def __init__(self, engine):
      super().__init__(engine)

      #handle chunking
      self.chunk_path = os.path.join(self.engine.payload['cache_txt_out'], 'chunks.txt')


    def get_chunk_data(self):
      lines = self.read_lines(self.chunk_path)
      chunks = []

      for line in lines:
        cleaned = line[1:-2].split(', ')
        chunks.append((float(cleaned[0]), float(cleaned[-1])))

      ic(chunks)
      return chunks


    def if_videos_cliped(self) -> bool:
       if self.file_exists(os.path.join(self.engine.payload['clips_out'], 'video00.mp4')):
          return True
       return False

    def split_into_clips(self):
      if not self.if_videos_cliped():

        chunk_times = self.get_chunk_data()
        ic("Clipping {len(chunk_times} number of clips")

        out_pattern = os.path.join(self.engine.payload['clips_out'], 'video{}.mp4')
        # total_time = 0

        for i, chunk in enumerate(chunk_times):
          start = chunk[0]
          end = chunk[-1]

          time = round(end - start, 3)
          # total_time += time
          ic(time)

          if 0 < i or i < 10:
              out_filename_tail = str(0) + str(i)
              out_filename = out_pattern.format(out_filename_tail)
          else:
              out_filename = out_pattern.format(i, i=i)

          ic(out_filename)

          self.split_video(
            in_filename=self.engine.payload['in_filename'],
            out_filename=out_filename,
            start=start,
            time=time
          )
        # self.append_to_file()
    def on_done(self):
      ic("Finished Pipeline")
      self.engine.machine.current = AnalyzeClipsPipe(self.engine)

    def on_run(self):
      ic.enable()
      self.split_into_clips()
      self.on_done()
