"""
  Compile stage handles compiling the clips I chose together into the final product

  Things I want
    -> go over the time line to make sure none of the video are overlapping
    -> get total time values
    -> given the total time we should be able to be more harsh with picking out more ideal clips
    -> In general, make sure all the clips not causing problems (last check) then compile
"""

from rich.console import Console

import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.aggregate.ffmpeg_component import FFMPEGAggregate
import ast


class CompileVideoPipe(Pipe, FFMPEGAggregate):
  def __init__(self, engine):
    super().__init__(engine)
    self.__DEBUG = False
    self.analyze_data = os.path.join(self.engine.payload['cache_txt_out'], "analyze_data.txt")

    self.data = self.get_analyze_data() #lines of data ()


  def get_analyze_data(self):
    lines = self.read_lines(self.analyze_data)
    data = []

    total_points = 0
    average_points = 0

    for line in lines:
      converted_dict = ast.literal_eval(line) #ast.literal_eval is crazy good
      ic("duration {}".format(converted_dict[1]))
      data.append(converted_dict)
      total_points += converted_dict[0]['points']

    average_points = total_points / len(lines)

    stats_obj = {
      'total_points' : total_points,
      'average_points' : average_points
    }

    data.append(stats_obj)
    return data

  # def count_video_size(self):
    # for clip_data in self.data:

  #threshold videos based on a cutoff of value
  def threshold_video_points(self, threshold :int):
    ic(len(self.data))
    for video in self.data[:-1]:
      try:
        if video[0]['points'] < threshold:
          self.data.remove(video)
      except:
        ic('ERROR', video)
        pass
    ic(len(self.data))

  #TODO

  """
    Feature scatter feature -> scatters the point system
                              this is good so we can have more diverse clips and not all the same
                            -> fix the ranking point system
                                -> too skewed to one direction
  """


  def sort_videos_in_order_high(self):
    def myFunc(e):
      return e[0]['points']

    ic(self.data)
    #sorts the list from
    without_stats = self.data[0:len(self.data)-1]
    without_stats.sort(key=myFunc, reverse=True)
    return without_stats
    # ic(self.data)

  def add_randomness_to_videos(self):
    import random
    from rich.console import Console
    from rich.table import Table

    data = self.sort_videos_in_order_high()

    table = Table(title="AFTER")
    table.add_column("S. No.", style="cyan", no_wrap=True)
    table.add_column("AFTER", style="magenta")
    table.add_column("BEFORE", justify="right", style="green")
    console = Console()

    for i, line in enumerate(data):
      if random.randint(1,3) == 1:
        sublist = data[i:i+3]
        random.shuffle(sublist)
        ic(data[i:i+3], sublist)
        data[i:i+3] = sublist  # Assign the shuffled sublist back to the original list

      if self.__DEBUG:
        table.add_row(str(i), str(line), str(self.data[i]))

    if self.__DEBUG:
      console.print(table)

    return data

  def compile_video(self):
    lines = self.add_randomness_to_videos() #sorts and adds randomness
    file_compile_lines = ["file {}".format(line[0]['name'].replace('\\', '/')) for line in lines[1:]]
    self.write_lines("tmp_file.txt", file_compile_lines)
    out_filename = os.path.join(self.engine.payload['clips_out'], self.engine.payload['video_name'] + ".mp4").replace('\\', '/')
    self.combine_videos_demuxer_method(out_filename)

  def on_done(self):
    self.engine.machine.current = None

  def on_run(self):

    if not self.__DEBUG:
      ic("Running Compile Video Pipe")
      average_points = self.data[-1]['average_points']
      self.threshold_video_points(average_points)
      self.compile_video()
      self.on_done()
      return
    #move the print statements here
    ic("DEBUGGING COMPILE PIPELINE")
    average_points = self.data[-1]['average_points']
    self.threshold_video_points(average_points)
    self.add_randomness_to_videos()
    self.on_done()


    """
      GOAL, videos that generate are too similar to each other
        scatter videos by duration
        scatter videos by SIFT algo
    """



