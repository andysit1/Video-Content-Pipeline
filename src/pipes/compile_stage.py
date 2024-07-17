"""
  Compile stage handles compiling the clips I chose together into the final product

  Things I want
    -> go over the time line to make sure none of the video are overlapping
    -> get total time values
    -> given the total time we should be able to be more harsh with picking out more ideal clips
    -> In general, make sure all the clips not causing problems (last check) then compile
"""


import os
from icecream import ic
from src.modules.pipeline_builder import Pipe
from src.aggregate.ffmpeg_component import FFMPEGAggregate
import ast


class CompileVideoPipe(Pipe, FFMPEGAggregate):
  def __init__(self, engine):
    super().__init__(engine)
    self.analyze_data = os.path.join(self.engine.payload['cache_txt_out'], "analyze_data.txt")
    self.data = self.get_analyze_data() #lines of data ()

  def get_analyze_data(self):
    lines = self.read_lines(self.analyze_data)
    data = []

    total_points = 0
    average_points = 0

    for line in lines:
      converted_dict = ast.literal_eval(line) #ast.literal_eval is crazy good
      data.append(converted_dict)
      total_points += converted_dict['points']

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
        if video['points'] < threshold:
          self.data.remove(video)
      except:
        ic('ERROR', video)
        pass
    ic(len(self.data))

  #TODO
  def sort_videos_in_order_high(self):
    def myFunc(e):
      return e['points']

    #sorts the list from
    self.data[0:len(self.data)-1].sort(key=myFunc)
    return self.data[::-1]

  def compile_video(self):
    lines = self.sort_videos_in_order_high()[0:-1]
    file_compile_lines = ["file {}".format(line['name'].replace('\\', '/')) for line in lines[1:]]

    self.write_lines("tmp_file.txt", file_compile_lines)

    out_filename = os.path.join(self.engine.payload['clips_out'], self.engine.payload['video_name'] + ".mp4").replace('\\', '/')
    self.combine_videos_demuxer_method(out_filename)

  def on_done(self):
    self.engine.machine.current = None

  def on_run(self):
    ic("Running Compile Video Pipe")
    average_points = self.data[-1]['average_points']
    self.threshold_video_points(average_points)
    self.compile_video()
    self.on_done()
