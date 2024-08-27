"""
  Compile stage handles compiling the clips I chose together into the final product

  Things I want
    -> go over the time line to make sure none of the video are overlapping
    -> get total time values
    -> given the total time we should be able to be more harsh with picking out more ideal clips
    -> In general, make sure all the clips not causing problems (last check) then compile
"""

from rich.console import Console
from queue import Queue
import os
from icecream import ic
from VidFlow.modules.pipeline_builder import Pipe
from VidFlow.aggregate.ffmpeg_component import FFMPEGAggregate
import ast
import logging
logger = logging.getLogger(__name__)

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
        if video[0]['points'] < threshold:
          self.data.remove(video)
      except:
        ic('ERROR', video)
        pass


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

  def compare_clips(self, line1: str, line2: str) -> bool:
    if line1[0] == line2[0]:
      return False
    ic("Comparing {} ANND {}".format(line1, line2))
    return True

  def get_video_order(self) -> list[str]:

    high_lower = self.threshold_video_points()
    q = Queue(maxsize=3)
    video_order = []
    for line in high_lower:
      if q.full():
        video_order.append(q.get())
        continue

      if q.qsize() <= 0:
        q.put(line)
        continue

      if self.compare_clips(line1=q.queue[0], line2=line):
        video_order.append(q.get())
        video_order.append(line)
      else:
        q.put(line)

    #just pushed the rest of q into video_order
    while not q.empty:
      video_order.append(q.get())

    return video_order

    """
      Implement Brain Storm
        for clip in high lower
          queue clip
            if queue.len <= 1
              continue to next clip to queue inorder for more than 2 clips

          compare_clips(line1, line2) -> true or false:

          if true
            apppend the two
          else:
            queue next
          check the next combinations

        "EXIT CONDITIONS" -> to never loop inf
        if queue lengh > 4:
          then pair first two clips and go.
          save the higher point clip for more idea and if it never matches i rather have a banger at the end of video

        if higher_lower is near end and no match
          just append rest to video

        ISSUE
          -> what if it never reaches conditions so we need an exit conditions
     """


  def compile_video(self):
    self.data.pop()
    lines = self.data #sorts and adds randomness
    file_compile_lines = ["file {}".format(line['name'].replace('\\', '/')) for line in lines[1:]]
    self.write_lines("tmp_file.txt", file_compile_lines)
    out_filename = os.path.join(self.engine.payload['clips_out'], self.engine.payload['video_name'] + ".mp4").replace('\\', '/')
    self.combine_videos_demuxer_method(out_filename)

  def on_done(self):
    #pass back to the engine to check if we still
    from VidFlow.pipes.dl_stage import DownloadPipe
    self.engine.machine.next_state = DownloadPipe(self.engine)

  def on_run(self):
    if not self.__DEBUG:
      ic("Running Compile Video Pipe")
      average_points = self.data[-1]['average_points']
      self.threshold_video_points(average_points) #drop the requirements...
      self.compile_video()
      self.on_done()
      return

    #move the print statements here
    ic("DEBUGGING COMPILE PIPELINE")
    average_points = self.data[-1]['average_points']
    self.threshold_video_points(average_points) # filter the list...
    self.get_video_order()
    self.on_done()


    """
      GOAL, videos that generate are too similar to each other
        scatter videos by duration
        scatter videos by SIFT algo

      Brain Storm
        If we want to do higher level analysis. We likely need to create sub data structure of a queue
          How this will work is we queue this clip and next clip, if they do not meet conditions
            THEN
              queue next video and see if there is a match or meets condition
              If DOES then match these two into list and move onto next video

        Conditions
          Focus point should relatively close.
          IF time is relevant
            do not sort base on points
            probably should force slice a video every 10 mins or 5 mins depending on size of stream.

          IF time is not relevant
            sort higher to low
            compare focus points and try to select points that line up nicely
    """



