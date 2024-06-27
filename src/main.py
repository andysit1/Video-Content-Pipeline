

#Still development and learning the components

"""
Basic outline - of video pipeline

1.) splice videos by silence

2.) take all timestamps and make clip objects

3.) proccess each clip and rank it by points

4.) take each node and graph it into a graph to see how it looks

"""

import os
from phase_one_extract.split_silence import get_clean_chunk_times
import glob
from base.clip_object import Clip
from clip_ranker import Ranker



if __name__ == "__main__":

    clip_file = "../output-video/"

    if not os.path.exists(clip_file):
        raise TypeError("Path not found")


    clips_filename = sorted(glob.glob(os.path.join(clip_file, "*mp4")), key=os.path.getmtime)
    clips_points = []
    ranker = Ranker()
    for clip in clips_filename:
        clip_obj = Clip(path=clip)
        ranker.load_clip(clip_obj)
        ranker.run()
        clips_points.append((clip, ranker.get_points()))

    # we need to create a way to scale points based on durations
    # clips that are 2 seconds long probably should not have high clip value as its shorter meaning
    # for clips to reach the same level of point it probably that clip had the crosshair over something interesting

    print(clips_points)

