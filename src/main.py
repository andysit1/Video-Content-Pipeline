

#Still development and learning the components

"""
Basic outline - of video pipeline

1.) splice videos by silence

2.) take all timestamps and make clip objects

3.) proccess each clip and rank it by points

4.) take each node and graph it into a graph to see how it looks

"""

import os
from phase_one_extract.split_silence import get_clean_chunk_times, split_audio
import glob
from base.clip_object import Clip
from clip_ranker import Ranker
from icecream import ic
from frame_extraction.utils import combine_select_output_videos_into_video, concat_demuxer_method



if __name__ == "__main__":
    in_filename='../TD/VODS/sanch.mp4',
    silence_threshold=-10,
    silence_duration=3,
    clip_file = "../output-video/"
    out_pattern = "../output-video/video{}.mp4"

    if not os.path.exists(clip_file):
        raise TypeError("Path not found")
    
    #assuming we called the clean_chunk_times() already for the output-video file to be full of clips
    split_audio(
        in_filename='../TD/VODS/sanch.mp4',
        silence_threshold=-10,
        silence_duration=3,
        out_pattern=out_pattern
    )

    clips_filename = sorted(glob.glob(os.path.join(clip_file, "*mp4")), key=os.path.getmtime)
    clips_points = []
    ranker = Ranker()

    for clip in clips_filename:
        clip_obj = Clip(path=clip)

        # TODO: we can make it so they save the data into a dict so we can cache the frame

        ranker.load_clip(clip_obj)
        ranker.run()
        clips_points.append((clip, ranker.get_points()))

    # we need to create a way to scale points based on durations
    # clips that are 2 seconds long probably should not have high clip value as its shorter meaning
    # for clips to reach the same level of point it probably that clip had the crosshair over something interesting

    values = 0
    for x in clips_points:
        values += x[1]

    avg = values / len(clips_points)
    selected_clips = list(filter(lambda x: (x[1] > avg), clips_points))


    ic(selected_clips)
    concat_demuxer_method(selected_clips)
    # for clip in selected_clips:



