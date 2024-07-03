

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
from icecream import ic
from frame_extraction.utils import combine_select_output_videos_into_video, concat_demuxer_method



if __name__ == "__main__":

    # clip_file = "../output-video/"

    # if not os.path.exists(clip_file):
    #     raise TypeError("Path not found")


    # clips_filename = sorted(glob.glob(os.path.join(clip_file, "*mp4")), key=os.path.getmtime)
    # clips_points = []
    # ranker = Ranker()
    # for clip in clips_filename:
    #     clip_obj = Clip(path=clip)
    #     ranker.load_clip(clip_obj)
    #     ranker.run()
    #     clips_points.append((clip, ranker.get_points()))

    # # we need to create a way to scale points based on durations
    # # clips that are 2 seconds long probably should not have high clip value as its shorter meaning
    # # for clips to reach the same level of point it probably that clip had the crosshair over something interesting

    # print(clips_points)


    clips_points = [('../output-video\\video00.mp4', 2333.4755555555544), ('../output-video\\video02.mp4', 5973.075555555551), ('../output-video\\video04.mp4', 2095.946666666667), ('../output-video\\video05.mp4', 98.73777777777777), ('../output-video\\video06.mp4', 3812.0044444444447), ('../output-video\\video07.mp4', 393.14666666666676), ('../output-video\\video08.mp4', 658.72), ('../output-video\\video09.mp4', 2533.475555555558), ('../output-video\\video010.mp4', 93.6444444444444), ('../output-video\\video011.mp4', 18.013333333333343), ('../output-video\\video012.mp4', 2221.631111111115), ('../output-video\\video013.mp4', 534.7955555555556), ('../output-video\\video014.mp4', 182.65777777777657), ('../output-video\\video015.mp4', 218.0400000000001), ('../output-video\\video016.mp4', 2077.7555555555555), ('../output-video\\video017.mp4', 293.6444444444445), ('../output-video\\video018.mp4', 369.599999999999), ('../output-video\\video019.mp4', 1723.964444444444), ('../output-video\\video020.mp4', 976.0844444444447), ('../output-video\\video021.mp4', 220.40000000000003), ('../output-video\\video022.mp4', 89.85777777777776), ('../output-video\\video023.mp4', 1898.9288888888875), ('../output-video\\video024.mp4', 5717.871111111073), ('../output-video\\video025.mp4', 10720.622222222224), ('../output-video\\video026.mp4', 644.5466666666667), ('../output-video\\video027.mp4', 960.604444444442), ('../output-video\\video028.mp4', 1732.8799999999962), ('../output-video\\video030.mp4', 130.17777777777775), ('../output-video\\video032.mp4', 2007.835555555551), ('../output-video\\video033.mp4', 1934.297777777777), ('../output-video\\video034.mp4', 4266.199999999989), ('../output-video\\video035.mp4', 1790.6355555555515), ('../output-video\\video037.mp4', 6404.657777777777), ('../output-video\\video039.mp4', 2887.315555555559)]

    values = 0
    for x in clips_points:
        values += x[1]

    avg = values / len(clips_points)
    selected_clips = list(filter(lambda x: (x[1] > avg), clips_points))

    ic(selected_clips)

    concat_demuxer_method(selected_clips)
    # for clip in selected_clips:



