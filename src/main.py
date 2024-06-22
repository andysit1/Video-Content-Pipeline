

#Still development and learning the components

"""
Basic outline - of video pipeline

1.) splice videos by silence

2.) take all timestamps and make clip objects

3.) process them in a loop, ideally do it in parrel or some sort of async to speed it up
    Maybe thread it

4.) After iterate between the videos and form relations based on the data collected
    ALSO plot the data based on the time stamp to a histogram or bargraph

"""


from phase_one_extract.split_silence import get_clean_chunk_times




if __name__ == "__main__":

    in_filename='../input-video/demo_valorant.mov'
    out_pattern='../output-video/video{}.mp4'
    silence_threshold = -14
    silence_duration = 0.5
    seconds_between_clips_varriance = 10


    silence_intervals = get_clean_chunk_times(in_filename=in_filename, seconds_between_clips_varriance=seconds_between_clips_varriance,silence_threshold=silence_threshold, silence_duration=0.5)
    print(silence_intervals)