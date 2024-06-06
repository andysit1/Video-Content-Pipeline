from split_silence import split_audio

def generate_inital_clips():
    #silence with long durations "2 seconds"
    split_audio(
        in_filename='../input-video/mine.mkv',
        out_pattern='../output-video/video{}.mp4',
        silence_threshold=-17,
        silence_duration=2
    )
# def remove_poor_clips():
class SingleVideoTool():
    def __init__(self):
        self.filename : str = None

    def on_fail(self):
        #returns a error to user and returns to endstate for developer
        #should return log of what when wrong if the very first video was error
        pass

    def on_success(self):
        #moves on to next phase of extraction return  True
        pass


class MultiVideoTool():
    def __init__(self):
        self.dir : str = None

    def on_fail(self):
        #returns a error to user and returns to endstate for developer
        #should return log of what when wrong if the very first video was error
        pass

    def on_success(self):
        #moves on to next phase of extraction return  True
        pass
