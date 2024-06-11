from .split_silence import split_audio, read_file_silence
import re
from .utils import _logged_popen, mean_volume_re, max_volume_re, histogram_re
import ffmpeg
freeze_start_re = re.compile(r'lavfi\.freezedetect\.freeze_start=(?P<start>[0-9]+(\.[0-9]*)?)')
freeze_end_re = re.compile(r'lavfi\.freezedetect\.freeze_end=(?P<end>[0-9]+(\.[0-9]*)?)')
freeze_duration_re = re.compile(r'lavfi\.freezedetect\.freeze_duration=(?P<duration>[0-9]+(\.[0-9]*)?)')


import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

def generate_inital_clips():
    #silence with long durations "2 seconds"
    split_audio(
        in_filename='../input-video/mine.mkv',
        out_pattern='../output-video/video{}.mp4',
        silence_threshold=-17,
        silence_duration=2
    )

def read_file_volumedetect(filename):
    data = open(filename, "r").read().splitlines()

    # ISSUE, not reading the matches for some reason when we pull using open
    # Works fine when commented as data varible.

    # Find matches
    for line in data:
        mean_volume_match = mean_volume_re.search(line)
        max_volume_match = max_volume_re.search(line)
        histogram_matches = histogram_re.findall(line)

        if mean_volume_match:
            print("mean_volume:", mean_volume_match.group('mean_volume'))

        if max_volume_match:
            print("max_volume:", max_volume_match.group('max_volume'))

        for match in histogram_matches:
            db_level, count = match
            print(f"histogram_{db_level}db: {count}")


#Notes
    #Freeze does an amazing job finding low action clips
def read_file_freeze(filename):
    lines = open(filename, "r").read().splitlines()

    chunk_start = []
    chunk_end = []
    chunk_duration = []

    for line in lines:
        start = freeze_start_re.search(line)
        end = freeze_end_re.search(line)
        length = freeze_duration_re.search(line)

        if start:
            chunk_start.append(float(start.group('start')))

        if end:
            chunk_end.append(float(end.group('end')))

        if length:
            chunk_duration.append(float(length.group('duration')))

    if len(chunk_start) == 0:
        # No silence found.
        chunk_start.append(start.group('start'))
        chunk_duration.append(0)


    if len(chunk_start) > len(chunk_end):
        # Finished with non-silence.
        chunk_end.append(end.group('end') or 10000000.)

    return list(zip(chunk_start, chunk_end, chunk_duration))

# type.... (start, end, duration)
def generate_video_freeze(chunks):
    in_file = "../input-video/mine.mkv"
    for i, (start, end, duration) in enumerate(chunks):
        out_file = "../output-video/video{}.mp4".format(i)
        logging.info('start:{} end:{} time:{}'.format(start, end, duration))
        _logged_popen(
            (ffmpeg
                .input(in_file, ss=start, t=duration)
                .output(out_file)
                .overwrite_output()
                .compile()
            )
        ).communicate()



if __name__ == "__main__":
    # print(read_file_silence("../output-text/vol.txt"))
    # chunks = read_file_freeze("../output-text/mine_freeze.txt")
    read_file_volumedetect("../output-text/mine_volumedetect.txt")


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
