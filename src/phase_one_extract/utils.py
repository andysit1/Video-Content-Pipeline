import ffmpeg
import subprocess
import logging


logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)

#runs a command
def _logged_popen(cmd_line, *args, **kwargs):
    logger.debug('Running command: {}'.format(subprocess.list2cmdline(cmd_line)))
    return subprocess.Popen(cmd_line, *args, **kwargs)



import re


n_samples_re = re.compile(r'n_samples: (?P<n_samples>\d+)')

# Regular expression for mean_volume
mean_volume_re = re.compile(r'mean_volume: (?P<mean_volume>-?\d+(\.\d+)?) dB')

# Regular expression for max_volume
max_volume_re = re.compile(r'max_volume: (?P<max_volume>-?\d+(\.\d+)?) dB')

# Regular expression for histogram entries
histogram_re = re.compile(r'histogram_(?P<db_level>\d+)db: (?P<count>\d+)')


def get_mean_max(in_filename):
    p = _logged_popen(
      (ffmpeg
          .input(in_filename)
          .filter('volumedetect')
          .output('-', format='null')
          .compile()
      ) + ['-nostats'],  # FIXME: use .nostats() once it's implemented in ffmpeg-python.
      stderr=subprocess.PIPE
    )
    output = p.communicate()[1].decode('utf-8')
    lines = output.splitlines()

    for line in lines:
        mean_volume_match = mean_volume_re.search(line)
        max_volume_match = max_volume_re.search(line)
        histogram_matches = histogram_re.findall(line)

    print(mean_volume_match, max_volume_match, histogram_matches)

def detect_voice() -> bool:
    pass

if __name__ == "__main__":
    get_mean_max("../input-video/mine.mkv")