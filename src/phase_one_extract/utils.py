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
    print(lines)

def detect_voice() -> bool:
    pass

if __name__ == "__main__":
    get_mean_max("../input-video/mine.mkv")