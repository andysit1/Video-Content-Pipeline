import subprocess
import sys
from utils import _logged_popen
from vosk import Model, KaldiRecognizer, SetLogLevel
import ffmpeg

#iterate through folder to read results...
SAMPLE_RATE = 16000
SetLogLevel(0)

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)


def is_detect_voice(in_filename) -> bool:
    p = _logged_popen(
      (ffmpeg
          .input(in_filename, loglevel="quiet", ac=1, f="s16le", ar=SAMPLE_RATE)
          .output('-', format='null')
          .compile()
      ) + ['-nostats'],  # FIXME: use .nostats() once it's implemented in ffmpeg-python.
      stderr=subprocess.PIPE
    )


    output = p.communicate()[1].decode('utf-8')
    lines = output.splitlines()
    print(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levels): %(message)s')
    logger.setLevel(logging.DEBUG)
    is_detect_voice("../output-video/video94.mp4")

# model = Model(lang="en-us")
# rec = KaldiRecognizer(model, SAMPLE_RATE)

# with subprocess.Popen(["ffmpeg", "-loglevel", "quiet", "-i",
#                             sys.argv[1],
#                             "-ar", str(SAMPLE_RATE) , "-ac", "1", "-f", "s16le", "-"],
#                             stdout=subprocess.PIPE) as process:

#     while True:
#         data = process.stdout.read(4000)
#         if len(data) == 0:
#             break
#         if rec.AcceptWaveform(data):
#             print(rec.Result())
#         else:
#             print(rec.PartialResult())

#     print(rec.FinalResult())