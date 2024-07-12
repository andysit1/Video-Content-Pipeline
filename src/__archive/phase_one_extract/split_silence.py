
#EXAMPLE from offical repository

from __future__ import unicode_literals

import argparse
import errno
import ffmpeg
import logging
import os
import re
import subprocess
import sys

from icecream import ic

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

DEFAULT_DURATION = 0.3
DEFAULT_THRESHOLD = -60

parser = argparse.ArgumentParser(description='Split media into separate chunks wherever silence occurs')
parser.add_argument('in_filename', help='Input filename (`-` for stdin)')
parser.add_argument('out_pattern', help='Output filename pattern (e.g. `out/chunk_{:04d}.wav`)')
parser.add_argument('--silence-threshold', default=DEFAULT_THRESHOLD, type=int, help='Silence threshold (in dB)')
parser.add_argument('--silence-duration', default=DEFAULT_DURATION, type=float, help='Silence duration')
parser.add_argument('--start-time', type=float, help='Start time (seconds)')
parser.add_argument('--end-time', type=float, help='End time (seconds)')
parser.add_argument('-v', dest='verbose', action='store_true', help='Verbose mode')

silence_start_re = re.compile(r' silence_start: (?P<start>[0-9]+(\.?[0-9]*))$')
silence_end_re = re.compile(r' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')
total_duration_re = re.compile(
    r'size=[^ ]+ time=(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9\.]{5}) bitrate=')


def _logged_popen(cmd_line, *args, **kwargs):
    logger.debug('Running command: {}'.format(subprocess.list2cmdline(cmd_line)))
    return subprocess.Popen(cmd_line, *args, **kwargs)


def output_to_file(text):
    with open('debug.txt', 'a') as f:
        f.write(text + '\n')


def read_file_silence(filename, start_time=None, end_time=None):

    lines = open(filename, "r").read().splitlines()
    chunk_starts = []
    chunk_ends = []


    for line in lines:
        silence_start_match = silence_start_re.search(line)
        silence_end_match = silence_end_re.search(line)
        total_duration_match = total_duration_re.search(line)
        if silence_start_match:
            chunk_ends.append(float(silence_start_match.group('start')))
            if len(chunk_starts) == 0:
                # Started with non-silence.
                chunk_starts.append(start_time or 0.)
        elif silence_end_match:
            chunk_starts.append(float(silence_end_match.group('end')))
        elif total_duration_match:
            hours = int(total_duration_match.group('hours'))
            minutes = int(total_duration_match.group('minutes'))
            seconds = float(total_duration_match.group('seconds'))
            end_time = hours * 3600 + minutes * 60 + seconds

    if len(chunk_starts) == 0:
        # No silence found.
        chunk_starts.append(start_time)

    if len(chunk_starts) > len(chunk_ends):
        # Finished with non-silence.
        chunk_ends.append(end_time or 10000000.)

    return list(zip(chunk_starts, chunk_ends))


def _makedirs(path):
    """Python2-compatible version of ``os.makedirs(path, exist_ok=True)``."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST or not os.path.isdir(path):
            raise

ic.configureOutput(prefix='DEBUG| ', outputFunction=output_to_file)

def get_chunk_times(in_filename, silence_threshold, silence_duration, start_time=None, end_time=None):
    input_kwargs = {}
    if start_time is not None:
        input_kwargs['ss'] = start_time
    else:
        start_time = 0.
    if end_time is not None:
        input_kwargs['t'] = end_time - start_time

    p = _logged_popen(
        (ffmpeg
            .input(in_filename, **input_kwargs)
            .filter('silencedetect', n='{}dB'.format(silence_threshold), d=silence_duration)
            .output('-', format='null')
            .compile()
        ) + ['-nostats'],
        stderr=subprocess.PIPE
    )

    output = p.communicate()[1].decode('utf-8')
    if p.returncode != 0:
        sys.stderr.write(output)
        sys.exit(1)
    logger.debug(output)
    lines = output.splitlines()

    # Chunks start when silence ends, and chunks end when silence starts.
    chunk_starts = []
    chunk_ends = []
    for line in lines:
        silence_start_match = silence_start_re.search(line)
        silence_end_match = silence_end_re.search(line)
        total_duration_match = total_duration_re.search(line)
        if silence_start_match:
            chunk_ends.append(float(silence_start_match.group('start')))
            if len(chunk_starts) == 0:
                # Started with non-silence.
                chunk_starts.append(start_time or 0.)
        elif silence_end_match:
            chunk_starts.append(float(silence_end_match.group('end')))
        elif total_duration_match:
            hours = int(total_duration_match.group('hours'))
            minutes = int(total_duration_match.group('minutes'))
            seconds = float(total_duration_match.group('seconds'))
            end_time = hours * 3600 + minutes * 60 + seconds

    if len(chunk_starts) == 0:
        # No silence found.
        chunk_starts.append(start_time)

    if len(chunk_starts) > len(chunk_ends):
        # Finished with non-silence.
        chunk_ends.append(end_time or 10000000.)

    return list(zip(chunk_starts, chunk_ends))


#needs to add like 10-14 min leeway for the clippping so that clips dont end to akwardly

def get_clean_chunk_times(in_filename, silence_threshold, silence_duration, seconds_between_clips_varriance : int):

    silence_intervals = get_chunk_times(in_filename=in_filename, silence_threshold=silence_threshold, silence_duration=silence_duration)
    ic.configureOutput(prefix='DEBUG| ', outputFunction=output_to_file)
    ic(silence_intervals)

    previous_end = 0

    #merges clips intervals together if within 3 second intervals of each other
    for i, (start_time, end_time) in enumerate(silence_intervals):
        if i == 0:
            previous_end = end_time
            continue

        if start_time - previous_end <= seconds_between_clips_varriance:
            silence_intervals[i - 1] = silence_intervals[i - 1] + silence_intervals[i]
            silence_intervals.remove(silence_intervals[i])
        previous_end = end_time

    #clips should be atless 1.5 inlength
    cleaned_intervals = [interval for interval in silence_intervals if round(interval[1] - interval[0], 3) >= 1.5]
    ic(cleaned_intervals)
    return cleaned_intervals


#RULE: The game audio must be lower than the voice in general. This is because it ruins the scripts
#and overall video quality.
def split_audio(
    in_filename,
    out_pattern,
    silence_threshold=DEFAULT_THRESHOLD,
    silence_duration=DEFAULT_DURATION,
    start_time=None,
    end_time=None,
    verbose=False,
):
    chunk_times = get_clean_chunk_times(in_filename, silence_threshold, silence_duration, seconds_between_clips_varriance=3)
    ic(chunk_times)

    for i, chunk in enumerate(chunk_times):

        start = chunk[0]
        end = chunk[-1]

        time = round(end - start, 3)
        ic(time)

        if 0 < i or i < 10:
            out_filename_tail = str(0) + str(i)
            out_filename = out_pattern.format(out_filename_tail)
        else:
            out_filename = out_pattern.format(i, i=i)

        _makedirs(os.path.dirname(out_filename))
        ic(out_filename)
        logger.info('{}: start={:.02f}, end={:.02f}, duration={:.02f}'.format(out_filename, start, end,
            time))
        _logged_popen(
            (ffmpeg
                .input(in_filename, ss=start, t=time)
                .output(out_filename)
                .overwrite_output()
                .compile()
            ),
            stdout=subprocess.PIPE if not verbose else None,
            stderr=subprocess.PIPE if not verbose else None,
        ).communicate()



if __name__ == '__main__':
    # kwargs = vars(parser.parse_args())
    # if kwargs['verbose']:
    # split_audio(**kwargs)


    # c = split_audio(
    #     in_filename='../TD/VODS/sanch.mp4',
    #     silence_threshold=-10,
    #     silence_duration=3,
    # )

    split_audio(
        in_filename='../TD/VODS/imaqtpie.mp4',
        silence_threshold=-13,
        silence_duration=0.5,
        out_pattern='../output-video/video{}.mp4'
    )


