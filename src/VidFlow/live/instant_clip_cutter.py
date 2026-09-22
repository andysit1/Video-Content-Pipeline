"""
Cuts a finished clip out of the rolling buffer once a trigger fires:
picks the segments covering [event_time - pre, event_time + post] and
concatenates them with ffmpeg's concat demuxer (the same technique
combine_videos_demuxer_method uses in the batch pipeline).
"""
import os
import re
import subprocess

_SEGMENT_RE = re.compile(r"buf_(\d+)\.ts$")


def segments_covering_window(recorder, event_time, pre_seconds, post_seconds):
    """
    Picks buffer segments overlapping [event_time - pre, event_time + post].
    `event_time` is seconds since the recorder started -- segment N covers
    roughly [N * segment_seconds, (N+1) * segment_seconds), since each
    segment is reset_timestamps'd from recording start.
    """
    seg_len = recorder.segment_seconds
    start_idx = max(0, int((event_time - pre_seconds) // seg_len))
    end_idx = int((event_time + post_seconds) // seg_len)

    picked = []
    for seg in recorder.segments():
        idx = int(_SEGMENT_RE.search(seg).group(1))
        if start_idx <= idx <= end_idx:
            picked.append(seg)
    return picked


def cut_clip_from_segments(segment_paths, output_path):
    """Concatenates segment files (in order) into one clip at output_path."""
    if not segment_paths:
        raise ValueError("no segments given to cut a clip from")

    list_path = output_path + ".concat.txt"
    with open(list_path, "w") as f:
        for seg in segment_paths:
            escaped = os.path.abspath(seg).replace("'", "'\\''")
            f.write("file '{}'\n".format(escaped))

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", list_path,
        "-c", "copy",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    finally:
        os.remove(list_path)

    if result.returncode != 0:
        raise RuntimeError("ffmpeg concat failed: {}".format(result.stderr))
    return output_path
