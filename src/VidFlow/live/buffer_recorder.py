"""
Tier 1 (instant clips): continuously records an ffmpeg-readable input into
fixed-length segment files ("the buffer"), deleting segments older than a
retention window so disk usage stays bounded. This is the "instant replay"
window a highlight trigger reaches back into -- chat/crowd reaction lags
the actual moment by a couple seconds, so you need to already have it on
disk by the time something signals "clip this".

`input_url` is deliberately just whatever ffmpeg's `-i` accepts: a live
stream URL (resolved by streamlink -- see watcher.py) in production, or a
looped local test file in tests. The recorder doesn't know or care which.
"""
import glob
import os
import re
import subprocess

_SEGMENT_RE = re.compile(r"buf_(\d+)\.ts$")


class RollingBufferRecorder:
    def __init__(self, input_url, output_dir, segment_seconds=5, retention_seconds=90,
                 extra_input_args=None):
        self.input_url = input_url
        self.output_dir = output_dir
        self.segment_seconds = segment_seconds
        self.retention_seconds = retention_seconds
        self.extra_input_args = extra_input_args or []
        self._process = None
        os.makedirs(output_dir, exist_ok=True)

    def _segment_pattern(self):
        return os.path.join(self.output_dir, "buf_%06d.ts")

    def start(self):
        cmd = (
            ["ffmpeg", "-y", "-loglevel", "error"]
            + self.extra_input_args
            + ["-i", self.input_url,
               "-c", "copy",
               "-f", "segment",
               "-segment_time", str(self.segment_seconds),
               "-reset_timestamps", "1",
               self._segment_pattern()]
        )
        self._process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return self._process

    def is_running(self):
        return self._process is not None and self._process.poll() is None

    def stop(self):
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()

    def segments(self):
        """Segment file paths in recording order (oldest first)."""
        files = glob.glob(os.path.join(self.output_dir, "buf_*.ts"))
        return sorted(files, key=lambda p: int(_SEGMENT_RE.search(p).group(1)))

    def prune_old_segments(self):
        """Deletes segments beyond the retention window, oldest first."""
        segs = self.segments()
        max_segments = max(1, int(self.retention_seconds / self.segment_seconds) + 2)
        for old in segs[:-max_segments] if len(segs) > max_segments else []:
            try:
                os.remove(old)
            except FileNotFoundError:
                pass
