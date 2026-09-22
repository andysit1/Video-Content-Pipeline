"""
Twitch chat velocity: a lightweight, read-only IRC client tracking a
rolling messages-per-second rate. Chat reaction is Tier 1's second trigger
signal -- audio spikes catch loud moments, but a chat-rate spike often
confirms one, or catches something quieter but still exciting on its own
(a clutch play, a funny moment, a big reveal) that audio alone would miss.

Twitch's chat IRC allows anonymous read-only access: connect with an
unauthenticated "justinfan#####" nick and no password. That means this
needs no OAuth token or user login, unlike a bot that wants to post.

`connection_factory` is injectable (returns anything with .write/.flush/
.readline/.close) so tests exercise the real line-parsing and rate-window
logic against a fake in-memory connection instead of a real socket.
"""
import random
import socket
import threading
import time
from collections import deque

TWITCH_IRC_HOST = "irc.chat.twitch.tv"
TWITCH_IRC_PORT = 6667


class ChatMonitor:
    def __init__(self, channel, connection_factory=None, window_seconds=10):
        self.channel = channel.lower()
        self.connection_factory = connection_factory or self._connect_socket
        self.window_seconds = window_seconds
        self._timestamps = deque()
        self._lock = threading.Lock()
        self._conn = None
        self._thread = None
        self._running = False

    def _connect_socket(self):
        sock = socket.create_connection((TWITCH_IRC_HOST, TWITCH_IRC_PORT), timeout=10)
        return sock.makefile("rw", encoding="utf-8", newline="\r\n")

    def start(self):
        self._conn = self.connection_factory()
        nick = "justinfan{}".format(random.randint(10000, 99999))
        self._send("NICK {}".format(nick))
        self._send("JOIN #{}".format(self.channel))

        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _send(self, line):
        self._conn.write(line + "\r\n")
        self._conn.flush()

    def _read_loop(self):
        while self._running:
            try:
                line = self._conn.readline()
            except Exception:
                break
            if not line:
                break
            self.handle_line(line.strip())

    def handle_line(self, line):
        """
        Processes one raw IRC line. Exposed directly (not just via the
        background thread) so tests can feed lines synchronously.
        """
        if line.startswith("PING"):
            self._send(line.replace("PING", "PONG", 1))
            return
        if " PRIVMSG #{} ".format(self.channel) in line:
            self._record_message()

    def _record_message(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            self._timestamps.append(now)
            self._prune(now)

    def _prune(self, now):
        cutoff = now - self.window_seconds
        while self._timestamps and self._timestamps[0] < cutoff:
            self._timestamps.popleft()

    def messages_per_second(self, now=None):
        now = time.time() if now is None else now
        with self._lock:
            self._prune(now)
            return len(self._timestamps) / self.window_seconds
