"""
ChatMonitor tests: line parsing / rate-window logic exercised directly
(no thread/socket needed), plus one test that drives the real background
thread against a fake in-memory connection to prove the threading and
PING/PONG wiring actually works.
"""
import queue
import time

import pytest

from VidFlow.live.chat_monitor import ChatMonitor


class FakeIRCConnection:
    """A queue-backed stand-in for the socket file object ChatMonitor uses."""

    def __init__(self):
        self.incoming = queue.Queue()
        self.sent = []
        self.closed = False

    def write(self, data):
        self.sent.append(data)

    def flush(self):
        pass

    def readline(self):
        try:
            return self.incoming.get(timeout=2)
        except queue.Empty:
            return "" if self.closed else "\r\n"

    def close(self):
        self.closed = True
        self.incoming.put("")  # unblock a pending readline()


def privmsg(channel, user, text):
    return ":{user}!{user}@{user}.tmi.twitch.tv PRIVMSG #{channel} :{text}\r\n".format(
        user=user, channel=channel, text=text)


class TestLineParsingDirect:
    """Exercises handle_line() synchronously -- no thread, no timing. These
    never call start(), so connection_factory is irrelevant; _conn is set
    directly so handle_line()'s PING reply has somewhere to write to."""

    def test_privmsg_for_the_watched_channel_is_recorded(self):
        monitor = ChatMonitor("somechannel")
        monitor._conn = FakeIRCConnection()
        monitor.handle_line(privmsg("somechannel", "viewer1", "hello").strip())
        assert monitor.messages_per_second() > 0

    def test_privmsg_for_a_different_channel_is_ignored(self):
        monitor = ChatMonitor("somechannel")
        monitor._conn = FakeIRCConnection()
        monitor.handle_line(privmsg("otherchannel", "viewer1", "hello").strip())
        assert monitor.messages_per_second() == 0

    def test_ping_is_answered_with_pong(self):
        monitor = ChatMonitor("somechannel")
        monitor._conn = FakeIRCConnection()
        monitor.handle_line("PING :tmi.twitch.tv")
        assert monitor._conn.sent == ["PONG :tmi.twitch.tv\r\n"]

    def test_messages_outside_the_window_are_pruned(self):
        monitor = ChatMonitor("somechannel", window_seconds=10)
        monitor._record_message(now=0.0)
        monitor._record_message(now=1.0)
        # both should have aged out by t=15 (window is [5, 15))
        assert monitor.messages_per_second(now=15.0) == 0

    def test_rate_reflects_messages_within_the_window(self):
        monitor = ChatMonitor("somechannel", window_seconds=10)
        for t in [1.0, 2.0, 3.0, 4.0, 5.0]:
            monitor._record_message(now=t)
        assert monitor.messages_per_second(now=5.0) == pytest.approx(5 / 10)


def test_start_joins_channel_and_processes_messages_via_the_background_thread():
    fake_conn = FakeIRCConnection()
    monitor = ChatMonitor("somechannel", connection_factory=lambda: fake_conn, window_seconds=10)

    monitor.start()
    try:
        assert any("JOIN #somechannel" in s for s in fake_conn.sent)

        fake_conn.incoming.put(privmsg("somechannel", "viewer1", "pog"))
        fake_conn.incoming.put(privmsg("somechannel", "viewer2", "pog"))
        fake_conn.incoming.put("PING :tmi.twitch.tv\r\n")

        deadline = time.time() + 5
        while time.time() < deadline and monitor.messages_per_second() == 0:
            time.sleep(0.05)

        assert monitor.messages_per_second() > 0
        assert any(s == "PONG :tmi.twitch.tv\r\n" for s in fake_conn.sent)
    finally:
        monitor.stop()

    assert fake_conn.closed
