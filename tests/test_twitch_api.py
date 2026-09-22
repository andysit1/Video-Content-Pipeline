"""
TwitchClient tests against a fake requests.Session -- no real network or
credentials needed, since the client's `session` is injectable.
"""
import pytest

from VidFlow.live.twitch_api import TwitchAPIError, TwitchClient


class FakeResponse:
    def __init__(self, status_code, json_data, text=""):
        self.status_code = status_code
        self._json_data = json_data
        self.text = text or str(json_data)

    def json(self):
        return self._json_data


class FakeSession:
    """Records every call and replays canned responses in call order per-verb."""

    def __init__(self):
        self.post_responses = []
        self.get_responses = []
        self.post_calls = []
        self.get_calls = []

    def post(self, url, data=None):
        self.post_calls.append((url, data))
        return self.post_responses.pop(0)

    def get(self, url, headers=None, params=None):
        self.get_calls.append((url, headers, params))
        return self.get_responses.pop(0)


@pytest.fixture
def fake_session():
    return FakeSession()


@pytest.fixture
def client(fake_session):
    return TwitchClient(client_id="cid", client_secret="secret", session=fake_session)


def queue_token(fake_session, expires_in=3600):
    fake_session.post_responses.append(
        FakeResponse(200, {"access_token": "tok123", "expires_in": expires_in})
    )


class TestAuth:
    def test_fetches_and_caches_token_across_calls(self, client, fake_session):
        queue_token(fake_session)
        fake_session.get_responses.append(FakeResponse(200, {"data": []}))
        fake_session.get_responses.append(FakeResponse(200, {"data": []}))

        client.get_users(["someone"])
        client.get_users(["someone_else"])

        assert len(fake_session.post_calls) == 1, "token should only be fetched once while valid"
        for _, headers, _ in fake_session.get_calls:
            assert headers["Authorization"] == "Bearer tok123"
            assert headers["Client-Id"] == "cid"

    def test_refetches_token_once_expired(self, client, fake_session, monkeypatch):
        import VidFlow.live.twitch_api as twitch_api_module

        fake_time = [1000.0]
        monkeypatch.setattr(twitch_api_module.time, "time", lambda: fake_time[0])

        queue_token(fake_session, expires_in=100)
        fake_session.get_responses.append(FakeResponse(200, {"data": []}))
        client.get_users(["a"])

        fake_time[0] += 200  # past expiry
        queue_token(fake_session, expires_in=100)
        fake_session.get_responses.append(FakeResponse(200, {"data": []}))
        client.get_users(["b"])

        assert len(fake_session.post_calls) == 2

    def test_raises_on_token_failure(self, client, fake_session):
        fake_session.post_responses.append(FakeResponse(401, {}, text="invalid client"))
        with pytest.raises(TwitchAPIError):
            client.get_users(["a"])


class TestEndpoints:
    def test_get_streams_returns_live_entries(self, client, fake_session):
        queue_token(fake_session)
        fake_session.get_responses.append(FakeResponse(200, {"data": [
            {"user_login": "somebody", "id": "123", "title": "playing games"}
        ]}))

        streams = client.get_streams(["somebody"])
        assert streams[0]["user_login"] == "somebody"

        url, headers, params = fake_session.get_calls[0]
        assert url.endswith("/streams")
        assert ("user_login", "somebody") in params

    def test_get_clips_paginates_until_enough_collected(self, client, fake_session):
        queue_token(fake_session)
        fake_session.get_responses.append(FakeResponse(200, {
            "data": [{"id": "clip1", "view_count": 10}],
            "pagination": {"cursor": "abc"},
        }))
        fake_session.get_responses.append(FakeResponse(200, {
            "data": [{"id": "clip2", "view_count": 5}],
            "pagination": {},
        }))

        clips = client.get_clips(broadcaster_id="42", first=100)

        assert [c["id"] for c in clips] == ["clip1", "clip2"]
        assert len(fake_session.get_calls) == 2
        _, _, second_params = fake_session.get_calls[1]
        assert second_params["after"] == "abc"

    def test_get_clips_stops_once_first_is_reached(self, client, fake_session):
        queue_token(fake_session)
        fake_session.get_responses.append(FakeResponse(200, {
            "data": [{"id": "clip1"}, {"id": "clip2"}],
            "pagination": {"cursor": "more"},
        }))

        clips = client.get_clips(broadcaster_id="42", first=2)

        assert len(clips) == 2
        assert len(fake_session.get_calls) == 1  # never asked for the next page

    def test_raises_on_api_error(self, client, fake_session):
        queue_token(fake_session)
        fake_session.get_responses.append(FakeResponse(500, {}, text="server error"))
        with pytest.raises(TwitchAPIError):
            client.get_streams(["a"])
