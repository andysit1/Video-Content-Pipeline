"""
Thin wrapper around the Twitch Helix API. Only the read-only, app-level
endpoints this pipeline needs are covered: resolving a login to a user id,
checking who's currently live, listing a broadcaster's VODs, and listing
clips (Tier 2's audience-driven highlight signal -- a clip's `vod_offset`
is seconds into its VOD, which is exactly what popular_clips_stage.py
needs to locate the moment).

Auth uses the client-credentials ("app access token") flow -- no user
login needed, just a Client ID/Secret from a Twitch developer application
(https://dev.twitch.tv/console/apps).

`session` is injectable so tests exercise the real request-building /
response-parsing logic against a fake transport instead of the network.
"""
import time

import requests


class TwitchAPIError(Exception):
    pass


class TwitchClient:
    TOKEN_URL = "https://id.twitch.tv/oauth2/token"
    API_BASE = "https://api.twitch.tv/helix"

    def __init__(self, client_id, client_secret, session=None):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session or requests.Session()
        self._access_token = None
        self._token_expires_at = 0

    def _ensure_token(self):
        if self._access_token and time.time() < self._token_expires_at:
            return

        resp = self.session.post(self.TOKEN_URL, data={
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "grant_type": "client_credentials",
        })
        if resp.status_code != 200:
            raise TwitchAPIError("token request failed: {} {}".format(resp.status_code, resp.text))

        data = resp.json()
        self._access_token = data["access_token"]
        # refresh a minute early so a call doesn't race an expiring token
        self._token_expires_at = time.time() + data.get("expires_in", 3600) - 60

    def _headers(self):
        self._ensure_token()
        return {
            "Client-Id": self.client_id,
            "Authorization": "Bearer {}".format(self._access_token),
        }

    def _get(self, path, params=None):
        resp = self.session.get(self.API_BASE + path, headers=self._headers(), params=params or {})
        if resp.status_code != 200:
            raise TwitchAPIError("GET {} failed: {} {}".format(path, resp.status_code, resp.text))
        return resp.json()

    def get_users(self, logins):
        """Resolve one or more channel logins to Twitch user objects (includes `id`)."""
        data = self._get("/users", params=[("login", login) for login in logins])
        return data.get("data", [])

    def get_streams(self, user_logins):
        """Returns the stream objects for whichever of these logins are currently live."""
        data = self._get("/streams", params=[("user_login", login) for login in user_logins])
        return data.get("data", [])

    def get_videos(self, user_id, video_type="archive", first=5):
        """Lists a broadcaster's VODs, most recent first."""
        data = self._get("/videos", params={"user_id": user_id, "type": video_type, "first": first})
        return data.get("data", [])

    def get_clips(self, broadcaster_id, started_at=None, ended_at=None, first=100):
        """
        Lists clips for a broadcaster, optionally restricted to a time
        window (e.g. a specific VOD's start/end, RFC3339 timestamps).
        Paginates until `first` clips are collected or the API runs out.
        """
        page_size = min(first, 100)
        params = {"broadcaster_id": broadcaster_id, "first": page_size}
        if started_at:
            params["started_at"] = started_at
        if ended_at:
            params["ended_at"] = ended_at

        clips = []
        cursor = None
        while len(clips) < first:
            if cursor:
                params["after"] = cursor
            data = self._get("/clips", params=params)
            page = data.get("data", [])
            clips.extend(page)
            cursor = data.get("pagination", {}).get("cursor")
            if not cursor or not page:
                break
        return clips[:first]
