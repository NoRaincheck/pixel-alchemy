"""Song creation databag — port of App/services/songCreationData.py"""

from __future__ import annotations


class SongCreationData:
    def __init__(self, logger=None):
        self.song_description = ""
        self.theme = ""
        self.melody = ""
        self.rhythm = ""
        self.lyrics = ""
        self.structure = ""
        self.segments: dict | str = {}
        self.total_duration = "120"
        self.arrangements = ""
        self.sonicpi_code = ""
        self.review = ""
        self.album_url = ""
        self.samples = ""
        self.logger = logger

    def set_parameter(self, name: str, value):
        if hasattr(self, name):
            setattr(self, name, value)

    def get_parameter(self, name: str):
        return getattr(self, name, None)

    def update_parameters_from_response(self, data: dict):
        for k, v in data.items():
            if not hasattr(self, k):
                continue
            cur = getattr(self, k)
            if isinstance(v, dict) and isinstance(cur, dict):
                cur.update(v)
            else:
                setattr(self, k, v)

    def as_dict(self) -> dict:
        return {
            k: getattr(self, k)
            for k in [
                "song_description",
                "theme",
                "melody",
                "rhythm",
                "lyrics",
                "structure",
                "segments",
                "total_duration",
                "arrangements",
                "sonicpi_code",
                "review",
                "samples",
            ]
        }
