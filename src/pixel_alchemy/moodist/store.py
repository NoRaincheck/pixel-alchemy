"""Port of `moodist/src/stores/sound.ts` Zustand store to Python."""

from __future__ import annotations

import copy
from dataclasses import dataclass

from .catalog import sound_ids
from .utils import pick_many
from .utils import random as rand_float


@dataclass
class SoundValue:
    is_favorite: bool = False
    is_selected: bool = False
    volume: float = 0.5


class SoundStore:
    """Mirrors Moodist SoundStore API (without persistence)."""

    def __init__(self) -> None:
        self.sounds: dict[str, SoundValue] = {sid: SoundValue() for sid in sound_ids()}
        self.is_playing: bool = False
        self.locked: bool = False
        self.history: dict[str, SoundValue] | None = None

    # queries
    def get_favorites(self) -> list[str]:
        return [sid for sid, v in self.sounds.items() if v.is_favorite]

    def no_selected(self) -> bool:
        return all(not v.is_selected for v in self.sounds.values())

    # actions
    def lock(self) -> None:
        self.locked = True

    def unlock(self) -> None:
        self.locked = False

    def play(self) -> None:
        self.is_playing = True

    def pause(self) -> None:
        self.is_playing = False

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing

    def select(self, sid: str) -> None:
        self._require(sid)
        self.history = None
        self.sounds[sid].is_selected = True

    def unselect(self, sid: str) -> None:
        self._require(sid)
        self.sounds[sid].is_selected = False

    def unselect_all(self, push_to_history: bool = False) -> None:
        if self.no_selected():
            return
        if push_to_history:
            self.history = copy.deepcopy(self.sounds)
        for v in self.sounds.values():
            v.is_selected = False
            v.volume = 0.5
        # keep history if pushed

    def restore_history(self) -> None:
        if self.history is None:
            return
        self.sounds = self.history
        self.history = None

    def set_volume(self, sid: str, volume: float) -> None:
        self._require(sid)
        self.sounds[sid].volume = max(0.0, min(1.0, volume))

    def toggle_favorite(self, sid: str) -> None:
        self._require(sid)
        self.history = None
        self.sounds[sid].is_favorite = not self.sounds[sid].is_favorite

    def override(self, sounds: dict[str, float]) -> None:
        self.unselect_all()
        for sid, vol in sounds.items():
            if sid in self.sounds:
                self.sounds[sid].is_selected = True
                self.sounds[sid].volume = max(0.0, min(1.0, vol))
        self.history = None

    def shuffle(self) -> None:
        for v in self.sounds.values():
            v.is_selected = False
            v.volume = 0.5
        for sid in pick_many(list(self.sounds.keys()), 4):
            self.sounds[sid].is_selected = True
            self.sounds[sid].volume = rand_float(0.2, 1.0)
        self.history = None
        self.is_playing = True

    # helpers
    def selected(self) -> dict[str, float]:
        return {sid: v.volume for sid, v in self.sounds.items() if v.is_selected}

    def to_share(self) -> dict[str, float]:
        return self.selected()

    def _require(self, sid: str) -> None:
        if sid not in self.sounds:
            raise KeyError(f"unknown sound id: {sid}")
