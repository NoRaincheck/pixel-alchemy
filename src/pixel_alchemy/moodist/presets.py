"""Port of `moodist/src/stores/preset.ts` + curated builtin presets matching the frontend.

Builtin presets are a human-readable Enum (category) where each preset is a
mood that maps to a pool of sounds from the vendored frontend. Given a
``seed`` the same preset deterministically yields the same mix, while
different seeds give variations — useful for generating multiple versions.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from enum import StrEnum


@dataclass
class Preset:
    id: str
    label: str
    sounds: dict[str, float]


class PresetStore:
    def __init__(self) -> None:
        self.presets: list[Preset] = []

    def add(self, label: str, sounds: dict[str, float]) -> Preset:
        p = Preset(id=str(uuid.uuid4()), label=label, sounds=dict(sounds))
        self.presets.insert(0, p)
        return p

    def rename(self, pid: str, new_name: str) -> None:
        for p in self.presets:
            if p.id == pid:
                p.label = new_name
                return
        raise KeyError(pid)

    def delete(self, pid: str) -> None:
        self.presets = [p for p in self.presets if p.id != pid]

    def get(self, pid: str) -> Preset | None:
        for p in self.presets:
            if p.id == pid:
                return p
        return None


# ------------------------------------------------------------------ builtin presets (match frontend categories/moods)


class MoodPreset(StrEnum):
    """Human-readable preset categories matching the Moodist frontend.

    Each value is the display label; the enum name is the code.
    """

    RAIN = "Rainy Day"
    STORM = "Thunderstorm"
    FOREST = "Forest"
    OCEAN = "Ocean Waves"
    CAFE = "Cafe"
    CITY = "City Life"
    NIGHT = "Night Time"
    WINTER = "Winter"
    FOCUS = "Deep Focus"
    MEDITATION = "Meditation"
    FIREPLACE = "Fireplace"
    WHITE_NOISE = "White Noise"


# Pools mirror the frontend's categories (moodist/src/data/sounds/*).
# Each preset draws from a curated subset so variations stay on-theme.
_PRESET_POOLS: dict[MoodPreset, list[str]] = {
    MoodPreset.RAIN: [
        "light-rain",
        "heavy-rain",
        "rain-on-window",
        "rain-on-car-roof",
        "rain-on-umbrella",
        "rain-on-tent",
        "rain-on-leaves",
        "droplets",
        "river",
        "thunder",
    ],
    MoodPreset.STORM: [
        "heavy-rain",
        "thunder",
        "rain-on-window",
        "howling-wind",
        "wind",
        "wind-in-trees",
        "waterfall",
    ],
    MoodPreset.FOREST: [
        "river",
        "wind-in-trees",
        "jungle",
        "birds",
        "crickets",
        "owl",
        "frog",
        "woodpecker",
        "wind",
        "droplets",
        "walk-on-leaves",
    ],
    MoodPreset.OCEAN: [
        "waves",
        "wind",
        "seagulls",
        "whale",
        "sailboat",
        "rowing-boat",
        "river",
        "waterfall",
        "droplets",
    ],
    MoodPreset.CAFE: [
        "cafe",
        "restaurant",
        "library",
        "office",
        "paper",
        "keyboard",
        "typewriter",
        "clock",
        "crowd",
        "bubbles",
    ],
    MoodPreset.CITY: [
        "traffic",
        "busy-street",
        "highway",
        "road",
        "crowd",
        "subway-station",
        "construction-site",
        "airport",
        "ambulance-siren",
        "fireworks",
    ],
    MoodPreset.NIGHT: [
        "crickets",
        "owl",
        "wolf",
        "howling-wind",
        "wind",
        "night-village",
        "crowded-bar",
        "clock",
        "frog",
        "crows",
    ],
    MoodPreset.WINTER: [
        "walk-in-snow",
        "walk-on-gravel",
        "wind",
        "howling-wind",
        "wind-in-trees",
        "river",
        "campfire",
    ],
    MoodPreset.FOCUS: [
        "brown-noise",
        "pink-noise",
        "white-noise",
        "light-rain",
        "rain-on-window",
        "keyboard",
        "typewriter",
        "clock",
        "ceiling-fan",
        "paper",
    ],
    MoodPreset.MEDITATION: [
        "singing-bowl",
        "wind-chimes",
        "bubbles",
        "droplets",
        "campfire",
        "binaural-alpha",
        "binaural-theta",
        "binaural-delta",
        "temple",
        "church",
        "jungle",
    ],
    MoodPreset.FIREPLACE: [
        "campfire",
        "wind",
        "crickets",
        "owl",
        "woodpecker",
        "rain-on-window",
        "clock",
        "wind-in-trees",
    ],
    MoodPreset.WHITE_NOISE: [
        "white-noise",
        "pink-noise",
        "brown-noise",
        "binaural-alpha",
        "binaural-beta",
        "binaural-theta",
        "binaural-gamma",
        "binaural-delta",
    ],
}


def preset_pool(preset: MoodPreset) -> list[str]:
    return list(_PRESET_POOLS[preset])


def preset_mix(
    preset: MoodPreset,
    seed: int | None = None,
    *,
    count: int | tuple[int, int] = (3, 5),
    volume_range: tuple[float, float] = (0.25, 0.85),
) -> dict[str, float]:
    """Generate a seeded variation of ``preset``.

    Args:
        preset: human-readable enum category.
        seed: if given, deterministic output; different seeds give variations.
        count: how many sounds to pick (int) or range (min, max) inclusive.
        volume_range: (min, max) volume for each sound.

    Returns:
        Mapping sound_id -> volume, ready for ``mix()`` or ``SoundStore.override()``.
    """
    pool = _PRESET_POOLS[preset]
    rng = random.Random(seed)

    if isinstance(count, tuple):
        lo, hi = count
        n = rng.randint(lo, hi)
    else:
        n = count
    n = max(1, min(n, len(pool)))

    chosen = rng.sample(pool, n)
    lo_v, hi_v = volume_range
    return {sid: rng.uniform(lo_v, hi_v) for sid in chosen}


def all_presets() -> list[MoodPreset]:
    return list(MoodPreset)
