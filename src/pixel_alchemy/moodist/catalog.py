"""Moodist catalog — port of vendored `moodist/src/data/sounds/*`.

Resolves `src` to ``samples/sounds/<category>/<id>.{mp3,wav}`` instead of
``/sounds/...`` asset paths. Keeps Moodist IDs/labels/categories verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ------------------------------------------------------------------ paths


def _samples_root() -> Path:
    # src/pixel_alchemy/moodist/catalog.py -> parents[3] = repo root
    cand = Path(__file__).resolve().parents[3] / "samples" / "sounds"
    if cand.exists():
        return cand
    # fallback: cwd
    cand2 = Path.cwd() / "samples" / "sounds"
    if cand2.exists():
        return cand2
    return cand


SAMPLES_ROOT = _samples_root()


def _src(category: str, filename: str) -> Path:
    return SAMPLES_ROOT / category / filename


# ------------------------------------------------------------------ models


@dataclass(frozen=True)
class Sound:
    id: str
    label: str
    src: Path
    category: str


@dataclass(frozen=True)
class Category:
    id: str
    title: str
    sounds: list[Sound]


# ------------------------------------------------------------------ catalog (verbatim Moodist data)


def _cat(cid: str, title: str, items: list[tuple[str, str, str]]) -> Category:
    return Category(
        id=cid,
        title=title,
        sounds=[Sound(id=sid, label=label, src=_src(cid, fname), category=cid) for sid, label, fname in items],
    )


CATEGORIES: list[Category] = [
    _cat(
        "nature",
        "Nature",
        [
            ("river", "River", "river.mp3"),
            ("waves", "Waves", "waves.mp3"),
            ("campfire", "Campfire", "campfire.mp3"),
            ("wind", "Wind", "wind.mp3"),
            ("howling-wind", "Howling Wind", "howling-wind.mp3"),
            ("wind-in-trees", "Wind in Trees", "wind-in-trees.mp3"),
            ("waterfall", "Waterfall", "waterfall.mp3"),
            ("walk-in-snow", "Walk in Snow", "walk-in-snow.mp3"),
            ("walk-on-leaves", "Walk on Leaves", "walk-on-leaves.mp3"),
            ("walk-on-gravel", "Walk on Gravel", "walk-on-gravel.mp3"),
            ("droplets", "Droplets", "droplets.mp3"),
            ("jungle", "Jungle", "jungle.mp3"),
        ],
    ),
    _cat(
        "rain",
        "Rain",
        [
            ("light-rain", "Light Rain", "light-rain.mp3"),
            ("heavy-rain", "Heavy Rain", "heavy-rain.mp3"),
            ("thunder", "Thunder", "thunder.mp3"),
            ("rain-on-window", "Rain on Window", "rain-on-window.mp3"),
            ("rain-on-car-roof", "Rain on Car Roof", "rain-on-car-roof.mp3"),
            ("rain-on-umbrella", "Rain on Umbrella", "rain-on-umbrella.mp3"),
            ("rain-on-tent", "Rain on Tent", "rain-on-tent.mp3"),
            ("rain-on-leaves", "Rain on Leaves", "rain-on-leaves.mp3"),
        ],
    ),
    _cat(
        "animals",
        "Animals",
        [
            ("birds", "Birds", "birds.mp3"),
            ("seagulls", "Seagulls", "seagulls.mp3"),
            ("crickets", "Crickets", "crickets.mp3"),
            ("wolf", "Wolf", "wolf.mp3"),
            ("owl", "Owl", "owl.mp3"),
            ("frog", "Frog", "frog.mp3"),
            ("dog-barking", "Dog Barking", "dog-barking.mp3"),
            ("horse-gallop", "Horse Gallop", "horse-gallop.mp3"),
            ("cat-purring", "Cat Purring", "cat-purring.mp3"),
            ("crows", "Crows", "crows.mp3"),
            ("whale", "Whale", "whale.mp3"),
            ("beehive", "Beehive", "beehive.mp3"),
            ("woodpecker", "Woodpecker", "woodpecker.mp3"),
            ("chickens", "Chickens", "chickens.mp3"),
            ("cows", "Cows", "cows.mp3"),
            ("sheep", "Sheep", "sheep.mp3"),
        ],
    ),
    _cat(
        "urban",
        "Urban",
        [
            ("highway", "Highway", "highway.mp3"),
            ("road", "Road", "road.mp3"),
            ("ambulance-siren", "Ambulance Siren", "ambulance-siren.mp3"),
            ("busy-street", "Busy Street", "busy-street.mp3"),
            ("crowd", "Crowd", "crowd.mp3"),
            ("traffic", "Traffic", "traffic.mp3"),
            ("fireworks", "Fireworks", "fireworks.mp3"),
        ],
    ),
    _cat(
        "places",
        "Places",
        [
            ("cafe", "Cafe", "cafe.mp3"),
            ("airport", "Airport", "airport.mp3"),
            ("church", "Church", "church.mp3"),
            ("temple", "Temple", "temple.mp3"),
            ("construction-site", "Construction Site", "construction-site.mp3"),
            ("underwater", "Underwater", "underwater.mp3"),
            ("crowded-bar", "Crowded Bar", "crowded-bar.mp3"),
            ("night-village", "Night Village", "night-village.mp3"),
            ("subway-station", "Subway Station", "subway-station.mp3"),
            ("office", "Office", "office.mp3"),
            ("supermarket", "Supermarket", "supermarket.mp3"),
            ("carousel", "Carousel", "carousel.mp3"),
            ("laboratory", "Laboratory", "laboratory.mp3"),
            ("laundry-room", "Laundry Room", "laundry-room.mp3"),
            ("restaurant", "Restaurant", "restaurant.mp3"),
            ("library", "Library", "library.mp3"),
        ],
    ),
    _cat(
        "transport",
        "Transport",
        [
            ("train", "Train", "train.mp3"),
            ("inside-a-train", "Inside a Train", "inside-a-train.mp3"),
            ("airplane", "Airplane", "airplane.mp3"),
            ("submarine", "Submarine", "submarine.mp3"),
            ("sailboat", "Sailboat", "sailboat.mp3"),
            ("rowing-boat", "Rowing Boat", "rowing-boat.mp3"),
        ],
    ),
    _cat(
        "things",
        "Things",
        [
            ("keyboard", "Keyboard", "keyboard.mp3"),
            ("typewriter", "Typewriter", "typewriter.mp3"),
            ("paper", "Paper", "paper.mp3"),
            ("clock", "Clock", "clock.mp3"),
            ("wind-chimes", "Wind Chimes", "wind-chimes.mp3"),
            ("singing-bowl", "Singing Bowl", "singing-bowl.mp3"),
            ("ceiling-fan", "Ceiling Fan", "ceiling-fan.mp3"),
            ("dryer", "Dryer", "dryer.mp3"),
            ("slide-projector", "Slide Projector", "slide-projector.mp3"),
            ("boiling-water", "Boiling Water", "boiling-water.mp3"),
            ("bubbles", "Bubbles", "bubbles.mp3"),
            ("tuning-radio", "Tuning Radio", "tuning-radio.mp3"),
            ("morse-code", "Morse Code", "morse-code.mp3"),
            ("washing-machine", "Washing Machine", "washing-machine.mp3"),
            ("vinyl-effect", "Vinyl Effect", "vinyl-effect.mp3"),
            ("windshield-wipers", "Windshield Wipers", "windshield-wipers.mp3"),
        ],
    ),
    _cat(
        "noise",
        "Noise",
        [
            ("white-noise", "White Noise", "white-noise.wav"),
            ("pink-noise", "Pink Noise", "pink-noise.wav"),
            ("brown-noise", "Brown Noise", "brown-noise.wav"),
        ],
    ),
    _cat(
        "binaural",
        "Binaural Beats",
        [
            ("binaural-delta", "Delta", "binaural-delta.wav"),
            ("binaural-theta", "Theta", "binaural-theta.wav"),
            ("binaural-alpha", "Alpha", "binaural-alpha.wav"),
            ("binaural-beta", "Beta", "binaural-beta.wav"),
            ("binaural-gamma", "Gamma", "binaural-gamma.wav"),
        ],
    ),
]

# extras not in Moodist categories but present in samples/sounds
_EXTRAS: list[Sound] = [
    Sound(id="silence", label="Silence", src=SAMPLES_ROOT / "silence.wav", category="__extra__"),
    Sound(id="alarm", label="Alarm", src=SAMPLES_ROOT / "alarm.mp3", category="__extra__"),
]

# ------------------------------------------------------------------ indexes

_ID_TO_SOUND: dict[str, Sound] = {s.id: s for c in CATEGORIES for s in c.sounds}
_ID_TO_SOUND.update({s.id: s for s in _EXTRAS})


def all_sounds() -> list[Sound]:
    return [s for c in CATEGORIES for s in c.sounds] + _EXTRAS


def get_sound(sound_id: str) -> Sound | None:
    return _ID_TO_SOUND.get(sound_id)


def get_category(cat_id: str) -> Category | None:
    for c in CATEGORIES:
        if c.id == cat_id:
            return c
    return None


def sound_ids() -> list[str]:
    return list(_ID_TO_SOUND.keys())


def validate() -> list[str]:
    """Return list of missing files (empty if all present)."""
    return [str(s.src) for s in all_sounds() if not s.src.exists()]
