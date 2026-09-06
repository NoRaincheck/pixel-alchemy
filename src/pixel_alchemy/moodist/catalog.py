"""Moodist catalog — port of vendored `moodist/src/data/sounds/*`.

Resolves `src` to ``samples/sounds/<category>/<id>.ogg`` (Opus) instead of
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
    p = SAMPLES_ROOT / category / filename
    if p.exists():
        return p
    # fallback: try other extensions if OGG not yet present (legacy)
    for ext in (".ogg", ".mp3", ".wav"):
        cand = SAMPLES_ROOT / category / (Path(filename).stem + ext)
        if cand.exists():
            return cand
    return p


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
            ("river", "River", "river.ogg"),
            ("waves", "Waves", "waves.ogg"),
            ("campfire", "Campfire", "campfire.ogg"),
            ("wind", "Wind", "wind.ogg"),
            ("howling-wind", "Howling Wind", "howling-wind.ogg"),
            ("wind-in-trees", "Wind in Trees", "wind-in-trees.ogg"),
            ("waterfall", "Waterfall", "waterfall.ogg"),
            ("walk-in-snow", "Walk in Snow", "walk-in-snow.ogg"),
            ("walk-on-leaves", "Walk on Leaves", "walk-on-leaves.ogg"),
            ("walk-on-gravel", "Walk on Gravel", "walk-on-gravel.ogg"),
            ("droplets", "Droplets", "droplets.ogg"),
            ("jungle", "Jungle", "jungle.ogg"),
        ],
    ),
    _cat(
        "rain",
        "Rain",
        [
            ("light-rain", "Light Rain", "light-rain.ogg"),
            ("heavy-rain", "Heavy Rain", "heavy-rain.ogg"),
            ("thunder", "Thunder", "thunder.ogg"),
            ("rain-on-window", "Rain on Window", "rain-on-window.ogg"),
            ("rain-on-car-roof", "Rain on Car Roof", "rain-on-car-roof.ogg"),
            ("rain-on-umbrella", "Rain on Umbrella", "rain-on-umbrella.ogg"),
            ("rain-on-tent", "Rain on Tent", "rain-on-tent.ogg"),
            ("rain-on-leaves", "Rain on Leaves", "rain-on-leaves.ogg"),
        ],
    ),
    _cat(
        "animals",
        "Animals",
        [
            ("birds", "Birds", "birds.ogg"),
            ("seagulls", "Seagulls", "seagulls.ogg"),
            ("crickets", "Crickets", "crickets.ogg"),
            ("wolf", "Wolf", "wolf.ogg"),
            ("owl", "Owl", "owl.ogg"),
            ("frog", "Frog", "frog.ogg"),
            ("dog-barking", "Dog Barking", "dog-barking.ogg"),
            ("horse-gallop", "Horse Gallop", "horse-gallop.ogg"),
            ("cat-purring", "Cat Purring", "cat-purring.ogg"),
            ("crows", "Crows", "crows.ogg"),
            ("whale", "Whale", "whale.ogg"),
            ("beehive", "Beehive", "beehive.ogg"),
            ("woodpecker", "Woodpecker", "woodpecker.ogg"),
            ("chickens", "Chickens", "chickens.ogg"),
            ("cows", "Cows", "cows.ogg"),
            ("sheep", "Sheep", "sheep.ogg"),
        ],
    ),
    _cat(
        "urban",
        "Urban",
        [
            ("highway", "Highway", "highway.ogg"),
            ("road", "Road", "road.ogg"),
            ("ambulance-siren", "Ambulance Siren", "ambulance-siren.ogg"),
            ("busy-street", "Busy Street", "busy-street.ogg"),
            ("crowd", "Crowd", "crowd.ogg"),
            ("traffic", "Traffic", "traffic.ogg"),
            ("fireworks", "Fireworks", "fireworks.ogg"),
        ],
    ),
    _cat(
        "places",
        "Places",
        [
            ("cafe", "Cafe", "cafe.ogg"),
            ("airport", "Airport", "airport.ogg"),
            ("church", "Church", "church.ogg"),
            ("temple", "Temple", "temple.ogg"),
            ("construction-site", "Construction Site", "construction-site.ogg"),
            ("underwater", "Underwater", "underwater.ogg"),
            ("crowded-bar", "Crowded Bar", "crowded-bar.ogg"),
            ("night-village", "Night Village", "night-village.ogg"),
            ("subway-station", "Subway Station", "subway-station.ogg"),
            ("office", "Office", "office.ogg"),
            ("supermarket", "Supermarket", "supermarket.ogg"),
            ("carousel", "Carousel", "carousel.ogg"),
            ("laboratory", "Laboratory", "laboratory.ogg"),
            ("laundry-room", "Laundry Room", "laundry-room.ogg"),
            ("restaurant", "Restaurant", "restaurant.ogg"),
            ("library", "Library", "library.ogg"),
        ],
    ),
    _cat(
        "transport",
        "Transport",
        [
            ("train", "Train", "train.ogg"),
            ("inside-a-train", "Inside a Train", "inside-a-train.ogg"),
            ("airplane", "Airplane", "airplane.ogg"),
            ("submarine", "Submarine", "submarine.ogg"),
            ("sailboat", "Sailboat", "sailboat.ogg"),
            ("rowing-boat", "Rowing Boat", "rowing-boat.ogg"),
        ],
    ),
    _cat(
        "things",
        "Things",
        [
            ("keyboard", "Keyboard", "keyboard.ogg"),
            ("typewriter", "Typewriter", "typewriter.ogg"),
            ("paper", "Paper", "paper.ogg"),
            ("clock", "Clock", "clock.ogg"),
            ("wind-chimes", "Wind Chimes", "wind-chimes.ogg"),
            ("singing-bowl", "Singing Bowl", "singing-bowl.ogg"),
            ("ceiling-fan", "Ceiling Fan", "ceiling-fan.ogg"),
            ("dryer", "Dryer", "dryer.ogg"),
            ("slide-projector", "Slide Projector", "slide-projector.ogg"),
            ("boiling-water", "Boiling Water", "boiling-water.ogg"),
            ("bubbles", "Bubbles", "bubbles.ogg"),
            ("tuning-radio", "Tuning Radio", "tuning-radio.ogg"),
            ("morse-code", "Morse Code", "morse-code.ogg"),
            ("washing-machine", "Washing Machine", "washing-machine.ogg"),
            ("vinyl-effect", "Vinyl Effect", "vinyl-effect.ogg"),
            ("windshield-wipers", "Windshield Wipers", "windshield-wipers.ogg"),
        ],
    ),
    _cat(
        "noise",
        "Noise",
        [
            ("white-noise", "White Noise", "white-noise.ogg"),
            ("pink-noise", "Pink Noise", "pink-noise.ogg"),
            ("brown-noise", "Brown Noise", "brown-noise.ogg"),
        ],
    ),
    _cat(
        "binaural",
        "Binaural Beats",
        [
            ("binaural-delta", "Delta", "binaural-delta.ogg"),
            ("binaural-theta", "Theta", "binaural-theta.ogg"),
            ("binaural-alpha", "Alpha", "binaural-alpha.ogg"),
            ("binaural-beta", "Beta", "binaural-beta.ogg"),
            ("binaural-gamma", "Gamma", "binaural-gamma.ogg"),
        ],
    ),
]

# extras not in Moodist categories but present in samples/sounds
_EXTRAS: list[Sound] = [
    Sound(id="silence", label="Silence", src=SAMPLES_ROOT / "silence.ogg", category="__extra__"),
    Sound(id="alarm", label="Alarm", src=SAMPLES_ROOT / "alarm.ogg", category="__extra__"),
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
