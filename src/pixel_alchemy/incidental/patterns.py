import random

ARPEGGIO_PATTERNS = {
    "up": lambda chord: chord,
    "down": lambda chord: list(reversed(chord)),
    "up_down": lambda chord: chord + list(reversed(chord[1:-1])),
    "down_up": lambda chord: list(reversed(chord)) + chord[1:-1],
    "alberti": lambda chord: [chord[0], chord[2], chord[1], chord[2]] if len(chord) >= 3 else chord,
    "converge": lambda chord: [chord[0], chord[-1], chord[1], chord[-2]] if len(chord) >= 4 else chord,
    "diverge": lambda chord: [chord[len(chord)//2], chord[0], chord[-1], chord[1]] if len(chord) >= 4 else chord,
    "held": lambda chord: chord,  # block chord
    "inside_out": lambda chord: [chord[1], chord[0], chord[2], chord[1]] if len(chord) >= 3 else chord,
}

# For ambient: slow, sparse, long notes
RHYTHMS = {
    "elongated": [2.0, 2.0, 4.0, 2.0],  # beats at slow bpm
    "sparse": [4.0, 4.0, 2.0],
    "drone": [8.0],
    "gentle": [1.0, 1.0, 2.0, 2.0],
    "waltz": [1.5, 1.5, 3.0],
}


def arpeggiate(chord: list[int], pattern: str = "up", octave_shift: int = 0, spread: bool = False) -> list[int]:
    func = ARPEGGIO_PATTERNS.get(pattern, ARPEGGIO_PATTERNS["up"])
    notes = func(chord)
    if spread:
        # spread voicing across octaves for guitar/piano openness
        out = []
        for i, n in enumerate(notes):
            off = (i // len(chord)) * 12
            out.append(n + off + octave_shift * 12)
        return out
    return [n + octave_shift * 12 for n in notes]


def elongated_melody(chord: list[int], scale_pitches: list[int], rng: random.Random, beats: float = 8.0) -> list[tuple[int, float]]:
    """Generate very slow elongated melody over one chord.

    Returns list of (midi, duration_beats). Durations 2-8 beats, sparse rests.
    Prefers chord tones, occasional passing tones, wide intervals stretched over time.
    """
    # candidate scale notes near chord octave
    candidates = [p for p in scale_pitches if 48 <= p <= 84]
    if not candidates:
        candidates = chord

    out: list[tuple[int, float]] = []
    remaining = beats
    last: int | None = None
    while remaining > 0.1:
        # pick duration: long
        dur = rng.choice([2.0, 2.0, 3.0, 4.0, 4.0, 6.0, 8.0])
        dur = min(dur, remaining)
        # occasional rest (15%)
        if rng.random() < 0.15 and remaining > 2:
            out.append((None, min(2.0, remaining)))  # type: ignore
            remaining -= 2.0
            continue
        if rng.random() < 0.7:
            note = rng.choice(chord)
            # octave wander +/-1 for elongation
            note += rng.choice([-12, 0, 12, 0, 0])
        else:
            # passing tone from scale, near last note if possible
            if last is not None:
                near = sorted(candidates, key=lambda p: abs(p - last))
                note = rng.choice(near[:4])
            else:
                note = rng.choice(candidates)
        # clamp
        note = max(36, min(84, note))
        # avoid immediate repeat 50% -> shift octave
        if last is not None and note % 12 == last % 12 and rng.random() < 0.5:
            note += rng.choice([-12, 12])
            note = max(36, min(84, note))
        out.append((note, dur))
        last = note
        remaining -= dur
    return out
