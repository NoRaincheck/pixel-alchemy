import random

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_TO_PC = {n: i for i, n in enumerate(NOTE_NAMES)}
# flats
NOTE_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE = [0, 2, 3, 5, 7, 8, 10]
DORIAN = [0, 2, 3, 5, 7, 9, 10]
LYDIAN = [0, 2, 4, 6, 7, 9, 11]

SCALES = {
    "major": MAJOR_SCALE,
    "minor": MINOR_SCALE,
    "dorian": DORIAN,
    "lydian": LYDIAN,
}


def note_to_midi(name: str, octave: int = 4) -> int:
    pc = NOTE_TO_PC[name]
    return 12 * (octave + 1) + pc


def midi_to_note(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def sonic_pi_note(midi: int) -> str:
    # Sonic Pi uses :Cs4 / :Fs4 for sharps (not :C#4 which is invalid Ruby symbol)
    raw = midi_to_note(midi)  # e.g. F#4
    sp = raw.replace("#", "s").replace("b", "b")  # keep flats as e.g. Bb4
    # Sonic Pi symbols are lowercase root, e.g. :fs4 — but uppercase also works; normalize to lower
    # Keep as :c4 style for safety: Sonic Pi accepts both :C4 and :c4
    note = sp[0].lower() + sp[1:]
    return f":{note}"


def scale_pitches(root_pc: int, scale_intervals: list[int], octaves: int = 2, base_octave: int = 4) -> list[int]:
    root_midi = 12 * (base_octave + 1) + root_pc
    out = []
    for o in range(octaves):
        for iv in scale_intervals:
            out.append(root_midi + iv + 12 * o)
    return sorted(out)


def chord_for_degree(root_pc: int, scale_intervals: list[int], degree: int, quality: str = "triad7") -> list[int]:
    """Build chord by stacking thirds from scale degree (1-indexed)."""
    idx = (degree - 1) % 7
    root_midi_pc = (root_pc + scale_intervals[idx]) % 12
    octave = 4
    root_midi = 12 * (octave + 1) + root_midi_pc
    key_center = 12 * (octave + 1) + root_pc
    while root_midi > key_center + 7:
        root_midi -= 12
    while root_midi < key_center - 5:
        root_midi += 12

    def pc_for_degree(d: int) -> int:
        pc_idx = (d - 1) % 7
        return (root_pc + scale_intervals[pc_idx]) % 12

    root_pc_val = pc_for_degree(degree)
    third_pc = pc_for_degree(degree + 2)
    fifth_pc = pc_for_degree(degree + 4)
    seventh_pc = pc_for_degree(degree + 6)

    def semitone_interval(from_pc: int, to_pc: int) -> int:
        return (to_pc - from_pc) % 12

    third_iv = semitone_interval(root_pc_val, third_pc)
    fifth_iv = semitone_interval(root_pc_val, fifth_pc)
    seventh_iv = semitone_interval(root_pc_val, seventh_pc)

    # ensure ascending: if interval wraps past, keep diatonic size (3-4, 6-7, 10-11)
    # modulo already gives 0-11; for close voicing we want fifth > third, seventh > fifth
    if fifth_iv <= third_iv:
        fifth_iv += 12
    if seventh_iv <= fifth_iv:
        # bring seventh just above fifth but keep <12 if possible
        # if it wrapped, add 12 then normalize to 10-11 range
        while seventh_iv <= fifth_iv:
            seventh_iv += 12
        if seventh_iv > 11:
            seventh_iv -= 12
            # if still <= fifth, lift an octave
            if seventh_iv <= fifth_iv:
                seventh_iv += 12

    third = root_midi + third_iv
    fifth = root_midi + fifth_iv
    seventh = root_midi + seventh_iv
    # if seventh jumped octave beyond 11, keep it but ensure reasonable voicing (<12)
    if seventh - root_midi > 11:
        seventh = root_midi + (seventh_iv % 12) + 12

    if quality == "triad":
        return [root_midi, third, fifth]
    if quality == "triad7":
        return [root_midi, third, fifth, seventh]
    return [root_midi, third, fifth]


def progression_chords(key: str, scale: str, degrees: list[int], quality: str = "triad7") -> list[list[int]]:
    root_pc = NOTE_TO_PC[key]
    intervals = SCALES[scale]
    return [chord_for_degree(root_pc, intervals, d, quality) for d in degrees]


def pick_key_scale(rng: random.Random) -> tuple[str, str]:
    keys = ["C", "G", "D", "A", "E", "F", "Bb", "Eb", "Ab"]
    scales = ["major", "minor", "dorian", "lydian"]
    weights = [4, 3, 1, 1]  # bias major/minor
    scale = rng.choices(scales, weights=weights, k=1)[0]
    key = rng.choice(keys)
    return key, scale
