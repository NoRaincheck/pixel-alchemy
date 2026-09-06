import random

from .patterns import arpeggiate, elongated_melody
from .progressions import BY_NAME, PROGRESSIONS
from .theory import NOTE_TO_PC, SCALES, pick_key_scale, progression_chords, scale_pitches, sonic_pi_note

PIANO_SYNTHS = [":piano", ":blade", ":hollow", ":dark_ambience", ":pretty_bell"]
GUITAR_SYNTHS = [":pluck", ":pluck", ":hollow", ":sine"]
PAD_SYNTHS = [":hollow", ":dark_ambience", ":blade", ":prophet", ":fm"]

ARPEGGIO_KEYS = ["up", "up_down", "alberti", "converge", "inside_out", "down", "held"]


TEXTURE_SNIPPETS = {
    "ocean": [
        "    # -- texture: ocean (ref ocean.rb) --",
        "    live_loop :oceans do",
        "      s = synth [:bnoise, :cnoise, :gnoise].choose, amp: rrand(0.3, 0.8), attack: rrand(0, 4), sustain: rrand(0, 2), release: rrand(1, 5), cutoff: rrand(60, 100), pan: rrand(-1, 1)",
        "      control s, pan: rrand(-1, 1), cutoff: rrand(60, 110)",
        "      sleep rrand(2, 4)",
        "    end",
    ],
    "haunted": [
        "    # -- texture: haunted windchimes (ref haunted.rb) --",
        "    live_loop :haunted do",
        "      sample :perc_bell, rate: rrand(-1.5, 1.5), amp: 0.35, pan: rrand(-1, 1)",
        "      sleep rrand(1, 4)",
        "    end",
    ],
    "choral": [
        "    # -- texture: choral (ref choral.rb) --",
        "    live_loop :choral do",
        "      r = (ring 0.5, 1.0/3, 3.0/5).choose",
        "      8.times do",
        "        sample :ambi_choir, rate: r, amp: 0.45, pan: rrand(-1, 1)",
        "        sleep 0.5",
        "      end",
        "    end",
    ],
    "ambient": [
        "    # -- texture: Darin Wilson phasing (ref ambient_darin.rb) --",
        "    with_fx :reverb, mix: 0.35 do",
        "      live_loop :shimmer1 do; play choose([:D4,:E4]), attack: 6, release: 6, amp: 0.18; sleep 8; end",
        "      live_loop :shimmer2 do; play choose([:Fs4,:G4]), attack: 4, release: 5, amp: 0.16; sleep 10; end",
        "      live_loop :shimmer3 do; play choose([:A4,:Cs5]), attack: 5, release: 5, amp: 0.15; sleep 11; end",
        "    end",
    ],
}


def generate_piece(
    seed: int | None = None,
    key: str | None = None,
    scale: str | None = None,
    progression: str | None = None,
    bpm: int | None = None,
    bars: int = 16,
    instrument: str = "piano",  # piano|guitar|pad
    texture: str | None = None,
) -> dict:
    rng = random.Random(seed)
    if key is None or scale is None:
        k, s = pick_key_scale(rng)
        key = key or k
        scale = scale or s
    if scale not in SCALES:
        scale = "major"
    if progression is None:
        prog = rng.choice(PROGRESSIONS)
    else:
        prog = BY_NAME.get(progression, rng.choice(PROGRESSIONS))

    degrees: list[int] = prog["degrees"]
    # repeat progression to fill bars
    loop_degrees = (degrees * ((bars // len(degrees)) + 2))[:bars]

    chords = progression_chords(key, scale, loop_degrees, quality="triad")
    # for ambient, 7ths sometimes
    if rng.random() < 0.4:
        chords7 = progression_chords(key, scale, loop_degrees, quality="triad7")
        # mix: 30% of chords become 7th
        chords = [c7 if rng.random() < 0.3 else c for c, c7 in zip(chords, chords7)]

    bpm = bpm or rng.randint(42, 62)
    arp_pattern = rng.choice(ARPEGGIO_KEYS)
    # elongated melody scale pitches
    root_pc = NOTE_TO_PC[key]
    sc_pitches = scale_pitches(root_pc, SCALES[scale], octaves=3, base_octave=3)

    # instrument picks synth
    if instrument == "guitar":
        synth = rng.choice(GUITAR_SYNTHS)
    elif instrument == "pad":
        synth = rng.choice(PAD_SYNTHS)
    else:
        synth = rng.choice(PIANO_SYNTHS)

    # per-bar arpeggio + melody
    bars_data = []
    for chord in chords:
        arp = arpeggiate(chord, pattern=arp_pattern, octave_shift=0, spread=(instrument == "guitar"))
        # each bar is 4 beats; for slow ambient, play arp notes stretched
        mel = elongated_melody(chord, sc_pitches, rng, beats=4.0)
        bars_data.append({"chord": chord, "arp": arp, "melody": mel})

    return {
        "seed": seed,
        "key": key,
        "scale": scale,
        "bpm": bpm,
        "progression": prog,
        "degrees": loop_degrees,
        "instrument": instrument,
        "synth": synth,
        "arp_pattern": arp_pattern,
        "bars_data": bars_data,
        "texture": texture,
    }


def to_sonic_pi(piece: dict) -> str:
    bpm = piece["bpm"]
    synth = piece["synth"]
    arp_pat = piece["arp_pattern"]
    prog_label = piece["progression"]["label"]
    lines: list[str] = []
    lines.append(f"# Ambient incidental — {piece['key']} {piece['scale']} — {prog_label}")
    lines.append(f"# pattern={arp_pat} synth={synth}  seed={piece['seed']}")
    lines.append(f"use_bpm {bpm}")
    lines.append("")
    lines.append("with_fx :reverb, room: 0.85, mix: 0.35 do")
    lines.append("  with_fx :lpf, cutoff: 95 do")
    lines.append(f"    use_synth {synth}")
    lines.append("")

    # define chord progression as midi arrays
    lines.append("    chords = [")
    for bd in piece["bars_data"]:
        notes = ", ".join(sonic_pi_note(n) for n in bd["chord"])
        lines.append(f"      [{notes}],")
    lines.append("    ]")
    lines.append("")

    # arpeggio helper
    lines.append("    # -- elongated arpeggio --")
    lines.append("    live_loop :arp do")
    lines.append("      sync :bar if tick != 0")
    lines.append("      c = chords.tick")
    # map pattern to sonic pi code
    pat_code = {
        "up": "c",
        "down": "c.reverse",
        "up_down": "(c + c.reverse[1...-1])",
        "down_up": "(c.reverse + c[1...-1])",
        "alberti": "[c[0], c[2], c[1], c[2]]",
        "converge": "[c[0], c[-1], c[1], c[-2]]",
        "held": "c",
        "inside_out": "[c[1], c[0], c[2], c[1]]",
    }.get(arp_pat, "c")

    # play style depends on held vs arpeggiated
    if arp_pat == "held":
        lines.append("      # block chord, very slow")
        lines.append("      play_chord c, attack: 2, sustain: 2, release: 3, amp: 0.45")
        lines.append("      sleep 4")
    else:
        lines.append(f"      notes = {pat_code}")
        lines.append("      notes.each do |n|")
        # slow elongated sleeps: each note ~1-1.5 beats
        if piece["instrument"] == "guitar":
            lines.append("        play n, attack: 0.15, sustain: 0.6, release: 1.2, amp: 0.5, cutoff: 90")
        else:
            lines.append("        play n, attack: 0.3, sustain: 0.7, release: 1.6, amp: 0.42")
        lines.append("        sleep [1, 1, 1.5, 2].choose * 0.95")
        lines.append("      end")
        # fill remainder of 4-beat bar if needed
        lines.append("      sleep 0.5")
    lines.append("    end")
    lines.append("")

    # elongated melody
    lines.append("    # -- very slow elongated melody --")
    lines.append("    live_loop :melody do")
    lines.append("      sync :bar if tick != 0")
    lines.append("      c = chords.look")
    lines.append("      # pick chord tone 70% else scale passing tone")
    lines.append("      note = (c + scale(:c4, :" + piece["scale"] + ", num_octaves: 2)).choose")
    lines.append("      # prefer chord tones")
    lines.append("      note = c.choose if one_in(3) == false")
    lines.append("      use_synth :hollow if one_in(4)")
    lines.append("      play note + [0, 12, -12].choose, attack: 2.5, sustain: 3, release: 4, amp: 0.28, cutoff: 85")
    lines.append("      sleep [4, 6, 8, 3].choose")
    lines.append("      # occasional rest")
    lines.append("      sleep 2 if one_in(6)")
    lines.append("    end")
    lines.append("")

    lines.append("    live_loop :bar do; sleep 4; end")
    # optional texture layer guided by reference examples
    texture = piece.get("texture")
    if texture in TEXTURE_SNIPPETS:
        lines.append("")
        lines.extend(TEXTURE_SNIPPETS[texture])
    lines.append("  end")
    lines.append("end")
    lines.append("")
    return "\n".join(lines)


def to_midi_like(piece: dict) -> list[tuple[int, float, float]]:
    """Flatten to (midi, start_beat, duration_beats) for simple wav rendering."""
    events: list[tuple[int, float, float]] = []
    beat = 0.0
    for bd in piece["bars_data"]:
        # arp notes spread across bar
        arp = bd["arp"]
        if piece["arp_pattern"] == "held":
            for n in bd["chord"]:
                events.append((n, beat, 4.0))
        else:
            # estimate arpeggio timing: split 4 beats
            step = 4.0 / max(len(arp), 1)
            for i, n in enumerate(arp):
                # play only first len that fits elongated feel: use 3-4 notes per bar
                if i >= 4:
                    break
                events.append((n, beat + i * step, step * 0.9))
        # melody
        mbeat = beat
        for midi, dur in bd["melody"]:
            if midi is not None:
                events.append((midi + 12, mbeat, dur * 0.92))
            mbeat += dur
        beat += 4.0
    return events
