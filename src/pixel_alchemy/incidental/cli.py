import argparse
import pathlib
import random

from .generator import TEXTURE_SNIPPETS, generate_piece, to_midi_like, to_sonic_pi
from .progressions import PROGRESSIONS
from .references import list_references
from .render import render_wav


def main():
    p = argparse.ArgumentParser(description="Generate ambient incidental music (Sonic Pi / WAV)")
    p.add_argument("--seed", type=int, default=None, help="random seed (int)")
    p.add_argument("--key", type=str, default=None, help="key e.g. C, G, Bb")
    p.add_argument("--scale", type=str, default=None, choices=["major", "minor", "dorian", "lydian"])
    p.add_argument("--progression", type=str, default=None, help="progression name (see --list)")
    p.add_argument("--bpm", type=int, default=None, help="BPM 30-80, default random 42-62")
    p.add_argument("--bars", type=int, default=16, help="number of bars (4 beats each)")
    p.add_argument("--instrument", type=str, default="piano", choices=["piano", "guitar", "pad"])
    p.add_argument("--texture", type=str, default=None, choices=sorted(TEXTURE_SNIPPETS.keys()), help="optional texture layer from reference examples")
    p.add_argument("-o", "--out", type=str, default=None, help="output .rb or .wav path (ext decides)")
    p.add_argument("--list", action="store_true", help="list available progressions and exit")
    p.add_argument("--list-references", action="store_true", help="list reference .rb examples and exit")
    p.add_argument("--wav", action="store_true", help="also render .wav alongside .rb")
    p.add_argument("--duration", type=str, default=None, help="alias for bars via seconds, e.g. 2m (overrides --bars)")
    args = p.parse_args()

    if args.list:
        for prog in PROGRESSIONS:
            print(f"{prog['name']:22s}  {prog['label']:40s}  {prog['degrees']}")
        return
    if args.list_references:
        for name in list_references():
            print(name)
        return

    seed = args.seed if args.seed is not None else random.randint(0, 999999)
    bars = args.bars
    if args.duration:
        # parse like 30s, 2m, 1h
        d = args.duration.strip()
        if d.endswith("s"):
            sec = int(d[:-1])
        elif d.endswith("m"):
            sec = int(d[:-1]) * 60
        elif d.endswith("h"):
            sec = int(d[:-1]) * 3600
        else:
            sec = int(d)
        # estimate bars from bpm guess ~52
        bpm_guess = args.bpm or 52
        bars = max(4, round(sec / (60 / bpm_guess * 4)))

    piece = generate_piece(
        seed=seed, key=args.key, scale=args.scale, progression=args.progression, bpm=args.bpm, bars=bars, instrument=args.instrument, texture=args.texture
    )
    code = to_sonic_pi(piece)

    # decide output
    if args.out:
        out = pathlib.Path(args.out)
        if out.suffix == ".wav":
            events = to_midi_like(piece)
            render_wav(events, piece["bpm"], str(out))
            # also write rb alongside if --wav not needed (wav is primary)
            rb = out.with_suffix(".rb")
            rb.write_text(code)
            print(f"Wrote {rb} and {out}  ({piece['key']} {piece['scale']} {piece['progression']['label']} {piece['bpm']}bpm {piece['arp_pattern']}) seed={seed}")
        else:
            if not out.suffix:
                out = out.with_suffix(".rb")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(code)
            print(f"Wrote {out}  ({piece['key']} {piece['scale']} {piece['progression']['label']} {piece['bpm']}bpm {piece['arp_pattern']}) seed={seed}")
            if args.wav:
                wav = out.with_suffix(".wav")
                events = to_midi_like(piece)
                render_wav(events, piece["bpm"], str(wav))
                print(f"Wrote {wav}")
    else:
        # print to stdout
        print(code)


if __name__ == "__main__":
    main()
