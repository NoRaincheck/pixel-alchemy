"""CLI for MusicAgent — sample analysis + Sonic Pi generation via tau-ai.

Examples::

    # analyze a sample and write metadata
    uv run python -m pixel_alchemy.music_agent.cli --input samples/sounds/cafe.ogg --analyze-only -o out.json

    # analyze then generate .rb (requires LMStudio at http://localhost:1234/v1 with qwen3.6-35b)
    uv run python -m pixel_alchemy.music_agent.cli --input samples/sounds/cafe.ogg --genre LoFi --duration 120 -o cafe.rb

    # generate without audio input (just genre/duration)
    uv run python -m pixel_alchemy.music_agent.cli --genre House --duration 60 -o house.rb

    # also validate via Sonic Pi OSC
    uv run python -m pixel_alchemy.music_agent.cli --input in.wav -o out.rb --validate

    # download ONNX YAMNet (TF-free)
    uv run python -m pixel_alchemy.music_agent.cli --download-yamnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="MusicAgent — sample -> metadata -> Sonic Pi (tau-ai)")
    p.add_argument("--input", "-i", type=str, default=None, help="input audio file or directory")
    p.add_argument("--genre", type=str, default="LoFi", help="genre (LoFi, House, Trance, etc.)")
    p.add_argument("--duration", type=str, default="120", help="duration seconds or 30s/2m/1h")
    p.add_argument("--prompt", type=str, default="", help="extra prompt text")
    p.add_argument("--song-name", type=str, default="generated", help="song name for output dir")
    p.add_argument("-o", "--out", type=str, default=None, help="output .rb or .json path")
    p.add_argument("--analyze-only", action="store_true", help="only analyze, don't generate")
    p.add_argument("--samples-pool", type=str, default=None, help="samples pool dir or metadata json")
    p.add_argument("--model", type=str, default=None, help="model name (default qwen3.6-35b-a3b-mtp)")
    p.add_argument("--base-url", type=str, default=None, help="LMStudio base URL (default http://localhost:1234/v1)")
    p.add_argument("--validate", action="store_true", help="validate .rb via Sonic Pi OSC after generation")
    p.add_argument("--sonic-pi-host", type=str, default="localhost")
    p.add_argument("--sonic-pi-port", type=int, default=4557)
    p.add_argument("--use-yamnet", action="store_true", help="use ONNX YAMNet if available (default off)")
    p.add_argument("--download-yamnet", action="store_true", help="download ONNX YAMNet and exit")
    p.add_argument("--list-phases", action="store_true", help="list phases and exit")
    p.add_argument("--fallback", action="store_true", help="force deterministic fallback (no LLM, instant)")
    args = p.parse_args()

    if args.download_yamnet:
        from .analyze import download_yamnet_onnx

        out = download_yamnet_onnx()
        print(f"Downloaded YAMNet ONNX to {out}")
        return

    if args.list_phases:
        from .config import load_json_config

        phases = load_json_config("phases.json")
        for name, cfg in phases.items():
            print(f"{name:22s}  type={cfg.get('type',''):12s} role={cfg.get('assistant_role_name','')}")
        return

    # parse duration
    d = args.duration.strip()
    try:
        if d.endswith("s"):
            duration = int(d[:-1])
        elif d.endswith("m"):
            duration = int(d[:-1]) * 60
        elif d.endswith("h"):
            duration = int(d[:-1]) * 3600
        else:
            duration = int(d)
    except ValueError:
        print(f"bad --duration {args.duration!r}", file=sys.stderr)
        sys.exit(1)

    meta = None
    if args.input:
        ip = Path(args.input)
        if ip.is_dir():
            from .analyze import analyze_directory

            # analyze directory and optionally write json
            out_json = args.out if (args.analyze_only and args.out and args.out.endswith(".json")) else None
            metas = analyze_directory(ip, out_json=out_json, use_yamnet=args.use_yamnet)
            if args.analyze_only:
                if not out_json:
                    print(json.dumps(metas, indent=2, ensure_ascii=False))
                print(f"Analyzed {len(metas)} files")
                return
            # pick first file as reference for generation
            if metas:
                meta = metas[0]
        elif ip.is_file():
            from .analyze import analyze_sample

            meta = analyze_sample(ip, use_yamnet=args.use_yamnet)
            if "Error" in meta:
                print(f"analyze error: {meta['Error']}", file=sys.stderr)
                sys.exit(1)
            print(f"Analyzed: {meta['Filename']}  {meta['Key']}  {meta['BPM']} BPM  {meta['Vibe']}")
            if args.analyze_only:
                out = Path(args.out) if args.out else None
                if out:
                    out.write_text(json.dumps([meta], indent=2, ensure_ascii=False), encoding="utf-8")
                    print(f"Wrote {out}")
                else:
                    print(json.dumps(meta, indent=2, ensure_ascii=False))
                return
        else:
            print(f"input not found: {args.input}", file=sys.stderr)
            sys.exit(1)
    else:
        if args.analyze_only:
            print("--analyze-only requires --input", file=sys.stderr)
            sys.exit(1)

    # generate
    from .agent import MusicAgent

    agent = MusicAgent(
        model=args.model,
        base_url=args.base_url,
        song_name=args.song_name,
    )
    pool = args.samples_pool or ("samples/sounds" if Path("samples/sounds").exists() else None)
    if args.fallback:
        # offline deterministic path
        code = agent._fallback_generate(args.genre, duration)
        agent.data.set_parameter("sonicpi_code", code)
        from .song import Song as _Song

        _Song(args.song_name).create_song_file(agent.data)
    else:
        code = agent.generate(
            genre=args.genre,
            duration=duration,
            prompt_extra=args.prompt,
            audio_meta=meta,
            samples_pool=pool,
        )

    # decide output path
    if args.out:
        out = Path(args.out)
        if out.suffix == ".json":
            # user asked for json but we generated rb — write both?
            out.write_text(json.dumps({"sonicpi_code": code, "meta": meta}, indent=2), encoding="utf-8")
            rb = out.with_suffix(".rb")
            rb.write_text(code, encoding="utf-8")
            print(f"Wrote {out} and {rb}")
        else:
            if not out.suffix:
                out = out.with_suffix(".rb")
            # also go through Song for versioning/header
            from .song import Song

            Song(args.song_name).create_song_file(agent.data)
            # also write to requested path
            if out != Path(f"songs/{args.song_name}/{args.song_name}.rb"):
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(code, encoding="utf-8")
            print(f"Wrote {out}  ({agent.data.data.get_parameter('theme') if hasattr(agent.data,'data') else ''} {duration}s)")
    else:
        print(code)

    if args.validate:
        from .sonic_pi import validate_sonic_pi_file

        rb_path = Path(args.out) if args.out and Path(args.out).suffix == ".rb" else Path(f"songs/{args.song_name}/{args.song_name}.rb")
        ok, msg = validate_sonic_pi_file(rb_path, host=args.sonic_pi_host, port=args.sonic_pi_port)
        print(f"[validate] {'OK' if ok else 'FAIL'}: {msg}")
        if not ok:
            sys.exit(2)


if __name__ == "__main__":
    main()
