---
name: soundscape
description: Generate ambient soundscapes of exact duration by layering moodist sounds with preset mixes and effects. Use when creating cafe, city-walk, office, focus, or custom ambient audio.
---

# Soundscape

Layer multiple ambient sounds from `moodist/public/sounds` into a single audio file of precise length. Loops short sources, mixes with per-layer volume, adds fades and optional effects (reverb, lowpass, highpass). Presets bundle curated mixes for common moods.

Requires `ffmpeg` on PATH (`brew install ffmpeg`). Sources are 84 sounds across 8 categories: nature, rain, animals, urban, places, transport, things, noise.

## Usage

```bash
# list available presets and sounds
./soundscape.py --list-presets
./soundscape.py --list-sounds

# preset to exact duration (seconds or 30s / 5m / 1h)
./soundscape.py --preset cafe --duration 10m -o cafe_10m.mp3
./soundscape.py --preset city-walk --duration 600 -o city.mp3
./soundscape.py --preset office --duration 1h -o office_hour.mp3

# custom mix: id:volume pairs (id matches moodist sound id, with or without category/)
./soundscape.py --sounds "cafe:0.8,keyboard:0.4,light-rain:0.5" --duration 5m -o custom.mp3
./soundscape.py --sounds "places/cafe:0.7,things/keyboard:0.3" --duration 300 -o custom.mp3

# combine preset + extra layers / overrides
./soundscape.py --preset rainy-night --sounds "thunder:0.2" --duration 30m -o rainy.mp3

# effects
./soundscape.py --preset cafe --duration 5m --fade 3 --lowpass 3500 -o cafe_warm.mp3
./soundscape.py --preset fireplace --duration 10m --reverb --highpass 80 -o fire.mp3
./soundscape.py --preset deep-focus --duration 1h --normalize -o focus.mp3

# dry run (print ffmpeg command without running)
./soundscape.py --preset cafe --duration 10 -o out.mp3 --dry-run
```

## Presets

| preset | mood | layers (id:vol) | effect |
|---|---|---|---|
| `cafe` | busy cafe, chatter + kitchen | cafe 0.85, keyboard 0.35, dishes/bubbles 0.2, crowd 0.15 | warm lowpass |
| `city-walk` | walking in the city | busy-street 0.7, traffic 0.5, crowd 0.4, wind 0.25 | dry/outdoor |
| `office` | working in the office | office 0.8, keyboard 0.5, paper 0.3, ceiling-fan 0.25, clock 0.15 | dry |
| `library` | quiet study | library 0.8, paper 0.35, clock 0.2, keyboard 0.25 | muffled low |
| `rainy-night` | rain + night calm | light-rain 0.7, rain-on-window 0.4, thunder 0.2, wind-in-trees 0.35, crickets 0.3 | warm |
| `fireplace` | cabin / hygge | campfire 0.8, wind 0.3, rain-on-tent 0.25, owl 0.15 | warm reverb |
| `deep-focus` | brown noise focus | brown-noise 0.6, light-rain 0.3, keyboard 0.15 | lowpass |
| `night-village` | village evening | night-village 0.8, crickets 0.4, wind 0.3, church 0.2 | spacious |
| `construction` | urban work site | construction-site 0.7, traffic 0.4, busy-street 0.3 | dry |
| `underwater` | submerged dream | underwater 0.8, whale 0.4, waves 0.3, droplets 0.2 | lowpass + reverb |
| `train-ride` | inside a moving train | inside-a-train 0.8, rain-on-window 0.3, clock 0.15 | bandpass |
| `jungle` | dense forest | jungle 0.8, birds 0.5, droplets 0.3, wind 0.2 | lush |

Use `--list-presets` for full layer details.

## Options

| option | default | meaning |
|---|---|---|
| `--preset NAME` | — | preset name (see table) |
| `--sounds SPEC` | — | comma list `id:vol` (vol 0–1, default 0.5 if omitted) |
| `--duration DUR` | `5m` | target length: seconds int or `30s`/`5m`/`1h`/`1h30m` |
| `-o OUT` | `<preset_or_mix>.mp3` | output file (ext picks codec: mp3/wav/ogg/m4a/flac) |
| `--fade SEC` | `3` | fade in/out seconds (0 to disable) |
| `--reverb` | off | light room reverb (aecho) |
| `--lowpass FREQ` | preset default | lowpass Hz (e.g. 3000 for warmth) |
| `--highpass FREQ` | off | highpass Hz (e.g. 80 to cut rumble) |
| `--normalize` | off | loudness normalize final mix (loudnorm) |
| `--source DIR` | `moodist/public/sounds` | folder to search for source audio |
| `--list-presets` | — | print presets and exit |
| `--list-sounds` | — | scan source dir and print ids |
| `--dry-run` | off | print ffmpeg cmd and exit |

Source ids match moodist ids: `cafe`, `light-rain`, `brown-noise`, `office`, etc. Category prefix is optional (`places/cafe` == `cafe`).

Short files are looped (`-stream_loop -1`) to reach `--duration`; output is exactly that long.

## Examples

```bash
# 1 hour cafe for video background
./soundscape.py --preset cafe --duration 1h -o cafe_1h.mp3 --fade 5 --normalize

# 30 min city walk, extra siren layer, highpass to thin it
./soundscape.py --preset city-walk --duration 30m --sounds "ambulance-siren:0.15" --highpass 120 -o walk.mp3

# pure custom: jungle rain loop 10 minutes
./soundscape.py --sounds "jungle:0.7,heavy-rain:0.6,thunder:0.3" --duration 10m -o storm_jungle.mp3

# office focus with brown noise bed
./soundscape.py --sounds "office:0.5,brown-noise:0.4,keyboard:0.3" --duration 45m --lowpass 3000 -o work.mp3
```
