---
name: incidental
description: Generate ambient/incidental music via Sonic Pi Ruby (.rb) using known chord progressions, piano/guitar arpeggios at slow BPM, and very slow elongated melodies. Uses FoxDot-inspired theory but emits Sonic Pi DSL. Use when creating background ambient beds.
---

# Incidental

Generate ambient background incidental music as **Sonic Pi Ruby (`.rb`)** files. Sonic Pi scripts are Ruby DSL (`use_synth`, `live_loop`, `play`, `play_chord`, `sample`, `with_fx`, `choose`, `rrand`) and are loaded/run as `.rb` in Sonic Pi 5.x (`/Applications/Sonic Pi.app`). Also supports offline headless WAV rendering without the Sonic Pi server.

Core library: `src/pixel_alchemy/incidental/` — programmatic API + CLI. Reference idioms live in `reference/*.rb` (canonical Sonic Pi patterns to guide generation).

## File type

**`.rb`** — Ruby file with Sonic Pi DSL. Validated with `ruby -c <file>` (Sonic Pi symbols like `:Fs4`, `:Cs5` for sharps, not `:F#4`). Alternative extension `.spi` also accepted by Sonic Pi, but `.rb` is canonical.

## Reference examples

All 6 canonical idioms are vendored under `src/pixel_alchemy/incidental/reference/` — use as style guides for future generation:

| file | idiom | key pattern to reuse |
|---|---|---|
| `ambient_darin.rb` | ambient (Darin Wilson) | `use_synth :hollow` + 3 phasing `live_loop` with `choose([:D4,:E4])`, `attack: 6, release: 6`, `sleep 8/10/11` — out-of-phase long loops |
| `chord_cycle.rb` | chord | `[1,3,6,4].each` + `chord_degree d, :c, :major, 3, invert: i` + `play_chord` — diatonic `chord_degree` + `invert` sweep |
| `ocean.rb` | ocean sounds | `synth [:bnoise,:cnoise,:gnoise].choose` + `rrand` for `amp/cutoff/pan/attack/sustain/release` + `control` — noise-based texture |
| `ambient_sampling.rb` | ambient sampling | `sample_names :ambi` + `sample sp_name, cutoff: rrand(70,130), rate: choose([0.5,1]), pan: rrand` — `load_samples` + `with_fx :reverb, mix: 0.8` |
| `haunted.rb` | haunted windchimes | `sample :perc_bell, rate: rrand(-1.5,1.5)` + `sleep rrand(0.1,2)` — sparse aleatoric bells |
| `choral.rb` | choral | `ring 0.5, 1.0/3, 3.0/5` + `sample :ambi_choir, rate: r` + `cue :choir` — `use_debug false`, choral pad |

Load programmatically: `from pixel_alchemy.incidental.references import list_references, load_reference`

## Generation guidance

When composing new ambient beds, combine:

- **Progressions**: 49 in `progressions.py` (e.g. `I-V-vi-IV [1,5,6,4]`, `canon [1,5,6,3,4,1,4,5]`, `satie [1,2,4,1]`) — prefer `major`/`minor` at 42–62 BPM for ambient.
- **Piano/guitar**: `play`/`play_chord` with `:piano`/`:pluck`/`:hollow`/`:blade` + `attack: 0.3-6, release: 3-6, cutoff: 85-95` — arpeggio patterns `up/down/alberti/converge/held` at 1–2 beats per note.
- **Elongated melody**: `choose(c)` over chord tones + `scale(:c4,:minor)` passing tones, `sleep [4,6,8]` with occasional `sleep 2 if one_in(6)` — very sparse.
- **Texture**: layer `with_fx :reverb, mix: 0.5-0.8` +optional `:lpf cutoff: 95` or `:bnoise`/`sample :ambi_choir/:perc_bell` for ocean/haunted/choral beds (see references).
- **Aleatoric**: use `choose`, `rrand`, `one_in`, `ring(...).choose` per references — never hard-code only fixed notes.

## Usage

```bash
# programmatic
from pixel_alchemy.incidental.generator import generate_piece, to_sonic_pi
piece = generate_piece(seed=42, key="C", scale="major", progression="canon", bars=8, instrument="piano")
open("out.rb","w").write(to_sonic_pi(piece))  # -> Sonic Pi .rb

# CLI -> .rb (Sonic Pi) and/or .wav (headless)
uv run python -m pixel_alchemy.incidental.cli --seed 42 --progression canon --bars 16 -o ambient.rb
uv run python -m pixel_alchemy.incidental.cli --seed 42 -o ambient.wav         # offline wav
uv run python -m pixel_alchemy.incidental.cli --list                           # 49 progressions
ruby -c ambient.rb  # validate

# reference files
ls src/pixel_alchemy/incidental/reference/*.rb
```

## Options (CLI)

| option | default | meaning |
|---|---|---|
| `--seed N` | random | reproducible |
| `--key K` | random | `C,G,D,Bb` etc. |
| `--scale S` | random | `major,minor,dorian,lydian` |
| `--progression NAME` | random | see `--list` |
| `--bpm N` | 42–62 | slow ambient |
| `--bars N` | 16 | bars (4 beats each) |
| `--instrument I` | `piano` | `piano,guitar,pad` |
| `-o OUT` | stdout | `.rb` or `.wav` (ext decides), `--wav` also renders wav alongside rb |
| `--duration DUR` | — | `2m`/`10m` alias for bars |
