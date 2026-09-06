"""Offline WAV rendering — no Sonic Pi server required.

Simple additive synth (sine + soft overtone + ADSR) + one-pole lowpass + reverb tail
via freeverb-ish comb. Keeps dependencies to numpy/scipy/soundfile which already exist.
"""
import numpy as np
import soundfile as sf


def _midi_to_freq(midi: int) -> float:
    return 440.0 * (2 ** ((midi - 69) / 12))


def _adsr(n: int, sr: int, attack: float, sustain: float, release: float, dur: float) -> np.ndarray:
    a = int(attack * sr)
    r = int(release * sr)
    total = int(dur * sr)
    # pad/cut sustain to fit
    env = np.zeros(total)
    # attack linear
    if a > 0:
        env[:a] = np.linspace(0, 1, a)
    # sustain hold
    s_start = a
    s_end = total - r
    if s_end > s_start:
        env[s_start:s_end] = 1.0
        # slight decay during sustain
        env[s_start:s_end] *= np.linspace(1.0, 0.85, s_end - s_start)
    if r > 0 and total - r >= 0:
        env[total - r :] = np.linspace(env[total - r - 1] if total - r > 0 else 0.85, 0, r)
    return env


def render_wav(events: list[tuple[int, float, float]], bpm: int, out_path: str, sr: int = 44100):
    beat_sec = 60.0 / bpm
    # total duration
    end_beat = max((s + d) for _, s, d in events) if events else 0
    total_sec = end_beat * beat_sec + 2.0  # tail
    total_n = int(total_sec * sr)
    mix = np.zeros(total_n, dtype=np.float32)

    for midi, start_beat, dur_beats in events:
        freq = _midi_to_freq(midi)
        start = int(start_beat * beat_sec * sr)
        dur = dur_beats * beat_sec
        n = int(dur * sr)
        if n <= 0 or start >= total_n:
            continue
        n = min(n, total_n - start)
        t = np.arange(n) / sr
        # soft piano-like additive
        tone = (
            0.5 * np.sin(2 * np.pi * freq * t)
            + 0.22 * np.sin(2 * np.pi * 2 * freq * t) * np.exp(-t * 3)
            + 0.12 * np.sin(2 * np.pi * 3 * freq * t) * np.exp(-t * 5)
        )
        # warmer for low notes
        if midi < 60:
            tone *= 0.9
        # envelope: slow attack/release for ambient
        env = _adsr(n, sr, attack=0.35, sustain=dur * 0.6, release=1.1, dur=dur)
        if len(env) != n:
            env = env[:n]
        sig = tone * env * 0.22
        mix[start : start + n] += sig.astype(np.float32)

    # gentle lowpass (one-pole)
    alpha = 0.18
    y = np.zeros_like(mix)
    y[0] = mix[0]
    for i in range(1, len(mix)):
        y[i] = alpha * mix[i] + (1 - alpha) * y[i - 1]
    mix = y

    # simple reverb tail: short delay mix
    delay_s = 0.33
    delay_n = int(delay_s * sr)
    rev = np.zeros_like(mix)
    rev[delay_n:] += mix[:-delay_n] * 0.28
    rev[delay_n * 2 :] += mix[: -delay_n * 2] * 0.14
    mix = mix * 0.88 + rev * 0.42

    # normalize
    peak = np.max(np.abs(mix))
    if peak > 0:
        mix = mix / peak * 0.82

    # fade in/out 2s
    fade_n = int(2.0 * sr)
    fade_in = np.linspace(0, 1, fade_n)
    fade_out = np.linspace(1, 0, fade_n)
    mix[:fade_n] *= fade_in
    mix[-fade_n:] *= fade_out

    import pathlib

    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_path, mix, sr)
    return out_path
