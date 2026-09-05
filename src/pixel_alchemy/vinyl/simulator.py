"""Offline vinyl simulator — Python port of brookrichardson/vinyl-simulator.

Replicates the Web Audio API graph in numpy/scipy for offline file processing.

Signal chain (matches index.html:buildGraph):
    source -> wow/flutter delay -> 14.5k LPF -> vinyl-age LPF
           -> warmth peaking (220Hz) -> warmth highshelf (6kHz)
           -> RIAA lowshelf (200Hz) -> RIAA highshelf (3kHz)
           -> waveshaper -> stylus resonance (10kHz) -> worn highshelf (8kHz)
           -> output
    Parallel additives: surface noise (pink LPF 4kHz), crackle (BP 2.2kHz),
    turntable rumble (BP 55Hz), ghost echo (delay ~1.8s).

Example::

    from pixel_alchemy.vinyl.simulator import process_file

    process_file("input.wav", "output.wav", preset="well-played")
    process_file("input.wav", "output.wav", preset="worn-classic", warmth=80)

    # numpy array API
    from pixel_alchemy.vinyl.simulator import process_audio
    import soundfile as sf
    data, sr = sf.read("input.wav")
    out = process_audio(data, sr, preset="lo-fi")
    sf.write("out.wav", out, sr)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy.signal import lfilter  # type: ignore[import-untyped]
except ImportError:
    lfilter = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Presets — identical to JS PRESETS
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict[str, int]] = {
    "digital": {"warmth": 0, "noise": 0, "wow": 0, "crackle": 0, "age": 0, "resonance": 0, "worn": 0, "rumble": 0, "ghost": 0, "riaa": 50},
    "audiophile": {"warmth": 40, "noise": 0, "wow": 0, "crackle": 0, "age": 0, "resonance": 60, "worn": 0, "rumble": 0, "ghost": 0, "riaa": 42},
    "fresh-press": {"warmth": 25, "noise": 8, "wow": 12, "crackle": 3, "age": 0, "resonance": 45, "worn": 0, "rumble": 10, "ghost": 8, "riaa": 47},
    "well-played": {"warmth": 50, "noise": 40, "wow": 35, "crackle": 30, "age": 20, "resonance": 25, "worn": 15, "rumble": 20, "ghost": 15, "riaa": 45},
    "worn-classic": {"warmth": 75, "noise": 70, "wow": 50, "crackle": 75, "age": 80, "resonance": 10, "worn": 65, "rumble": 45, "ghost": 55, "riaa": 30},
    "lo-fi": {"warmth": 90, "noise": 85, "wow": 75, "crackle": 90, "age": 55, "resonance": 5, "worn": 80, "rumble": 65, "ghost": 50, "riaa": 25},
}


@dataclass
class VinylParams:
    warmth: int = 50
    noise: int = 40
    wow: int = 35
    crackle: int = 30
    age: int = 0
    resonance: int = 25
    worn: int = 0
    rumble: int = 20
    ghost: int = 15
    riaa: int = 50

    @classmethod
    def from_preset(cls, name: str, **overrides: int) -> "VinylParams":
        if name not in PRESETS:
            raise ValueError(f"unknown preset {name!r}, choices: {list(PRESETS)}")
        p = dict(PRESETS[name])
        p.update(overrides)
        return cls(**p)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Biquad helpers — RBJ cookbook, matches Web Audio BiquadFilter
# ---------------------------------------------------------------------------

def _biquad_coeffs(
    b0: float, b1: float, b2: float, a0: float, a1: float, a2: float
) -> tuple[np.ndarray, np.ndarray]:
    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def biquad_lowpass(fc: float, Q: float, sr: float) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * Q)
    return _biquad_coeffs((1 - cos_w0) / 2, 1 - cos_w0, (1 - cos_w0) / 2, 1 + alpha, -2 * cos_w0, 1 - alpha)


def biquad_peaking(fc: float, Q: float, gain_db: float, sr: float) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * Q)
    return _biquad_coeffs(1 + alpha * A, -2 * cos_w0, 1 - alpha * A, 1 + alpha / A, -2 * cos_w0, 1 - alpha / A)


def biquad_highshelf(fc: float, gain_db: float, sr: float, S: float = 1) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2 * math.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    sqrtA = math.sqrt(A)
    return _biquad_coeffs(
        A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha),
        -2 * A * ((A - 1) + (A + 1) * cos_w0),
        A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha),
        (A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha,
        2 * ((A - 1) - (A + 1) * cos_w0),
        (A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha,
    )


def biquad_lowshelf(fc: float, gain_db: float, sr: float, S: float = 1) -> tuple[np.ndarray, np.ndarray]:
    A = 10 ** (gain_db / 40)
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / 2 * math.sqrt((A + 1 / A) * (1 / S - 1) + 2)
    sqrtA = math.sqrt(A)
    return _biquad_coeffs(
        A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqrtA * alpha),
        2 * A * ((A - 1) - (A + 1) * cos_w0),
        A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqrtA * alpha),
        (A + 1) + (A - 1) * cos_w0 + 2 * sqrtA * alpha,
        -2 * ((A - 1) + (A + 1) * cos_w0),
        (A + 1) + (A - 1) * cos_w0 - 2 * sqrtA * alpha,
    )


def biquad_bandpass(fc: float, Q: float, sr: float) -> tuple[np.ndarray, np.ndarray]:
    w0 = 2 * math.pi * fc / sr
    cos_w0 = math.cos(w0)
    sin_w0 = math.sin(w0)
    alpha = sin_w0 / (2 * Q)
    return _biquad_coeffs(alpha, 0, -alpha, 1 + alpha, -2 * cos_w0, 1 - alpha)


def _apply_filter(x: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if lfilter is not None:
        return lfilter(b, a, x)
    # fallback manual (slow)
    y = np.zeros_like(x)
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for n in range(len(x)):
        y[n] = b0 * x[n]
        if n >= 1:
            y[n] += b1 * x[n - 1] - a1 * y[n - 1]
        if n >= 2:
            y[n] += b2 * x[n - 2] - a2 * y[n - 2]
    return y


def _apply_stereo(data: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    if data.ndim == 1:
        return _apply_filter(data, b, a)
    out = np.empty_like(data)
    for ch in range(data.shape[1]):
        out[:, ch] = _apply_filter(data[:, ch], b, a)
    return out


# ---------------------------------------------------------------------------
# Noise / crackle generation — matches JS createNoiseBuffer / createCrackleBuffer
# ---------------------------------------------------------------------------

def _pink_noise(n: int, channels: int, rng: np.random.Generator) -> np.ndarray:
    out = np.empty((n, channels) if channels > 1 else n, dtype=np.float64)
    # Paul Kellet coefficients
    for ch in range(channels if channels > 1 else 1):
        b0 = b1 = b2 = b3 = b4 = b5 = b6 = 0.0
        buf = np.empty(n, dtype=np.float64)
        for i in range(n):
            wn = rng.uniform(-1, 1)
            b0 = 0.99886 * b0 + wn * 0.0555179
            b1 = 0.99332 * b1 + wn * 0.0750759
            b2 = 0.96900 * b2 + wn * 0.1538520
            b3 = 0.86650 * b3 + wn * 0.3104856
            b4 = 0.55000 * b4 + wn * 0.5329522
            b5 = -0.7616 * b5 - wn * 0.0168980
            v = (b0 + b1 + b2 + b3 + b4 + b5 + b6 + wn * 0.5362) * 0.11
            b6 = wn * 0.115926
            buf[i] = v
        if channels > 1:
            out[:, ch] = buf
        else:
            out = buf
    return out


def _crackle_buffer(n: int, rng: np.random.Generator) -> np.ndarray:
    d = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if rng.random() < 0.00015:
            amp = rng.uniform(0.2, 1.0)
            w = int(rng.integers(10, 50))
            for j in range(w):
                if i + j >= n:
                    break
                d[i + j] += amp * math.exp(-j * 0.15) * rng.uniform(-1, 1)
    return d


# ---------------------------------------------------------------------------
# Waveshaper — matches JS makeCurve / WaveShaper
# ---------------------------------------------------------------------------

def _clamped_fc(fc: float, sr: float) -> float:
    return min(fc, sr * 0.45)


def _waveshaper(x: np.ndarray, warmth: int) -> np.ndarray:
    k = warmth / 100 * 30 + 1  # warmth 0-100 -> 1..31
    pi = math.pi
    return ((pi + k) * x) / (pi + k * np.abs(x))


# ---------------------------------------------------------------------------
# Wow & Flutter — variable delay with two LFOs
# ---------------------------------------------------------------------------

def _wow_flutter(data: np.ndarray, sr: float, wow_amt: float) -> np.ndarray:
    """wow_amt 0..1 (wow slider /100). Modulates delay with 0.5Hz + 7Hz."""
    if wow_amt == 0:
        return data
    n = len(data)
    channels = 1 if data.ndim == 1 else data.shape[1]
    base_delay = 0.02  # seconds
    wow_gain = wow_amt * 0.006
    flutter_gain = wow_amt * 0.0012

    t = np.arange(n) / sr
    mod = wow_gain * np.sin(2 * math.pi * 0.5 * t) + flutter_gain * np.sin(2 * math.pi * 7 * t)
    delay_s = base_delay + mod  # shape (n,)

    # fractional delay via linear interpolation
    # need input prepend with zeros for max delay
    max_d = int(math.ceil((base_delay + wow_gain + flutter_gain) * sr)) + 2
    if data.ndim == 1:
        padded = np.concatenate([np.zeros(max_d), data.astype(np.float64)])
        out = np.empty_like(data, dtype=np.float64)
        for i in range(n):
            d = delay_s[i] * sr
            idx = max_d + i - d  # fractional read position in padded
            lo = int(math.floor(idx))
            frac = idx - lo
            lo = max(0, min(lo, len(padded) - 2))
            out[i] = padded[lo] * (1 - frac) + padded[lo + 1] * frac
        return out
    else:
        padded = np.concatenate([np.zeros((max_d, channels)), data.astype(np.float64)], axis=0)
        out = np.empty_like(data, dtype=np.float64)
        for i in range(n):
            d = delay_s[i] * sr
            idx = max_d + i - d
            lo = int(math.floor(idx))
            frac = idx - lo
            lo = max(0, min(lo, len(padded) - 2))
            out[i] = padded[lo] * (1 - frac) + padded[lo + 1] * frac
        return out


# ---------------------------------------------------------------------------
# Ghost echo — delayed copy ~1.8s (60/33.33)
# ---------------------------------------------------------------------------

def _ghost_echo(data: np.ndarray, sr: float, ghost_amt: float) -> np.ndarray:
    if ghost_amt == 0:
        return np.zeros_like(data, dtype=np.float64)
    delay_s = 60 / 33.33
    delay_n = int(round(delay_s * sr))
    gain = ghost_amt * 0.2
    if data.ndim == 1:
        delayed = np.concatenate([np.zeros(delay_n), data[:-delay_n] if delay_n < len(data) else np.zeros(len(data))])
        # handle case data shorter than delay
        if delay_n >= len(data):
            delayed = np.zeros_like(data)
        else:
            delayed = np.concatenate([np.zeros(delay_n), data[: len(data) - delay_n]])
        return delayed * gain
    else:
        if delay_n >= len(data):
            return np.zeros_like(data, dtype=np.float64)
        delayed = np.concatenate([np.zeros((delay_n, data.shape[1])), data[: len(data) - delay_n]], axis=0)
        return delayed * gain


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def _resolve_params(preset: str | None, params: VinylParams | None, overrides: dict) -> VinylParams:
    if params is not None:
        for k, v in overrides.items():
            if hasattr(params, k):
                setattr(params, k, int(v))
        return params
    if preset is not None:
        return VinylParams.from_preset(preset, **overrides)
    # use overrides as direct params, defaults for missing
    p = VinylParams()
    for k, v in overrides.items():
        if hasattr(p, k):
            setattr(p, k, int(v))
    return p


def process_audio(
    data: np.ndarray,
    sr: int,
    preset: str | None = "well-played",
    params: VinylParams | None = None,
    *,
    seed: int | None = 0,
    **overrides: int,
) -> np.ndarray:
    """Apply vinyl simulation to a numpy audio array.

    Args:
        data: Audio array, shape (samples,) mono or (samples, channels) stereo.
              Values expected in [-1, 1] float. Integer PCM will be converted.
        sr: Sample rate in Hz.
        preset: Preset name (digital, audiophile, fresh-press, well-played,
                worn-classic, lo-fi). Ignored if params is given.
        params: Explicit VinylParams. If given, preset is ignored.
        seed: RNG seed for noise/crackle reproducibility. None for random.
        **overrides: Per-knob overrides 0-100 (warmth, noise, wow, crackle,
                     age, resonance, worn, rumble, ghost, riaa).

    Returns:
        Processed audio array with same shape as input (float64, clipped to [-1,1]).
    """
    p = _resolve_params(preset, params, overrides)
    rng = np.random.default_rng(seed)

    # normalise integer PCM
    if data.dtype.kind in "iu":
        data = data.astype(np.float64) / np.iinfo(data.dtype).max
    else:
        data = data.astype(np.float64)

    is_mono = data.ndim == 1
    n = len(data)
    channels = 1 if is_mono else data.shape[1]

    # --- main chain ---
    x = _wow_flutter(data, float(sr), p.wow / 100)

    # 14.5 kHz gentle LPF
    b, a = biquad_lowpass(_clamped_fc(14500, float(sr)), 0.4, float(sr))
    x = _apply_stereo(x, b, a)

    # vinyl age LPF
    age_fc = _clamped_fc(14500 - (p.age / 100) * 6500, float(sr))
    b, a = biquad_lowpass(age_fc, 0.5, float(sr))
    x = _apply_stereo(x, b, a)

    # warmth peaking
    if p.warmth > 0:
        b, a = biquad_peaking(220, 0.8, (p.warmth / 100) * 5, float(sr))
        x = _apply_stereo(x, b, a)
        b, a = biquad_highshelf(_clamped_fc(6000, float(sr)), -(p.warmth / 100) * 3, float(sr))
        x = _apply_stereo(x, b, a)

    # RIAA mismatch
    riaa_mix = (p.riaa - 50) / 50  # -1 .. +1
    if riaa_mix != 0:
        b, a = biquad_lowshelf(200, -riaa_mix * 3, float(sr))
        x = _apply_stereo(x, b, a)
        b, a = biquad_highshelf(_clamped_fc(3000, float(sr)), riaa_mix * 4, float(sr))
        x = _apply_stereo(x, b, a)

    # waveshaper / saturation
    x = _waveshaper(x, p.warmth)

    # stylus resonance
    if p.resonance > 0 and 10000 < float(sr) / 2:
        b, a = biquad_peaking(10000, 2.5, (p.resonance / 100) * 6, float(sr))
        x = _apply_stereo(x, b, a)

    # worn stylus highshelf cut
    if p.worn > 0:
        b, a = biquad_highshelf(_clamped_fc(8000, float(sr)), -(p.worn / 100) * 8, float(sr))
        x = _apply_stereo(x, b, a)

    # --- parallel additives ---
    out = x.copy()

    # surface noise: pink LPF 4kHz
    if p.noise > 0:
        noise = _pink_noise(n, channels, rng)
        b, a = biquad_lowpass(_clamped_fc(4000, float(sr)), 0.707, float(sr))
        if is_mono:
            noise = _apply_filter(noise, b, a)  # type: ignore[arg-type]
        else:
            for ch in range(channels):
                noise[:, ch] = _apply_filter(noise[:, ch], b, a)
        out = out + noise * (p.noise / 100 * 0.04)

    # crackle: bandpass 2.2kHz
    if p.crackle > 0:
        crack_mono = _crackle_buffer(n, rng)
        b, a = biquad_bandpass(_clamped_fc(2200, float(sr)), 0.5, float(sr))
        crack_mono = _apply_filter(crack_mono, b, a)
        gain = p.crackle / 100 * 0.25
        if is_mono:
            out = out + crack_mono * gain
        else:
            out = out + np.repeat(crack_mono[:, None], channels, axis=1) * gain

    # turntable rumble: pink BP 55Hz
    if p.rumble > 0:
        rumble = _pink_noise(n, channels, rng)
        b, a = biquad_bandpass(55, 0.8, float(sr))
        if is_mono:
            rumble = _apply_filter(rumble, b, a)  # type: ignore[arg-type]
        else:
            for ch in range(channels):
                rumble[:, ch] = _apply_filter(rumble[:, ch], b, a)
        out = out + rumble * (p.rumble / 100 * 0.4)

    # ghost echo
    if p.ghost > 0:
        out = out + _ghost_echo(x, float(sr), p.ghost / 100)

    # soft clip to [-1,1]
    out = np.clip(out, -1.0, 1.0)
    return out


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(path)
    # try soundfile first
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        data, sr = sf.read(str(path), always_2d=False)
        return data, sr
    except ImportError:
        pass
    except Exception as e:
        raise RuntimeError(f"failed to read {path}: {e}") from e

    # fallback: scipy wav
    try:
        from scipy.io import wavfile  # type: ignore[import-untyped]

        sr, data = wavfile.read(str(path))
        if data.dtype.kind in "iu":
            data = data.astype(np.float64) / np.iinfo(data.dtype).max
        # wavfile returns (samples, channels) for stereo
        return data, sr  # type: ignore[return-value]
    except ImportError:
        pass
    raise RuntimeError("need soundfile or scipy to read audio; pip install soundfile scipy")


def _save_audio(path: str | Path, data: np.ndarray, sr: int) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf  # type: ignore[import-untyped]

        sf.write(str(path), data, sr)
        return
    except ImportError:
        pass
    # fallback scipy wav (only wav)
    try:
        from scipy.io import wavfile  # type: ignore[import-untyped]

        # convert float to int16 for wav
        int_data = np.clip(data, -1, 1)
        int_data = (int_data * 32767).astype(np.int16)
        wavfile.write(str(path), sr, int_data)
        return
    except ImportError:
        pass
    raise RuntimeError("need soundfile or scipy to write audio; pip install soundfile scipy")


def process_file(
    input_path: str | Path,
    output_path: str | Path | None = None,
    preset: str | None = "well-played",
    params: VinylParams | None = None,
    *,
    seed: int | None = 0,
    **overrides: int,
) -> Path:
    """Read an audio file, apply vinyl simulation, write output.

    Args:
        input_path: Source audio file (wav, flac, ogg, mp3 if soundfile/libsndfile supports).
        output_path: Destination file. Defaults to <input_stem>_vinyl.<ext>.
        preset: Preset name. Ignored if params given.
        params: Explicit VinylParams.
        seed: RNG seed (0 for reproducible, None for random).
        **overrides: Per-knob overrides 0-100.

    Returns:
        Path to written output file.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")
    if output_path is None:
        output_path = input_path.with_name(f"{input_path.stem}_vinyl{input_path.suffix}")
    else:
        output_path = Path(output_path)

    data, sr = _load_audio(input_path)
    out = process_audio(data, sr, preset=preset, params=params, seed=seed, **overrides)
    _save_audio(output_path, out, sr)
    return output_path
