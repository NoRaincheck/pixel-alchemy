"""Mix selected Moodist sounds into a single wav/ogg.

Loops short samples to fill `duration`, applies per-sound volume and master
fade, then writes 44.1kHz stereo via soundfile. Mirrors Howler's
loop+volume+fade behavior in `moodist` without browser APIs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .catalog import get_sound

TARGET_SR = 44100


def _load_mono(path: Path, target_sr: int = TARGET_SR) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        # linear resample (avoid torchaudio dep)
        ratio = target_sr / sr
        new_len = int(len(data) * ratio)
        if new_len == 0:
            return np.zeros(1, dtype=np.float32), target_sr
        x_old = np.linspace(0, 1, len(data))
        x_new = np.linspace(0, 1, new_len)
        data = np.interp(x_new, x_old, data).astype(np.float32)
    return data, target_sr


def _loop_to(data: np.ndarray, samples: int) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(samples, dtype=np.float32)
    if len(data) >= samples:
        return data[:samples]
    reps = (samples + len(data) - 1) // len(data)
    return np.tile(data, reps)[:samples]


def mix(
    sounds: dict[str, float],
    output: str | Path,
    *,
    duration: float = 30.0,
    target_sr: int = TARGET_SR,
    fade_in: float = 1.0,
    fade_out: float = 1.0,
    normalize: bool = True,
) -> Path:
    """Mix `sounds` ({id: volume 0-1}) into `output` wav/ogg.

    Args:
        sounds: mapping sound_id -> volume. Missing ids raise FileNotFoundError.
        output: destination wav/ogg path (ogg uses Vorbis at 44.1kHz, Opus at 48kHz).
        duration: seconds to render (loops each source).
        fade_in/out: linear fade seconds at head/tail.
        normalize: peak-normalize to 0.89 if >1.0 else gentle boost if very quiet.

    Returns:
        Resolved output Path.
    """
    if not sounds:
        raise ValueError("no sounds selected")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    total = int(target_sr * duration)
    mix_buf = np.zeros(total, dtype=np.float32)

    for sid, vol in sounds.items():
        snd = get_sound(sid)
        if snd is None:
            raise KeyError(f"unknown sound: {sid}")
        if not snd.src.exists():
            raise FileNotFoundError(f"missing sample: {snd.src}")
        vol = max(0.0, min(1.0, float(vol)))
        if vol == 0:
            continue
        data, _ = _load_mono(snd.src, target_sr)
        looped = _loop_to(data, total)
        mix_buf += looped * vol

    # fades (mirrors Moodist smooth transitions)
    if fade_in > 0:
        n = min(total, int(target_sr * fade_in))
        mix_buf[:n] *= np.linspace(0, 1, n, dtype=np.float32)
    if fade_out > 0:
        n = min(total, int(target_sr * fade_out))
        mix_buf[-n:] *= np.linspace(1, 0, n, dtype=np.float32)

    # normalize
    if normalize:
        peak = float(np.abs(mix_buf).max()) if mix_buf.size else 0.0
        if peak > 1.0:
            mix_buf *= 0.89 / peak
        elif 0 < peak < 0.08:
            mix_buf *= 0.35 / (peak + 1e-6)
        mix_buf = np.clip(mix_buf, -1.0, 1.0)

    # mono -> stereo duplicate for richer output like Howler stereo
    stereo = np.stack([mix_buf, mix_buf], axis=1)
    if output.suffix.lower() == ".ogg":
        # Opus only supports 8/12/16/24/48k; Vorbis supports 44.1k
        if target_sr in (8000, 12000, 16000, 24000, 48000):
            sf.write(str(output), stereo, target_sr, format="OGG", subtype="OPUS")
        else:
            sf.write(str(output), stereo, target_sr, format="OGG", subtype="VORBIS")
    else:
        sf.write(str(output), stereo, target_sr)
    return output


def mix_store(store, output: str | Path, **kw) -> Path:
    """Convenience: mix from a SoundStore instance."""
    return mix(store.selected(), output, **kw)
