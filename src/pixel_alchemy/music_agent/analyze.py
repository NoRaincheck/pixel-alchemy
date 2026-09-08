"""Audio analysis for Sonic Pi generation — TF-free, optional ONNX YAMNet.

Replaces App/services/SampleMedataListing.py (librosa + TF YAMNet) with:
- librosa for BPM / key / spectral / energy (always available)
- optional ONNX YAMNet via onnxruntime (no tensorflow dep)
- graceful fallback if ONNX model not present

Example::

    from pixel_alchemy.music_agent.analyze import analyze_sample, analyze_directory

    meta = analyze_sample("input.wav")
    metas = analyze_directory("samples/sounds", out_json="sample_metadata.json")
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import librosa  # type: ignore

    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore
    HAS_LIBROSA = False

# ── ONNX YAMNet helpers (optional) ────────────────────────────────────────────

_ONNX_MODEL = None
_ONNX_CLASSES: list[str] | None = None
_ONNX_INPUT_NAME: str | None = None

YAMNET_ONNX_URL = "https://huggingface.co/onnx-community/yamnet-ONNX/resolve/main/onnx/model.onnx"
YAMNET_CLASS_MAP_URL = "https://huggingface.co/onnx-community/yamnet-ONNX/resolve/main/onnx/class_map.csv"
# legacy TF class map path still works for vendored assets if user has it

YAMNET_CLASSES = 521  # AudioSet


def _try_load_onnx(model_path: str | Path = "models/yamnet/model.onnx") -> bool:
    """Attempt to load ONNX YAMNet. Returns True on success."""
    global _ONNX_MODEL, _ONNX_CLASSES, _ONNX_INPUT_NAME
    if _ONNX_MODEL is not None:
        return True
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    mp = Path(model_path)
    if not mp.exists():
        # also try alternative locations
        for cand in [
            Path("models/yamnet.onnx"),
            Path.home() / ".cache" / "pixel-alchemy" / "yamnet.onnx",
        ]:
            if cand.exists():
                mp = cand
                break
        else:
            return False
    try:
        _ONNX_MODEL = ort.InferenceSession(str(mp), providers=["CPUExecutionProvider"])
        _ONNX_INPUT_NAME = _ONNX_MODEL.get_inputs()[0].name
        # load class map
        class_map = mp.parent / "class_map.csv"
        if not class_map.exists():
            class_map = mp.parent / "yamnet_class_map.csv"
        if class_map.exists():
            with open(class_map) as f:
                reader = csv.DictReader(f)
                # support both formats
                if "display_name" in (reader.fieldnames or []):
                    _ONNX_CLASSES = [r["display_name"] for r in reader]
                else:
                    _ONNX_CLASSES = [r.strip().split(",")[-1].strip().strip('"') for r in open(class_map)]
        else:
            _ONNX_CLASSES = [f"class_{i}" for i in range(YAMNET_CLASSES)]
        return True
    except Exception:
        _ONNX_MODEL = None
        return False


def _onnx_classify(waveform: np.ndarray, sr: int = 16000, top_k: int = 5) -> list[tuple[str, float]]:
    """Classify with ONNX YAMNet. waveform: mono float32 in [-1,1]."""
    if not _try_load_onnx():
        return []
    # YAMNet expects 16k mono, 0.96s windows (15600 samples) — we just feed whole clip
    # ONNX yamnet typically does framing internally or expects 15600 chunks; we average.
    # Simplify: resample handled by caller, chunk and average logits.
    if _ONNX_MODEL is None or _ONNX_CLASSES is None:
        return []
    # chunk into 0.96s windows
    win = 15600
    if len(waveform) < win:
        waveform = np.pad(waveform, (0, win - len(waveform)))
    scores_acc = None
    for i in range(0, len(waveform) - win + 1, win):
        chunk = waveform[i : i + win].astype(np.float32)[None, :]
        try:
            out = _ONNX_MODEL.run(None, {_ONNX_INPUT_NAME: chunk})
            # out[0] is logits or scores depending on export
            logits = out[0][0] if out[0].ndim == 2 else out[0]
            # softmax if needed
            if logits.max() > 1:
                e = np.exp(logits - logits.max())
                scores = e / e.sum()
            else:
                scores = logits
            scores_acc = scores if scores_acc is None else scores_acc + scores
        except Exception:
            continue
    if scores_acc is None:
        return []
    scores_acc = scores_acc / max(1, len(waveform) // win)
    top = np.argsort(scores_acc)[-top_k:][::-1]
    return [(_ONNX_CLASSES[i] if i < len(_ONNX_CLASSES) else f"class_{i}", float(scores_acc[i])) for i in top]


def download_yamnet_onnx(dest: str | Path = "models/yamnet/model.onnx") -> Path:
    """Download ONNX YAMNet model and class map. Requires huggingface_hub or urllib."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _dl(url: str, out: Path):
        if out.exists():
            return
        try:
            from huggingface_hub import hf_hub_download  # type: ignore

            # use hf_hub_download for robustness
            import urllib.request

            urllib.request.urlretrieve(url, str(out))
        except Exception:
            import urllib.request

            urllib.request.urlretrieve(url, str(out))

    _dl(YAMNET_ONNX_URL, dest)
    class_map = dest.parent / "class_map.csv"
    if not class_map.exists():
        _dl(YAMNET_CLASS_MAP_URL, class_map)
    return dest


# ── Key detection (from MusicAgent) ───────────────────────────────────────────

def detect_key(y: np.ndarray, sr: int) -> str:
    if not HAS_LIBROSA:
        return "Unknown"
    try:
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, n_chroma=12, hop_length=512)
        chroma_mean = chroma.mean(axis=1)
        if np.max(chroma_mean) == 0:
            return "Unknown"
        chroma_mean = chroma_mean / np.max(chroma_mean)
        major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1], dtype=float)
        minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0], dtype=float)
        major_scores = [np.correlate(chroma_mean, np.roll(major_template, i))[0] for i in range(12)]
        minor_scores = [np.correlate(chroma_mean, np.roll(minor_template, i))[0] for i in range(12)]
        best_major = int(np.argmax(major_scores))
        best_minor = int(np.argmax(minor_scores))
        key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        if major_scores[best_major] > minor_scores[best_minor]:
            return f"{key_names[best_major]} major"
        return f"{key_names[best_minor]} minor"
    except Exception:
        return "Unknown"


def _yamnet_tags(wav_path: Path) -> list[str]:
    """Return top YAMNet tags if ONNX available, else empty."""
    if not _try_load_onnx():
        return []
    if not HAS_LIBROSA:
        return []
    try:
        y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
        y = y.astype(np.float32)
        results = _onnx_classify(y, sr=16000, top_k=5)
        return [name for name, _ in results]
    except Exception:
        return []


# ── Core process_audio (ported from SampleMedataListing.py) ───────────────────

def analyze_sample(audio_path: str | Path, use_yamnet: bool = True) -> dict:
    """Analyze a single audio file into MusicAgent-compatible metadata.

    Returns dict with Filename, Duration, BPM, Key, Vibe, Tags, Description, Track Type.
    Never raises on bad files — returns dict with Error key.
    Uses librosa if available, else falls back to soundfile duration only.
    """
    p = Path(audio_path)
    try:
        if HAS_LIBROSA:
            y, sr = librosa.load(str(p), sr=None, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            if duration < 0.8:
                return {"Filename": p.as_posix(), "Error": "Audio too short to analyze"}

            n_fft = min(1024, len(y))
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            if isinstance(tempo, np.ndarray):
                tempo = float(tempo[0]) if tempo.size > 0 else 0.0
            else:
                tempo = float(tempo)

            tempo_category = "Relaxed" if tempo < 90 else "Moderate" if tempo < 120 else "Energetic"
            spectral = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft)
            brightness = "bright" if float(spectral.mean()) > 2000 else "warm"
            rms = float(librosa.feature.rms(y=y, frame_length=n_fft).mean())
            if rms < 0.005:
                energy = "very low energy"
            else:
                energy = "high energy" if rms > 0.05 else "low energy"
            max_amp = float(np.max(np.abs(y))) if len(y) else 0
            dynamic_range = "intense and punchy" if max_amp > 0.8 else "soft and smooth"
            key = detect_key(y, sr)
        else:
            # fallback without librosa: duration via soundfile, dummy musical attrs
            info = sf.info(str(p))
            duration = float(info.frames / info.samplerate) if info.samplerate else 0
            if duration < 0.8:
                return {"Filename": p.as_posix(), "Error": "Audio too short to analyze"}
            y, sr = sf.read(str(p), always_2d=False)
            if isinstance(y, np.ndarray) and y.ndim > 1:
                y = y.mean(axis=-1)
            y = np.asarray(y, dtype=np.float32)
            tempo = 100.0
            tempo_category = "Moderate"
            brightness = "warm"
            energy = "low energy"
            dynamic_range = "soft and smooth"
            key = "Unknown"

        classification_tags: list[str] = []
        if use_yamnet:
            classification_tags = _yamnet_tags(p)

        # track type heuristic
        instrumental_cats = {
            "Piano",
            "Electric Guitar",
            "Drums",
            "Bass Guitar",
            "Flute",
            "Violin",
            "Harmonica",
            "Synthesizer",
            "Trumpet",
            "Saxophone",
            "Acoustic Guitar",
            "Strings",
        }
        vocal_cats = {"Singing", "Choir", "Speech", "Vocal", "Opera", "Rap"}
        contains_vocals = any(t in vocal_cats for t in classification_tags)
        contains_instr = any(t in instrumental_cats for t in classification_tags)
        track_type = "Unknown"
        if contains_vocals and contains_instr:
            track_type = "Both Vocals and Instrumentals"
        elif contains_vocals:
            track_type = "Vocals Only"
        elif contains_instr:
            track_type = "Instrumentals Only"

        vibe = (
            f"The track has a {tempo_category} tempo at {round(tempo)} BPM, "
            f"featuring a {brightness} and {energy} sound. It feels {dynamic_range} "
            f"with a {key} tonality."
        )
        tags = [tempo_category, brightness, energy, dynamic_range, key] + classification_tags
        tags = [t for t in tags if t != "Unknown"]

        parent = p.parent.name
        result: dict = {
            "Filename": f"{parent}/{p.name}" if parent else p.name,
            "Duration": round(duration, 2),
            "BPM": round(tempo, 2),
            "Key": key,
            "Vibe": vibe,
            "Tags": tags,
            "Description": f"A {brightness}, {energy} track with a {tempo_category} tempo and a {key} tonality.",
        }
        if track_type != "Unknown":
            result["Track Type"] = track_type
        return result
    except Exception as e:
        return {"Filename": p.as_posix(), "Error": str(e)}


def analyze_directory(
    input_dir: str | Path,
    out_json: str | Path | None = None,
    use_yamnet: bool = True,
    exts: tuple[str, ...] = (".wav", ".mp3", ".flac", ".ogg", ".m4a"),
) -> list[dict]:
    """Walk input_dir, analyze each audio file, optionally write JSON array."""
    input_dir = Path(input_dir)
    results: list[dict] = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(exts):
                fp = Path(root) / f
                meta = analyze_sample(fp, use_yamnet=use_yamnet)
                if "Error" not in meta:
                    results.append(meta)
                else:
                    print(f"[skip] {fp}: {meta['Error']}")
    if out_json is not None and results:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, ensure_ascii=False)
        print(f"Wrote {len(results)} records to {out}")
    return results
