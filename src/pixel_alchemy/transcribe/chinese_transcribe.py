"""Chinese audio transcription and English translation pipeline.

Uses transcribe-cli with the MOSS-Transcribe-Diarize model for offline
Chinese speech-to-text with speaker attribution, then translates each
speaker segment to English via a local chat completions endpoint.

Example::

    from pixel_alchemy.transcribe.chinese_transcribe import transcribe_and_translate

    result = transcribe_and_translate(
        "meeting.wav",
        model_path="models/MOSS-Transcribe-Diarize-Q8_0.gguf",
    )
    for seg in result["segments"]:
        print(f"{seg['speaker']}: {seg['text']} -> {seg['translated_text']}")
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from itertools import pairwise
from pathlib import Path

MAX_CHUNK_SECONDS = 3600  # 1 hour
SILENCE_THRESHOLD_DB = -30
SILENCE_MIN_DURATION_S = 0.5


def _find_model() -> Path | None:
    """Auto-discover MOSS-Transcribe-Diarize model."""
    model_dir = Path.home() / ".local" / "share" / "transcribe.cpp" / "models"
    matches = sorted(model_dir.glob("MOSS-Transcribe-Diarize-*.gguf"))
    return matches[0] if matches else None


def _find_transcribe_cli() -> Path:
    binary = shutil.which("transcribe-cli")
    if binary is None:
        raise FileNotFoundError("transcribe-cli not found on PATH")
    return Path(binary)


def _convert_to_wav(input_path: Path) -> Path:
    """Convert audio to 16kHz mono WAV via ffmpeg."""
    tmp = Path(tempfile.mkdtemp()) / f"{input_path.stem}.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(input_path), "-ar", "16000", "-ac", "1", str(tmp)],
        check=True,
        capture_output=True,
    )
    return tmp


def _get_duration_s(wav_path: Path) -> float:
    """Get audio duration in seconds via ffprobe."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    return float(result.stdout.decode().strip())


def _detect_silence(wav_path: Path) -> list[float]:
    """Detect silence boundaries using ffmpeg silencedetect.

    Returns sorted list of timestamps (seconds) where silence starts.
    """
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(wav_path),
            "-af", (
                f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB"
                f":duration={SILENCE_MIN_DURATION_S}"
            ),
            "-f", "null", "-",
        ],
        check=True,
        capture_output=True,
    )
    stderr = result.stderr.decode()
    starts = [float(m.group(1)) for m in re.finditer(r"silence_start:\s*([\d.]+)", stderr)]
    return sorted(starts)


def _find_chunk_boundaries(silences: list[float], duration_s: float) -> list[tuple[float, float]]:
    """Pick split points from silence list, producing chunks <= MAX_CHUNK_SECONDS.

    Walks forward through silences, splitting at the nearest silence before
    each boundary.  Always yields contiguous non-overlapping ranges.
    """
    if duration_s <= MAX_CHUNK_SECONDS:
        return [(0.0, duration_s)]

    boundaries: list[float] = [0.0]
    cursor = 0.0

    while cursor + MAX_CHUNK_SECONDS < duration_s:
        target = cursor + MAX_CHUNK_SECONDS
        # find the last silence before (or near) the target
        best = None
        for s in silences:
            if cursor + 10 <= s <= target:  # at least 10s from cursor
                best = s
        if best is None:
            # no good silence found – just split at target
            best = target
        boundaries.append(best)
        cursor = best

    boundaries.append(duration_s)
    return list(pairwise(boundaries))


def _split_wav(wav_path: Path, chunks_dir: Path, ranges: list[tuple[float, float]]) -> list[Path]:
    """Split a WAV file into chunk files for the given time ranges."""
    paths = []
    for i, (start, end) in enumerate(ranges):
        out = chunks_dir / f"chunk_{i:04d}.wav"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(wav_path),
                "-ss", str(start),
                "-to", str(end),
                "-c", "copy",
                str(out),
            ],
            check=True,
            capture_output=True,
        )
        paths.append(out)
    return paths


def _parse_diarized_output(raw: str) -> list[dict]:
    """Parse [start][Sxx]text[end] markers into segment dicts."""
    pattern = re.compile(r"\[(\d+(?:\.\d+)?)\]\[(S\d+)\](.*?)\[(\d+(?:\.\d+)?)\]")
    segments = []
    for m in pattern.finditer(raw):
        segments.append({
            "start": float(m.group(1)),
            "speaker": m.group(2),
            "text": m.group(3).strip(),
            "end": float(m.group(4)),
        })
    return segments


def _clean_full_text(raw: str) -> str:
    """Strip diarization markers from raw output to get plain text."""
    return re.sub(r"\[\d+(?:\.\d+)?\]\[S\d+\]|\[\d+(?:\.\d+)?\]", " ", raw).strip()


def _transcribe_chunk(cli: Path, model_path: Path, wav_path: Path) -> str:
    """Run transcribe-cli on a single WAV chunk, return raw output."""
    result = subprocess.run(
        [
            str(cli),
            "-m", str(model_path),
            "-l", "zh",
            "-q",
            "--timestamps", "none",
            str(wav_path),
        ],
        check=True,
        capture_output=True,
    )
    stdout = result.stdout.decode()
    for line in stdout.splitlines():
        if line.startswith("text:"):
            return line[len("text:"):].strip()
    return ""


def transcribe_chinese(
    audio_path: str | Path,
    model_path: str | Path | None = None,
) -> dict:
    """Transcribe Chinese audio with speaker diarization.

    For files longer than 1 hour, automatically splits on silence boundaries
    and transcribes each chunk separately.

    Args:
        audio_path: Path to the input audio file.
        model_path: Path to MOSS-Transcribe-Diarize .gguf model.
            Auto-discovered from ~/.local/share/transcribe.cpp/models/ if None.

    Returns:
        {"full_text": "...", "segments": [{"start", "end", "speaker", "text"}, ...]}
    """
    if model_path is None:
        model_path = _find_model()
        if model_path is None:
            raise FileNotFoundError(
                "MOSS-Transcribe-Diarize model not found. "
                "Pass model_path explicitly or download from "
                "https://huggingface.co/handy-computer/MOSS-Transcribe-Diarize-gguf"
            )
    else:
        model_path = Path(model_path)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    cli = _find_transcribe_cli()

    print(f"  [stage] Converting {audio_path.name} to WAV...")
    wav_path = _convert_to_wav(audio_path)
    try:
        duration_s = _get_duration_s(wav_path)
        print(f"  [stage] Duration: {duration_s:.1f}s")

        if duration_s <= MAX_CHUNK_SECONDS:
            print("  [stage] Transcribing...")
            raw = _transcribe_chunk(cli, model_path, wav_path)
        else:
            silences = _detect_silence(wav_path)
            ranges = _find_chunk_boundaries(silences, duration_s)
            print(f"  [stage] Split into {len(ranges)} chunks, transcribing...")
            chunks_dir = Path(tempfile.mkdtemp())
            chunk_paths = _split_wav(wav_path, chunks_dir, ranges)
            try:
                raw = ""
                for i, chunk in enumerate(chunk_paths):
                    print(f"  [stage] Transcribing chunk {i+1}/{len(chunk_paths)}...")
                    raw += _transcribe_chunk(cli, model_path, chunk)
            finally:
                for p in chunk_paths:
                    p.unlink(missing_ok=True)
                chunks_dir.rmdir()
    finally:
        wav_path.unlink(missing_ok=True)
        wav_path.parent.rmdir()

    segments = _parse_diarized_output(raw)
    full_text = _clean_full_text(raw)
    print(f"  [stage] Transcription done: {len(segments)} segments")

    return {"full_text": full_text, "segments": segments}


def translate_segments(
    segments: list[dict],
    *,
    endpoint: str = "http://localhost:1234/v1/chat/completions",
    model: str = "qwen3.6-35b-a3b-mtp",
) -> list[dict]:
    """Translate Chinese segments to English via chat completions.

    Args:
        segments: List of dicts with at least a "text" key.
        endpoint: Chat completions API URL.
        model: Model name for the completions request.

    Returns:
        New list of segments with added "translated_text" key.
    """
    print(f"  [stage] Translating {len(segments)} segments ({endpoint}, model={model})...")
    translated = []
    for i, seg in enumerate(segments):
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "Translate the following Chinese text to English. Output only the translation, nothing else.",
                },
                {"role": "user", "content": seg["text"]},
            ],
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req) as resp:
                body = json.loads(resp.read())
        except urllib.error.URLError as e:
            print(f"  [error] Translation request failed for segment {i}: {e}")
            raise

        translated_text = body["choices"][0]["message"]["content"].strip()
        translated.append({**seg, "translated_text": translated_text})
        print(f"  [stage] Translated segment {i+1}/{len(segments)}")

    print(f"  [stage] Translation done")
    return translated


def transcribe_and_translate(
    audio_path: str | Path,
    model_path: str | Path | None = None,
    *,
    endpoint: str = "http://localhost:1234/v1/chat/completions",
    model: str = "qwen3.6-35b-a3b-mtp",
) -> dict:
    """Convenience wrapper: transcribe Chinese audio then translate to English.

    Args:
        audio_path: Path to the input audio file.
        model_path: Path to MOSS-Transcribe-Diarize .gguf model.
        endpoint: Chat completions API URL.
        model: Model name for translation.

    Returns:
        {"full_text": "...", "segments": [{"start", "end", "speaker", "text", "translated_text"}, ...]}
    """
    result = transcribe_chinese(audio_path, model_path)
    result["segments"] = translate_segments(
        result["segments"], endpoint=endpoint, model=model
    )
    return result
