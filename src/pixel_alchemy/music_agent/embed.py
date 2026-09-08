"""Simple sample selection — keyword match, no FAISS / no sentence_transformers.

Replaces App/services/agent.py::local_discussion FAISS retrieval with a
deterministic scorer suitable for small sample pools (e.g. samples/sounds 84 files).

The upstream bug encoded the entire metadata JSON as one vector; we fix
the concept by scoring per-item.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def select_samples(
    metadata: list[dict],
    query: str,
    k: int = 5,
) -> list[dict]:
    """Rank metadata items by keyword overlap with query, return top k.

    Falls back to first k if all scores zero (short query).
    """
    if not metadata:
        return []
    q_tokens = _tokenize(query)
    scored: list[tuple[int, dict]] = []
    for item in metadata:
        # build searchable text from tags/description/vibe/filename
        hay = " ".join(
            [
                " ".join(item.get("Tags", [])),
                item.get("Description", ""),
                item.get("Vibe", ""),
                item.get("Filename", ""),
                item.get("Key", ""),
            ]
        )
        h_tokens = _tokenize(hay)
        score = len(q_tokens & h_tokens)
        # small boost if BPM in query matches
        scored.append((score, item))
    scored.sort(key=lambda x: x[0], reverse=True)
    # if top score zero, just return first k in original order
    if scored[0][0] == 0:
        return metadata[:k]
    return [item for _, item in scored[:k]]


def select_samples_for_song(
    metadata_path: str | Path | None,
    theme: str = "",
    melody: str = "",
    rhythm: str = "",
    arrangements: str = "",
    song_description: str = "",
    k: int = 5,
) -> str:
    """Load metadata JSON and return JSON string of selected samples (as upstream).

    Mirrors song_creation_data.samples = json.dumps(results, separators=(',', ':'))
    """
    if metadata_path is None:
        return "[]"
    p = Path(metadata_path)
    if not p.exists():
        return "[]"
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return "[]"
    if not isinstance(data, list):
        return "[]"
    query = f"{theme} {melody} {rhythm} {arrangements} {song_description} samples"
    chosen = select_samples(data, query, k=k)
    return json.dumps(chosen, separators=(",", ":"))


def load_metadata(pool: str | Path) -> list[dict]:
    """Load metadata from file or directory (scans for *.json)."""
    p = Path(pool)
    if p.is_file() and p.suffix == ".json":
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except Exception:
            return []
    if p.is_dir():
        # look for sample_metadata.json or any *_metadata.json
        candidates = list(p.glob("sample_metadata.json")) + list(p.glob("*_metadata.json")) + list(p.glob("*.json"))
        for cand in candidates:
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if isinstance(data, list) and data:
                    return data
            except Exception:
                continue
        # fallback: no metadata — synthesize minimal entries from audio files
        from .analyze import analyze_sample

        exts = (".wav", ".mp3", ".flac", ".ogg", ".m4a")
        out: list[dict] = []
        for f in p.rglob("*"):
            if f.suffix.lower() in exts:
                m = analyze_sample(f, use_yamnet=False)
                if "Error" not in m:
                    out.append(m)
        return out
    return []
