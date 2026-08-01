# Transcription Recipe

Transcribe Chinese audio with speaker diarization and English translation.

## Prerequisites

- MOSS-Transcribe-Diarize model: `~/.local/share/transcribe.cpp/models/MOSS-Transcribe-Diarize-Q8_0.gguf`
- transcribe-cli CLI tool on PATH
- Local chat completions endpoint (default: `http://localhost:1234/v1/chat/completions`)

## Usage

```python
from pixel_alchemy.transcribe.chinese_transcribe import transcribe_and_translate

result = transcribe_and_translate(
    "meeting.wav",
    model_path="models/MOSS-Transcribe-Diarize-Q8_0.gguf",
)
for seg in result["segments"]:
    print(f"{seg['speaker']}: {seg['text']} -> {seg['translated_text']}")
```

## Batch Processing

Use `spaces/transcribe_all.py` to process all OGG files:

```bash
cd /Users/crn/dev/projects/pixel-alchemy/spaces
python3 transcribe_all.py
```

This splits long audio into chunks, transcribes each chunk with diarization,
translates segments to English, and saves results as `<name>.json`.

## Output Format

Each JSON result contains:

- `full_text`: Plain text without diarization markers
- `segments`: List of dicts with keys:
  - `start`, `end`: Timestamps in seconds
  - `speaker`: Speaker identifier (S0, S1, ...)
  - `text`: Transcribed Chinese text
  - `translated_text`: English translation

