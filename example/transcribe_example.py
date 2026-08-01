from pathlib import Path

from pixel_alchemy.transcribe.chinese_transcribe import transcribe_and_translate

# Process an OGG file from the spaces directory
audio_path = Path(__file__).parent / "spaces" / "2026-07-30__04-48-19-rick.ogg"

result = transcribe_and_translate(
    audio_path,
    model_path="models/MOSS-Transcribe-Diarize-Q8_0.gguf",
)

for seg in result["segments"]:
    print(f"[{seg['start']:.1f}-{seg['end']:.1f}] {seg['speaker']}: {seg['text']}")
    if "translated_text" in seg:
        print(f"  -> {seg['translated_text']}")

