"""Moodist Python port — combine samples/sounds into layered wav.

Mirrors the JS app's sound-store + Howler mixing, now in Python:
- catalog lists 9 categories / 91 sounds from samples/sounds/
- SoundStore selects/volumes/shuffle (port of moodist/src/stores/sound.ts)
- mixer loops + fades + peak-normalizes to a single wav
- MoodPreset enum gives seeded preset variations matching the frontend moods
"""

from pathlib import Path

from pixel_alchemy.moodist.catalog import CATEGORIES, validate
from pixel_alchemy.moodist.mixer import mix
from pixel_alchemy.moodist.presets import MoodPreset, PresetStore, preset_mix
from pixel_alchemy.moodist.store import SoundStore

out_dir = Path(__file__).parent
missing = validate()
assert not missing, f"missing samples: {missing[:3]}"

print(f"catalog: {len(CATEGORIES)} categories")
for cat in CATEGORIES:
    print(f"  {cat.id:10s} {cat.title:16s} {len(cat.sounds)} sounds")

# 1) explicit mix — rainy forest focus
store = SoundStore()
store.override({"light-rain": 0.6, "river": 0.4, "birds": 0.3, "brown-noise": 0.2})
out1 = out_dir / "moodist_focus.wav"
mix(store.selected(), out1, duration=10, fade_in=1, fade_out=1)
print(f"wrote {out1} ({out1.stat().st_size} bytes) from {store.selected()}")

# 2) shuffle mix — 4 random sounds like Moodist shuffle button
store2 = SoundStore()
store2.shuffle()
out2 = out_dir / "moodist_shuffle.wav"
mix(store2.selected(), out2, duration=10)
print(f"wrote {out2} ({out2.stat().st_size} bytes) from {store2.selected()}")

# 3) preset save/restore — port of moodist/src/stores/preset.ts
presets = PresetStore()
presets.add("rainy cafe", {"light-rain": 0.7, "cafe": 0.5})
presets.add("deep work", store.selected())
print(f"presets: {[(p.label, p.sounds) for p in presets.presets]}")
# reapply first preset
store.override(presets.presets[0].sounds)
out3 = out_dir / "moodist_preset.wav"
mix(store.selected(), out3, duration=10)
print(f"wrote {out3} from preset '{presets.presets[0].label}'")

# 4) builtin seeded presets — Enum (human-readable category) with seed for variations
# same preset + same seed => same mix; different seed => variation (like frontend presets but seedable)
print("\nbuiltin presets (seeded variations):")
for preset in [MoodPreset.RAIN, MoodPreset.FOREST, MoodPreset.CAFE, MoodPreset.FOCUS]:
    print(f"  {preset.value}: pool={preset_mix(preset, seed=0).keys()}")
for seed in [0, 1, 2]:
    sounds = preset_mix(MoodPreset.RAIN, seed=seed)
    out = out_dir / f"moodist_builtin_rain_s{seed}.wav"
    mix(sounds, out, duration=8)
    print(f"wrote {out.name} seed={seed} {sounds}")
# enum is string-comparable and human readable
assert MoodPreset.RAIN == "Rainy Day"
assert MoodPreset.RAIN.value == "Rainy Day"
print(f"all presets: {[p.value for p in MoodPreset]}")
