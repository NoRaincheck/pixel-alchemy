"""Popular chord progressions for ambient/incidental generation.

Degrees are scale-relative (1-indexed). Supports major/minor context;
generator resolves to absolute pitch classes at runtime.
"""

PROGRESSIONS: list[dict] = [
    # --- classic pop ---
    {"name": "pop_axis", "degrees": [1, 5, 6, 4], "label": "I-V-vi-IV"},
    {"name": "pop_axis_variant", "degrees": [6, 4, 1, 5], "label": "vi-IV-I-V"},
    {"name": "doo_wop", "degrees": [1, 6, 4, 5], "label": "I-vi-IV-V"},
    {"name": "doo_wop_minor", "degrees": [1, 6, 3, 7], "label": "i-VI-III-VII (minor)"},
    {"name": "fifty_progression", "degrees": [1, 6, 2, 5], "label": "I-vi-ii-V"},
    {"name": "canon", "degrees": [1, 5, 6, 3, 4, 1, 4, 5], "label": "Pachelbel I-V-vi-iii-IV-I-IV-V"},
    {"name": "beautiful_pop", "degrees": [1, 5, 6, 4, 1, 5, 4, 5], "label": "I-V-vi-IV I-V-IV-V"},
    # --- anthemic / uplifting ---
    {"name": "anthem", "degrees": [4, 1, 5, 6], "label": "IV-I-V-vi"},
    {"name": "uplift", "degrees": [1, 4, 6, 5], "label": "I-IV-vi-V"},
    {"name": "golden_hour", "degrees": [1, 4, 5, 4], "label": "I-IV-V-IV"},
    {"name": "open_sky", "degrees": [1, 5, 4, 5], "label": "I-V-IV-V"},
    {"name": "rise", "degrees": [6, 5, 4, 5], "label": "vi-V-IV-V"},
    # --- melancholic / ambient ---
    {"name": "ambient_minor", "degrees": [6, 4, 1, 5], "label": "vi-IV-I-V (relative minor feel)"},
    {"name": "lament", "degrees": [1, 3, 6, 4], "label": "I-iii-vi-IV"},
    {"name": "fragile", "degrees": [1, 6, 4, 1], "label": "I-vi-IV-I"},
    {"name": "bittersweet", "degrees": [6, 1, 4, 5], "label": "vi-I-IV-V"},
    {"name": "sorrow", "degrees": [6, 5, 4, 1], "label": "vi-V-IV-I"},
    {"name": "wistful", "degrees": [4, 5, 6, 1], "label": "IV-V-vi-I"},
    {"name": "nostalgia", "degrees": [1, 4, 2, 5], "label": "I-IV-ii-V"},
    {"name": "stillness", "degrees": [1, 4, 6, 6], "label": "I-IV-vi-vi (drone)"},
    {"name": "hush", "degrees": [6, 4, 6, 5], "label": "vi-IV-vi-V"},
    # --- jazz / neo-soul / lofi ---
    {"name": "two_five_one", "degrees": [2, 5, 1, 1], "label": "ii-V-I-I"},
    {"name": "two_five_one_minor", "degrees": [2, 5, 1, 6], "label": "ii-V-i-vi"},
    {"name": "rhythm_changes", "degrees": [1, 6, 2, 5], "label": "I-vi-ii-V (rhythm)"},
    {"name": "turnaround", "degrees": [1, 6, 2, 5, 1], "label": "I-vi-ii-V-I"},
    {"name": "neo_soul", "degrees": [2, 5, 1, 4], "label": "ii-V-I-IV"},
    {"name": "lofi_loop", "degrees": [4, 3, 6, 1], "label": "IV-iii-vi-I"},
    {"name": "maj7_wander", "degrees": [1, 4, 7, 3], "label": "Imaj7-IVmaj7-viim7-iiim7"},
    {"name": "chromatic_mediant", "degrees": [1, 6, 4, 6], "label": "I-vi-IV-vi (mediant)"},
    {"name": "cyclic", "degrees": [1, 4, 7, 3, 6, 2, 5, 1], "label": "cycle of 4ths"},
    # --- film / ambient / drone ---
    {"name": "film_minor", "degrees": [6, 5, 6, 4], "label": "vi-V-vi-IV"},
    {"name": "einaudi", "degrees": [1, 6, 4, 5, 1, 6, 4, 1], "label": "I-vi-IV-V I-vi-IV-I"},
    {"name": "glass", "degrees": [6, 4, 1, 1], "label": "vi-IV-I (held)"},
    {"name": "arvo", "degrees": [1, 5, 4, 6], "label": "I-V-IV-vi (tintinnabuli)"},
    {"name": "satie", "degrees": [1, 2, 4, 1], "label": "I-ii-IV-I (gymnopédie)"},
    {"name": "nils_frahm", "degrees": [1, 5, 6, 5, 4, 1, 6, 5], "label": "I-V-vi-V-IV-I-vi-V"},
    {"name": "max_richter", "degrees": [6, 1, 5, 4], "label": "vi-I-V-IV"},
    {"name": "olafur", "degrees": [1, 3, 4, 6], "label": "I-iii-IV-vi"},
    {"name": "ambient_drift", "degrees": [4, 1, 6, 5, 4, 1], "label": "IV-I-vi-V-IV-I"},
    {"name": "longing", "degrees": [1, 5, 4, 6, 1, 5, 4, 4], "label": "I-V-IV-vi long hold"},
    # --- extra colours ---
    {"name": "andalusian", "degrees": [1, 7, 6, 5], "label": "i-VII-VI-V (phrygian)"},
    {"name": "royal_road", "degrees": [4, 5, 3, 6], "label": "IV-V-iii-vi (J-pop)"},
    {"name": "royal_road_variant", "degrees": [4, 5, 3, 6, 4, 5, 1], "label": "IV-V-iii-vi IV-V-I"},
    {"name": "sensitive_female", "degrees": [6, 4, 1, 5], "label": "vi-IV-I-V (sensitive)"},
    {"name": "epic", "degrees": [6, 4, 1, 5, 6, 4, 5], "label": "vi-IV-I-V-vi-IV-V"},
    {"name": "descending_bass", "degrees": [1, 5, 6, 4, 2, 5, 1], "label": "I-V-vi-IV-ii-V-I (stepwise)"},
    {"name": "one_four_five", "degrees": [1, 4, 5, 1], "label": "I-IV-V-I"},
    {"name": "minor_plagal", "degrees": [1, 4, 6, 5], "label": "i-iv-VI-V"},
    {"name": "lullaby", "degrees": [1, 5, 6, 3, 4, 1, 2, 5], "label": "I-V-vi-iii-IV-I-ii-V"},
]

# quick index
BY_NAME = {p["name"]: p for p in PROGRESSIONS}
