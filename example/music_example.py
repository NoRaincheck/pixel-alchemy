from pixel_alchemy.music.pipeline import generate, generate_with_reference
from pixel_alchemy.vinyl.simulator import process_file

# ComfyUI model layout (auto-discovered if paths are None):
#   models/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors  (~9.3G)
#   models/vae/ace_1.5_vae.safetensors                               (~322M)
#   models/text_encoders/qwen_0.6b_ace15.safetensors  (optional, dummy fallback)
#   models/text_encoders/qwen_4b_ace15.safetensors    (optional, for audio codes)
#
# Explicit paths (like sd_cli_example.py) – replace with your local files:

DIFFUSION = "/Users/crn/Documents/comfy/ComfyUI/models/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors"
VAE = "/Users/crn/Documents/comfy/ComfyUI/models/vae/ace_1.5_vae.safetensors"
# TEXT_ENCODER = "/Users/crn/Documents/comfy/ComfyUI/models/clip/qwen_0.6b_ace15.safetensors"  # 1.1G
# LLM = "/Users/crn/Documents/comfy/ComfyUI/models/clip/qwen_4b_ace15.safetensors"  # 7.8G, for audio codes

# --- basic text-to-music (explicit ComfyUI paths, like sd_cli_example.py) ---
generate(
    prompt="slow atmospheric piano and strings playiing. melancholy, sad, intimate. No lyrics, no vocals, no choir",
    lyrics="",
    diffusion_model=DIFFUSION,
    vae=VAE,
    output="ace_output_instrumental.wav",
    duration=30,
    steps=25,
    cfg_scale=5.0,
    seed=0,
)

# --- post-process: apply the vinyl affect to the instrumental ---
# presets: digital, audiophile, fresh-press, well-played, worn-classic, lo-fi
process_file(
    "ace_output_instrumental.wav",
    "ace_output_instrumental_vinyl.wav",
    preset="lo-fi",
    seed=0,
)
