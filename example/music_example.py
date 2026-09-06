from pixel_alchemy.music.pipeline import generate
from tqdm import trange


DIFFUSION = "/Users/crn/Documents/comfy/ComfyUI/models/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors"
VAE = "/Users/crn/Documents/comfy/ComfyUI/models/vae/ace_1.5_vae.safetensors"
# TEXT_ENCODER = "/Users/crn/Documents/comfy/ComfyUI/models/clip/qwen_0.6b_ace15.safetensors"  # 1.1G
# LLM = "/Users/crn/Documents/comfy/ComfyUI/models/clip/qwen_4b_ace15.safetensors"  # 7.8G, for audio codes

# --- basic text-to-music (explicit ComfyUI paths, like sd_cli_example.py) ---

for seed in trange(20):
    generate(
        prompt="Largo, slow atmospheric choral, baroque, music. Organ, Strings. cathedral reverb, vast hall, ambient space",
        lyrics="",
        diffusion_model=DIFFUSION,
        vae=VAE,
        output=f"ace_output_instrumental_{seed}.wav",
        duration=45,
        steps=20,
        cfg_scale=1.0,
        seed=seed,
        bpm=40,
    )
