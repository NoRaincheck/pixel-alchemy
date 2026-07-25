from pathlib import Path

from pixel_alchemy.generation.sd_cli import edit, generate, inpaint

DM = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/flux-2-klein-9b-Q4_0.gguf"
VAE = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/flux2_dev_diffusion_pytorch_model.safetensors"
LLM = "/Users/crn/.local/share/stable-diffusion.cpp/build/models/Qwen3-8B-Q3_K_M.gguf"

# Generate
generate(
    "A lovely anime orange cat",
    diffusion_model=DM,
    vae=VAE,
    llm=LLM,
    output="orange-cat.png",
)

# # Edit
# edit(
#     reference=Path(__file__).with_name("hello-world.jpg"),
#     prompt="change to an orange cat",
#     diffusion_model=DM,
#     vae=VAE,
#     llm=LLM,
#     output="orange-cat-edited.png",
# )

# # Inpaint
# inpaint(
#     init_image="dog-bench.png",
#     mask="dog-bench-mask.png",
#     prompt="a lovely dog",
#     diffusion_model=DM,
#     vae=VAE,
#     llm=LLM,
#     output="dog-bench-inpainted.png",
# )
