"""Image generation via sd-cli (stable-diffusion.cpp).

Supported diffusion-model / VAE / LLM combinations:

+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+
| Profile   | Diffusion Model                              | VAE                                          | LLM (text encoder)                           |
+===========+==============================================+==============================================+==============================================+
| ZiT       | z_image_turbo-Q3_K.gguf                      | flux1_schnell_diffusion_pytorch_model.safetensors | Qwen3-4B-Instruct-2507-Q4_K_M.gguf        |
+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+
| Flux 2    | flux-2-klein-9b-Q4_0.gguf                    | flux2_dev_diffusion_pytorch_model.safetensors | Qwen3-8B-Q3_K_M.gguf                        |
+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+
| Bonsai    | bonsai_image_4b-mod_q8_0-q1_0.gguf           | flux2_dev_diffusion_pytorch_model.safetensors | Qwen3-4B-Instruct-2507-Q4_K_M.gguf        |
+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+
| Krea 2    | Krea-2-Turbo-Q8_0.gguf                       | wan_2.1_vae.safetensors                      | Qwen3VL-4B-Instruct-Q4_K_M.gguf             |
+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+
| Wan 2.2   | FastWan2.2-TI2V-5B-q8_0.gguf                 | (uses --tae / --vae-conv-direct)             | umt5-xxl-encoder-Q8_0.gguf (t5xxl)          |
+-----------+----------------------------------------------+----------------------------------------------+----------------------------------------------+

Notes:
- Wan 2.2 has no Metal support yet and requires ``-M vid_gen`` (video mode).
- Bonsai is a 1-bit quantised Flux variant; use ``--vae-tiling`` for large images.
- All profiles support generation, editing (``-r``), and inpainting (``--init-img`` + ``--mask``) except Wan 2.2 (video only).
- ``--cfg-scale`` should be kept at 1.0 — higher values cause artifacts with sd-cli.
- ``--diffusion-fa`` (flash attention) and ``--offload-to-cpu`` are enabled by default in this module.

Example (Flux 2, generation)::

    from pixel_alchemy.generation.sd_cli import generate

    generate(
        "A lovely anime orange cat",
        diffusion_model="/path/to/flux-2-klein-9b-Q4_0.gguf",
        vae="/path/to/flux2_dev_diffusion_pytorch_model.safetensors",
        llm="/path/to/Qwen3-8B-Q3_K_M.gguf",
        output="orange-cat.png",
    )

Example (ZiT, generation)::

    generate(
        "A cinematic, melancholic photograph...",
        diffusion_model="/path/to/z_image_turbo-Q3_K.gguf",
        vae="/path/to/flux1_schnell_diffusion_pytorch_model.safetensors",
        llm="/path/to/Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
        output="zi-output.png",
        steps=8,
        cfg_scale=1.0,
    )

Example (Flux 2, edit)::

    from pixel_alchemy.generation.sd_cli import edit

    edit(
        reference="cat.png",
        prompt="change to an orange cat",
        diffusion_model="/path/to/flux-2-klein-9b-Q4_0.gguf",
        vae="/path/to/flux2_dev_diffusion_pytorch_model.safetensors",
        llm="/path/to/Qwen3-8B-Q3_K_M.gguf",
        output="orange-cat.png",
    )

Example (Flux 2, inpaint)::

    from pixel_alchemy.generation.sd_cli import inpaint

    inpaint(
        init_image="bench.jpg",
        mask="dog-bench-mask.png",
        prompt="a lovely dog",
        diffusion_model="/path/to/flux-2-klein-9b-Q4_0.gguf",
        vae="/path/to/flux2_dev_diffusion_pytorch_model.safetensors",
        llm="/path/to/Qwen3-8B-Q3_K_M.gguf",
        output="dog-lovely-bench.png",
    )
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def _find_sd_cli() -> Path:
    binary = shutil.which("sd-cli")
    if binary is None:
        raise FileNotFoundError("sd-cli not found on PATH")
    return Path(binary)


def generate(
    prompt: str,
    diffusion_model: str | Path,
    vae: str | Path,
    llm: str | Path,
    output: str | Path | None = None,
    *,
    width: int = 512,
    height: int = 512,
    steps: int = 8,
    cfg_scale: float = 1.0,
    sampling_method: str = "euler",
    offload_to_cpu: bool = True,
    diffusion_fa: bool = True,
    verbose: bool = False,
) -> Path:
    """Generate an image using sd-cli with Flux 2.

    Args:
        prompt: Text prompt for generation.
        diffusion_model: Path to the diffusion model .gguf file.
        vae: Path to the VAE .safetensors file.
        llm: Path to the LLM .gguf file.
        output: Where to write the result. Defaults to <cwd>/generated.png.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale. Must be 1.0 — higher values cause artifacts with sd-cli (see https://github.com/leejet/stable-diffusion.cpp/issues/1309).
        sampling_method: Sampling method (e.g. "euler", "dpm++2s_a").
        offload_to_cpu: Offload computation to CPU.
        diffusion_fa: Enable flash attention for diffusion.
        verbose: Enable verbose output.

    Returns:
        The resolved output path as a pathlib.Path.
    """
    diffusion_model = Path(diffusion_model)
    vae = Path(vae)
    llm = Path(llm)

    for p, label in [(diffusion_model, "diffusion_model"), (vae, "vae"), (llm, "llm")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    if output is None:
        output = Path("generated.png")
    else:
        output = Path(output)

    args: list[str] = [
        str(_find_sd_cli()),
        "--diffusion-model", str(diffusion_model),
        "--vae", str(vae),
        "--llm", str(llm),
        "-p", prompt,
        "--output", str(output),
        "-H", str(height),
        "-W", str(width),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampling_method,
    ]
    if offload_to_cpu:
        args.append("--offload-to-cpu")
    if diffusion_fa:
        args.append("--diffusion-fa")
    if verbose:
        args.append("-v")

    subprocess.run(args, check=True)
    return output


def edit(
    reference: str | Path,
    prompt: str,
    diffusion_model: str | Path,
    vae: str | Path,
    llm: str | Path,
    output: str | Path | None = None,
    *,
    width: int = 512,
    height: int = 512,
    steps: int = 4,
    cfg_scale: float = 1.0,
    sampling_method: str = "euler",
    offload_to_cpu: bool = True,
    diffusion_fa: bool = True,
    verbose: bool = False,
) -> Path:
    """Edit an image using sd-cli with Flux 2.

    Args:
        reference: Path to the reference image to edit.
        prompt: Text prompt describing the edit.
        diffusion_model: Path to the diffusion model .gguf file.
        vae: Path to the VAE .safetensors file.
        llm: Path to the LLM .gguf file.
        output: Where to write the result. Defaults to <cwd>/edited.png.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale. Must be 1.0 — higher values cause artifacts with sd-cli (see https://github.com/leejet/stable-diffusion.cpp/issues/1309).
        sampling_method: Sampling method (e.g. "euler").
        offload_to_cpu: Offload computation to CPU.
        diffusion_fa: Enable flash attention for diffusion.
        verbose: Enable verbose output.

    Returns:
        The resolved output path as a pathlib.Path.
    """
    reference = Path(reference)
    diffusion_model = Path(diffusion_model)
    vae = Path(vae)
    llm = Path(llm)

    for p, label in [
        (reference, "reference"),
        (diffusion_model, "diffusion_model"),
        (vae, "vae"),
        (llm, "llm"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    if output is None:
        output = Path("edited.png")
    else:
        output = Path(output)

    args: list[str] = [
        str(_find_sd_cli()),
        "--diffusion-model", str(diffusion_model),
        "--vae", str(vae),
        "--llm", str(llm),
        "-r", str(reference),
        "-p", prompt,
        "--output", str(output),
        "-H", str(height),
        "-W", str(width),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampling_method,
    ]
    if offload_to_cpu:
        args.append("--offload-to-cpu")
    if diffusion_fa:
        args.append("--diffusion-fa")
    if verbose:
        args.append("-v")

    subprocess.run(args, check=True)
    return output


def inpaint(
    init_image: str | Path,
    mask: str | Path,
    prompt: str,
    diffusion_model: str | Path,
    vae: str | Path,
    llm: str | Path,
    output: str | Path | None = None,
    *,
    width: int = 512,
    height: int = 512,
    steps: int = 9,
    cfg_scale: float = 1.0,
    sampling_method: str = "euler",
    threads: int = 24,
    color: bool = True,
    vae_tiling: bool = True,
    vae_tile_overlap: float = 0.125,
    offload_to_cpu: bool = True,
    verbose: bool = False,
) -> Path:
    """Inpaint an image using sd-cli with Flux 2.

    Args:
        init_image: Path to the initial image.
        mask: Path to the mask image (white = inpaint area).
        prompt: Text prompt for the inpainted region.
        diffusion_model: Path to the diffusion model .gguf file.
        vae: Path to the VAE .safetensors file.
        llm: Path to the LLM .gguf file.
        output: Where to write the result. Defaults to <cwd>/inpainted.png.
        width: Image width in pixels.
        height: Image height in pixels.
        steps: Number of sampling steps.
        cfg_scale: Classifier-free guidance scale. Must be 1.0 — higher values cause artifacts with sd-cli (see https://github.com/leejet/stable-diffusion.cpp/issues/1309).
        sampling_method: Sampling method (e.g. "euler").
        threads: Number of CPU threads.
        color: Enable color correction.
        vae_tiling: Enable VAE tiling for large images.
        vae_tile_overlap: VAE tile overlap ratio.
        offload_to_cpu: Offload computation to CPU.
        verbose: Enable verbose output.

    Returns:
        The resolved output path as a pathlib.Path.
    """
    init_image = Path(init_image)
    mask = Path(mask)
    diffusion_model = Path(diffusion_model)
    vae = Path(vae)
    llm = Path(llm)

    for p, label in [
        (init_image, "init_image"),
        (mask, "mask"),
        (diffusion_model, "diffusion_model"),
        (vae, "vae"),
        (llm, "llm"),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    if output is None:
        output = Path("inpainted.png")
    else:
        output = Path(output)

    args: list[str] = [
        str(_find_sd_cli()),
        "--diffusion-model", str(diffusion_model),
        "--vae", str(vae),
        "--llm", str(llm),
        "--init-img", str(init_image),
        "--mask", str(mask),
        "-p", prompt,
        "--output", str(output),
        "-H", str(height),
        "-W", str(width),
        "--steps", str(steps),
        "--cfg-scale", str(cfg_scale),
        "--sampling-method", sampling_method,
        "-t", str(threads),
    ]
    if color:
        args.append("--color")
    if vae_tiling:
        args.append("--vae-tiling")
        args += ["--vae-tile-overlap", str(vae_tile_overlap)]
    if offload_to_cpu:
        args.append("--offload-to-cpu")
    if verbose:
        args.append("-v")

    subprocess.run(args, check=True)
    return output
