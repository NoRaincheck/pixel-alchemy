"""ACE-Step 1.5 via audio.cpp (Metal) — separate from vendored pipeline.

This backend shells out to ``audiocpp_cli`` (Metal) and never touches
ComfyUI Python. It supports the same ``generate`` signature as
``pipeline.generate`` but routes through audio.cpp ``text2music`` etc.,
and can convert the Comfy safetensors layout to GGUF via
``audiocpp_gguf`` per https://github.com/0xShug0/audio.cpp/blob/main/docs/models/ace_step.md

Requires the prebuilt Metal binaries (v0.7.2) at one of:
  - ~/.local/bin/audiocpp_cli
  - /tmp/audiocpp/audiocpp_cli
  - $AUDIOCPP_CLI

Model resolution order:
  1. $ACE_STEP_GGUF or $ACE_STEP_MODEL env
  2. ./models/ACE-Step1.5-GGUF  (turbo/xl-turbo)
  3. ~/.cache/audio.cpp/ACE-Step1.5-GGUF
  4. HuggingFace GGUF (auto-download to cache on first use)

See ``convert_from_comfy`` for local safetensors → GGUF.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ------------------------------------------------------------------ paths

def _find_cli(name: str = "audiocpp_cli") -> Path | None:
    for cand in [
        os.environ.get("AUDIOCPP_CLI"),
        str(Path.home() / ".local" / "bin" / name),
        "/tmp/audiocpp/audiocpp_cli" if name == "audiocpp_cli" else "/tmp/audiocpp/audiocpp_gguf",
        shutil.which(name),
    ]:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _find_gguf_cli() -> Path | None:
    return _find_cli("audiocpp_gguf")


def _default_gguf_root() -> Path:
    # check env first
    for env in ["ACE_STEP_GGUF", "ACE_STEP_MODEL"]:
        p = os.environ.get(env)
        if p and Path(p).exists():
            return Path(p)
    for cand in [
        Path.cwd() / "models" / "ACE-Step1.5-GGUF",
        Path.cwd() / "models" / "Ace-Step1.5",
        Path.home() / ".cache" / "audio.cpp" / "ACE-Step1.5-GGUF",
        Path.home() / "Documents" / "comfy" / "ComfyUI" / "models" / "ACE-Step1.5-GGUF",
    ]:
        if cand.exists():
            return cand
    return Path.home() / ".cache" / "audio.cpp" / "ACE-Step1.5-GGUF"


def _resolve_gguf(variant: str = "turbo") -> Path | None:
    """Find GGUF for variant: turbo, base, xl-turbo, xl-sft."""
    root = _default_gguf_root()
    # direct GGUF file env
    env = os.environ.get("ACE_STEP_GGUF")
    if env and Path(env).is_file():
        return Path(env)
    # layout ACE-Step1.5-GGUF/<variant>/ace-step-1.5-<variant>-*.gguf
    variant_map = {
        "turbo": ["turbo/ace-step-1.5-turbo-bf16.gguf", "turbo/ace-step-1.5-turbo-q8_0.gguf"],
        "base": ["base/ace-step-1.5-base-bf16.gguf", "base/ace-step-1.5-base-q8_0.gguf"],
        "xl-turbo": ["xl-turbo/ace-step-1.5-xl-turbo-bf16.gguf", "xl-sft/ace-step-1.5-xl-turbo-bf16.gguf", "xl-turbo/ace-step-1.5-xl-turbo-q8_0.gguf"],
        "xl-sft": ["xl-sft/ace-step-1.5-xl-sft-bf16.gguf"],
    }
    for rel in variant_map.get(variant, variant_map["turbo"]):
        p = root / rel
        if p.exists():
            return p
    # also allow root itself as model dir (safetensors tree)
    if (root / "acestep-v15-turbo" / "config.json").exists():
        return root
    # fallback: any gguf under root
    if root.exists():
        for gg in root.rglob("*.gguf"):
            if variant in gg.name or "turbo" in gg.name:
                return gg
    return None


# ------------------------------------------------------------------ install helper

def ensure_cli() -> Path:
    cli = _find_cli()
    if cli is None or not cli.exists():
        raise FileNotFoundError(
            "audiocpp_cli not found. Install Metal build:\n"
            "  curl -L -o /tmp/audio-metal.tar.gz "
            "https://github.com/0xShug0/audio.cpp/releases/download/v0.7.2/audio-v0.7.2-bin-macos-arm64-metal.tar.gz\n"
            "  tar -xzf /tmp/audio-metal.tar.gz -C /tmp/audiocpp --strip-components=1\n"
            "  mkdir -p ~/.local/bin && cp /tmp/audiocpp/audiocpp_* ~/.local/bin/"
        )
    return cli


# ------------------------------------------------------------------ GGUF conversion (Comfy -> GGUF per docs)

def convert_from_comfy(
    comfy_root: str | Path | None = None,
    output: str | Path | None = None,
    variant: str = "xl-turbo",
    gguf_type: str = "bf16",
) -> Path:
    """Convert Comfy safetensors layout to GGUF via audiocpp_gguf.

    This stages the flat Comfy files into the expected
    ``models/Ace-Step1.5`` tree and invokes the exact command from
    docs/models/ace_step.md. Requires the *full* upstream Ace-Step1.5
    tree for XL (turbo/base + lm + text_encoder + vae + silence_latents).

    For the common case (single Comfy file: acestep_v1.5_xl_turbo),
    this will error with a clear message and suggest downloading the
    prebuilt GGUF instead.

    Returns output GGUF path on success.
    """
    gguf_cli = _find_gguf_cli()
    if gguf_cli is None:
        raise FileNotFoundError("audiocpp_gguf not found — install Metal bundle first")
    if comfy_root is None:
        from .pipeline import _comfy_root  # reuse

        comfy_root = _comfy_root()
    if comfy_root is None:
        raise FileNotFoundError("ComfyUI root not found")
    comfy_root = Path(comfy_root)

    # The docs expect models/Ace-Step1.5 with subdirs. Comfy's flat files
    # are insufficient alone — try to locate an upstream tree.
    upstream = None
    for cand in [
        Path.cwd() / "models" / "Ace-Step1.5",
        Path.cwd() / "models" / "ACE-Step1.5",
        Path.home() / ".cache" / "audio.cpp" / "Ace-Step1.5",
        Path("/tmp/Ace-Step1.5"),
    ]:
        if (cand / "acestep-v15-turbo" / "model.safetensors").exists():
            upstream = cand
            break
    if upstream is None:
        raise FileNotFoundError(
            "Full Ace-Step1.5 upstream tree not found. "
            "convert_from_comfy needs models/Ace-Step1.5 with "
            "acestep-v15-turbo, acestep-v15-base, acestep-5Hz-lm-1.7B, "
            "Qwen3-Embedding-0.6B, vae + silence_latents. "
            "Your Comfy install only has flat files:\n"
            f"  {comfy_root}/models/diffusion_models/acestep_v1.5_xl_turbo_bf16.safetensors\n"
            f"  {comfy_root}/models/vae/ace_1.5_vae.safetensors\n"
            f"  {comfy_root}/models/clip/qwen_*.safetensors\n"
            "Either download the upstream tree (hf: ACE-Step/ACE-Step-v1.5) "
            "or use the prebuilt GGUF: ensure_gguf('turbo')."
        )

    if output is None:
        output = Path.cwd() / f"ace-step-1.5-{variant}-{gguf_type}.gguf"
    output = Path(output)

    # Build command per docs (XL example); adjust inputs for variant
    # docs: audiocpp_gguf --root models/Ace-Step1.5 --family ace_step --input ... --exclude-prefix ... --type bf16 --output ...
    args = [
        str(gguf_cli),
        "--root", str(upstream),
        "--family", "ace_step",
        "--input", f"dit_turbo_weights={upstream}/acestep-v15-turbo/model.safetensors",
        "--input", f"dit_turbo_silence_latent={upstream}/acestep-v15-turbo/silence_latent.safetensors",
        "--input", f"dit_base_weights={upstream}/acestep-v15-base/model.safetensors",
        "--input", f"dit_base_silence_latent={upstream}/acestep-v15-base/silence_latent.safetensors",
        "--input", f"lm_weights={upstream}/acestep-5Hz-lm-1.7B/model.safetensors",
        "--input", f"text_encoder_weights={upstream}/Qwen3-Embedding-0.6B/model.safetensors",
        "--input", f"vae_weights={upstream}/vae/diffusion_pytorch_model.safetensors",
        "--type", gguf_type,
        "--output", str(output),
    ]
    # XL adds extra inputs; docs exclude prefixes to isolate output
    if variant in ("xl-turbo", "xl-sft"):
        key = "dit_xl_turbo" if variant == "xl-turbo" else "dit_xl_sft"
        args.extend([
            "--input", f"{key}_weights={upstream}/acestep-v15-{variant}/model.safetensors.index.json",
            "--input", f"{key}_silence_latent={upstream}/acestep-v15-{variant}/silence_latent.safetensors",
            "--exclude-prefix", "dit_turbo_",
            "--exclude-prefix", "dit_base_",
        ])
        if variant == "xl-turbo":
            args.extend(["--exclude-prefix", "dit_xl_sft_"])
        else:
            args.extend(["--exclude-prefix", "dit_xl_turbo_"])
    else:
        # for turbo/base, exclude XL prefixes if present
        args.extend(["--exclude-prefix", "dit_xl_turbo_", "--exclude-prefix", "dit_xl_sft_"])

    print(f"[audio.cpp] { ' '.join(args)}", file=sys.stderr)
    subprocess.run(args, check=True)
    return output


def ensure_gguf(variant: str = "turbo", dtype: str = "bf16") -> Path:
    """Ensure GGUF exists, downloading prebuilt if needed (single-file)."""
    gguf = _resolve_gguf(variant)
    if gguf is not None and gguf.exists():
        return gguf
    # auto-download single file (much faster than full snapshot)
    try:
        from huggingface_hub import hf_hub_download

        repo = "audio-cpp/audio.cpp-gguf" if variant in ("turbo", "base") else "CaptainArni/audio.cpp-gguf"
        # pick smallest by default: q8_0 if bf16 fails or dtype==q8_0
        dtype = dtype or "bf16"
        candidates = {
            "turbo": [f"ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-{dtype}.gguf", "ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-q8_0.gguf", "ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-bf16.gguf"],
            "base": [f"ACE-Step1.5-GGUF/base/ace-step-1.5-base-{dtype}.gguf"],
            "xl-turbo": [f"ACE-Step1.5-GGUF/xl-turbo/ace-step-1.5-xl-turbo-{dtype}.gguf", "ACE-Step1.5-GGUF/xl-turbo/ace-step-1.5-xl-turbo-bf16.gguf"],
            "xl-sft": [f"ACE-Step1.5-GGUF/xl-sft/ace-step-1.5-xl-sft-{dtype}.gguf"],
        }.get(variant, [f"ACE-Step1.5-GGUF/turbo/ace-step-1.5-turbo-{dtype}.gguf"])
        cache = Path.home() / ".cache" / "audio.cpp" / "ACE-Step1.5-GGUF"
        cache.mkdir(parents=True, exist_ok=True)
        last_err = None
        for fname in candidates:
            try:
                print(f"[audio.cpp] downloading {fname} from {repo} ...", file=sys.stderr)
                local = hf_hub_download(repo_id=repo, filename=fname, local_dir=str(cache), local_dir_use_symlinks=False)
                # hf_hub_download with local_dir keeps repo prefix, resolve
                gguf = _resolve_gguf(variant)
                if gguf and gguf.exists():
                    return gguf
                p = cache / fname
                if p.exists():
                    return p
                if Path(local).exists():
                    return Path(local)
            except Exception as e:
                last_err = e
                continue
        if last_err:
            raise last_err
    except Exception as e:
        print(f"[audio.cpp] HF download failed: {e}", file=sys.stderr)
    raise FileNotFoundError(
        f"GGUF for {variant} ({dtype}) not found at {_default_gguf_root()}. "
        "Set $ACE_STEP_GGUF to a .gguf path or run convert_from_comfy().\n"
        "For XL you need CaptainArni/audio.cpp-gguf (14.2 GB) — "
        "try: hf download CaptainArni/audio.cpp-gguf ACE-Step1.5-GGUF/xl-turbo/ace-step-1.5-xl-turbo-bf16.gguf --local-dir ~/.cache/audio.cpp/ACE-Step1.5-GGUF"
    )


# ------------------------------------------------------------------ CLI wrapper (text2music etc.)

def _run_cli(args: list[str], log: bool = False) -> None:
    cli = ensure_cli()
    cmd = [str(cli)] + args
    print(f"[audio.cpp] {' '.join(cmd)}", file=sys.stderr)
    # audiocpp_cli writes progress to stderr; capture for log
    result = subprocess.run(cmd, capture_output=not log, text=True)
    if result.returncode != 0:
        msg = result.stderr if result.stderr else result.stdout
        raise RuntimeError(f"audiocpp_cli failed ({result.returncode}): {msg[:2000]}")


def generate(
    prompt: str,
    lyrics: str = "",
    output: str | Path | None = None,
    *,
    duration: float = 30,
    variant: str = "turbo",
    task_route: str = "text2music",
    model: str | Path | None = None,
    bpm: int | None = None,
    keyscale: str | None = None,
    timesignature: str | None = None,
    language: str = "en",
    reference_audio: str | Path | None = None,
    audio_codes: str | None = None,
    guidance_scale: float | None = None,
    num_inference_steps: int = 8,
    seed: int | None = None,
    backend: str = "metal",
    extra_args: list[str] | None = None,
) -> Path:
    """Generate via audio.cpp Metal (bypasses Comfy).

    Mirrors ``pipeline.generate`` but calls ``audiocpp_cli --backend metal``.

    Args:
        prompt: --text
        lyrics: --lyrics
        output: --out (default ace_output.wav)
        duration: --duration-seconds (-1 = auto)
        variant: dit_model_path: turbo, base, xl-turbo, xl-sft
        task_route: text2music, complete, lego, extract, cover, cover-nofsq, repaint
        model: path to GGUF or Ace-Step1.5 dir; auto-resolved if None
        bpm, keyscale, timesignature, language: --request-option
        reference_audio: --audio (required for lego/extract/cover)
        audio_codes: --request-option audio_codes=...
        guidance_scale: --guidance-scale
        num_inference_steps: --num-inference-steps
        seed: --seed
        backend: metal (mac) or cpu/cuda/hip
        extra_args: raw passthrough

    Returns output Path.
    """
    if output is None:
        output = Path("ace_output.wav")
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    if model is None:
        try:
            model = ensure_gguf(variant)
        except FileNotFoundError:
            # fallback to any resolved
            model = _resolve_gguf(variant) or _default_gguf_root()
    model = Path(model)

    args = [
        "--task", "gen",
        "--family", "ace_step",
        "--model", str(model),
        "--backend", backend,
        "--task-route", task_route,
        "--text", prompt,
        "--lyrics", lyrics or "",
        "--duration-seconds", str(duration),
        "--language", language,
        "--out", str(output),
        "--num-inference-steps", str(num_inference_steps),
    ]
    # dit selection per docs
    if variant != "turbo":
        args += ["--load-option", f"ace_step.dit_model_path=acestep-v15-{variant}"]
    if guidance_scale is not None:
        args += ["--guidance-scale", str(guidance_scale)]
    if seed is not None:
        args += ["--seed", str(seed)]
    if reference_audio is not None:
        args += ["--audio", str(reference_audio)]
    # metadata
    if bpm is not None:
        args += ["--request-option", f"bpm={bpm}"]
    if keyscale is not None:
        args += ["--request-option", f"keyscale={keyscale}"]
    if timesignature is not None:
        # docs strip /4
        ts = str(timesignature).replace("/4", "")
        args += ["--request-option", f"timesignature={ts}"]
    if audio_codes is not None:
        args += ["--request-option", f"audio_codes={audio_codes}"]
    if extra_args:
        args += extra_args

    _run_cli(args)
    if not output.exists():
        raise FileNotFoundError(f"audio.cpp did not produce {output}")
    return output


def generate_with_reference(*a, **kw) -> Path:
    return generate(*a, **kw, task_route=kw.pop("task_route", "cover"))
