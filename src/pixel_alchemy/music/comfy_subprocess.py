"""Helper to run ACE15 text encoding via ComfyUI's Python subprocess.

No triple-quote templating — args are passed via JSON to avoid injection.
"""

from __future__ import annotations

import json
import pathlib
import pickle
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

COMFY_PYTHON = "/Users/crn/Documents/comfy/ComfyUI/.venv/bin/python"
COMFY_ROOT = "/Users/crn/Documents/comfy/ComfyUI"


def _resolve_comfy_root() -> Path | None:
    import os

    for cand in [
        os.environ.get("COMFYUI_ROOT"),
        Path.home() / "Documents" / "comfy" / "ComfyUI",
        Path.home() / "comfy" / "ComfyUI",
        Path(COMFY_ROOT),
    ]:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _resolve_comfy_python() -> str:
    import os

    for cand in [
        os.environ.get("COMFY_PYTHON"),
        COMFY_PYTHON,
    ]:
        if cand and Path(cand).exists():
            return cand
    root = _resolve_comfy_root()
    if root is not None:
        p = root / ".venv" / "bin" / "python"
        if p.exists():
            return str(p)
    return COMFY_PYTHON


def _resolve_qwen_paths(comfy_root: Path) -> tuple[Path | None, Path | None]:
    for sub in ["clip", "text_encoders"]:
        c06 = comfy_root / "models" / sub / "qwen_0.6b_ace15.safetensors"
        c4b = comfy_root / "models" / sub / "qwen_4b_ace15.safetensors"
        if c06.exists() and c4b.exists():
            return c06, c4b
        if c06.exists():
            # try paired cross-folder
            alt4 = comfy_root / "models" / "clip" / "qwen_4b_ace15.safetensors"
            if alt4.exists():
                return c06, alt4
            alt4 = comfy_root / "models" / "text_encoders" / "qwen_4b_ace15.safetensors"
            if alt4.exists():
                return c06, alt4
    # fallback single checks
    q06 = None
    q4b = None
    for sub in ["clip", "text_encoders"]:
        p = comfy_root / "models" / sub / "qwen_0.6b_ace15.safetensors"
        if p.exists():
            q06 = p
            break
    for sub in ["clip", "text_encoders"]:
        p = comfy_root / "models" / sub / "qwen_4b_ace15.safetensors"
        if p.exists():
            q4b = p
            break
    return q06, q4b


_SCRIPT = textwrap.dedent("""\
    import sys, json, pathlib, pickle
    import torch
    from safetensors.torch import load_file

    inp = pathlib.Path(sys.argv[1])
    out_path = sys.argv[2]
    comfy_root = sys.argv[3]
    sys.path.insert(0, comfy_root)

    data = json.loads(inp.read_text())
    prompt = data["prompt"]
    lyrics = data["lyrics"]
    bpm = data["bpm"]
    duration = data["duration"]
    seed = data["seed"]
    timesignature = data["timesignature"]
    language = data["language"]
    keyscale = data["keyscale"]
    generate_audio_codes = data["generate_audio_codes"]
    cfg_scale = data["cfg_scale"]
    temperature = data["temperature"]
    top_p = data["top_p"]
    top_k = data["top_k"]
    min_p = data["min_p"]
    qwen_06 = data["qwen_06"]
    qwen_4b = data["qwen_4b"]

    from comfy.text_encoders.ace15 import ACE15Tokenizer, te

    tokenizer = ACE15Tokenizer(embedding_directory=None, tokenizer_data={})
    tokens = tokenizer.tokenize_with_weights(
        prompt,
        lyrics=lyrics,
        bpm=bpm,
        duration=duration,
        timesignature=timesignature,
        language=language,
        keyscale=keyscale,
        seed=seed,
        generate_audio_codes=generate_audio_codes,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )

    lm_model = "qwen3_4b" if generate_audio_codes else None
    # Use fp16 on CPU; bf16 for lm if available
    try:
        dtype_llama = torch.bfloat16
    except Exception:
        dtype_llama = torch.float16

    model_class = te(lm_model=lm_model, dtype_llama=dtype_llama)
    # Comfy ace15.M uses float16 for qwen3_06b; respect that
    model = model_class(device="cpu", dtype=torch.float16, model_options={})
    # Load 0.6b
    sd_06 = load_file(qwen_06)
    model.qwen3_06b.load_sd(sd_06)
    if lm_model is not None:
        sd_4b = load_file(qwen_4b)
        # sd_4b goes to qwen3_4b sub-model
        model.qwen3_4b.load_sd(sd_4b)
    model.eval()
    with torch.no_grad():
        cond = model.encode_token_weights(tokens)
    text_hidden = cond[0]
    lyric_hidden = cond[2].get("conditioning_lyrics")
    audio_codes = cond[2].get("audio_codes")
    if lyric_hidden is not None and lyric_hidden.ndim == 2:
        lyric_hidden = lyric_hidden.unsqueeze(0)
    payload = {
        "text_hidden": text_hidden.cpu(),
        "lyric_hidden": lyric_hidden.cpu() if lyric_hidden is not None else None,
        "audio_codes": audio_codes,
    }
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    """)


def encode_via_comfy(
    prompt: str,
    lyrics: str = "",
    duration: float = 30,
    seed: int = 0,
    bpm: int = 120,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    generate_audio_codes: bool = True,
    cfg_scale: float = 2.0,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
) -> dict | None:
    comfy_root = _resolve_comfy_root()
    if comfy_root is None:
        return None
    qwen_06, qwen_4b = _resolve_qwen_paths(comfy_root)
    if qwen_06 is None or not qwen_06.exists():
        return None
    if generate_audio_codes and (qwen_4b is None or not qwen_4b.exists()):
        # Fallback: generate without LLM (no audio codes)
        generate_audio_codes = False
        qwen_4b = qwen_4b or qwen_06

    comfy_python = _resolve_comfy_python()
    if not Path(comfy_python).exists():
        return None

    # normalize timesignature: Comfy tokenizer strips "/4"
    if isinstance(timesignature, int):
        timesignature = str(timesignature)
    # duration: tokenizer expects float or int; ceil handled inside
    payload = {
        "prompt": prompt,
        "lyrics": lyrics or "",
        "bpm": bpm,
        "duration": float(duration),
        "seed": int(seed),
        "timesignature": str(timesignature),
        "language": language,
        "keyscale": keyscale,
        "generate_audio_codes": bool(generate_audio_codes),
        "cfg_scale": float(cfg_scale),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "min_p": float(min_p),
        "qwen_06": str(qwen_06),
        "qwen_4b": str(qwen_4b) if qwen_4b else str(qwen_06),
    }

    inp_path = None
    script_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tf:
            json.dump(payload, tf)
            inp_path = tf.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tf:
            out_path = tf.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".py", mode="w") as sf:
            sf.write(_SCRIPT)
            script_path = sf.name

        result = subprocess.run(
            [comfy_python, script_path, inp_path, out_path, str(comfy_root)],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode != 0:
            print(f"Comfy encode failed: {result.stderr}", file=sys.stderr)
            return None
        with open(out_path, "rb") as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"encode_via_comfy failed: {e}", file=sys.stderr)
        return None
    finally:
        for p in [inp_path, out_path, script_path]:
            if p:
                try:
                    pathlib.Path(p).unlink(missing_ok=True)
                except Exception:
                    pass
