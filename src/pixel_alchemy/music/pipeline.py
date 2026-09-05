"""Music generation via ACE-Step 1.5 (vendored, no ComfyUI deps).

ComfyUI model layout:

+-------------------+---------------------------------------------------------------+
| Component         | Default ComfyUI path                                          |
+===================+===============================================================+
| DiT (diffusion)   | models/diffusion_models/acestep_v1.5_*.safetensors              |
+-------------------+---------------------------------------------------------------+
| VAE (DCAE)        | models/vae/ace_1.5_vae.safetensors                              |
+-------------------+---------------------------------------------------------------+
| Text encoder      | models/clip/qwen_0.6b_ace15.safetensors (or text_encoders/)     |
+-------------------+---------------------------------------------------------------+
| LLM (audio codes) | models/clip/qwen_4b_ace15.safetensors (optional)                |
+-------------------+---------------------------------------------------------------+

Example::

    from pixel_alchemy.music.pipeline import generate

    generate(
        prompt="cinematic piano, warm strings, hopeful",
        lyrics="[Verse] ...",
        diffusion_model="/path/to/acestep_v1.5_xl_turbo_bf16.safetensors",
        vae="/path/to/ace_1.5_vae.safetensors",
        text_encoder="/path/to/qwen_0.6b_ace15.safetensors",
        output="song.wav",
        duration=30,
        bpm=120,
        language="en",
    )

If paths are None, auto-discovers under ~/Documents/comfy/ComfyUI/models/
or $COMFYUI_ROOT.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import soundfile as sf

from .ace import AceStepConditionGenerationModel, get_silence_latent
from .dcae import AudioOobleckVAE, MusicDCAE


# ------------------------------------------------------------------ defaults

def _comfy_root() -> Path | None:
    for cand in [
        os.environ.get("COMFYUI_ROOT"),
        Path.home() / "Documents" / "comfy" / "ComfyUI",
        Path.home() / "comfy" / "ComfyUI",
    ]:
        if cand and Path(cand).exists():
            return Path(cand)
    return None


def _default_path(*parts: str) -> Path | None:
    root = _comfy_root()
    if root is None:
        return None
    p = root.joinpath(*parts)
    if p.exists():
        return p
    if parts[0] == "models" and parts[1] == "text_encoders":
        alt = root.joinpath("models", "clip", *parts[2:])
        if alt.exists():
            return alt
    if parts[0] == "models" and parts[1] == "clip":
        alt = root.joinpath("models", "text_encoders", *parts[2:])
        if alt.exists():
            return alt
    return None


def _resolve(p: str | Path | None, *default_parts: str, required: bool = True) -> Path | None:
    if p is not None:
        p = Path(p)
        if not p.exists() and required:
            raise FileNotFoundError(f"model not found: {p}")
        return p
    d = _default_path(*default_parts)
    if d is None and required:
        raise FileNotFoundError(f"model not found, tried default {'/'.join(default_parts)} – pass path explicitly")
    return d


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _dtype(device: torch.device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


# ------------------------------------------------------------------ model loading

def _load_dit(path: Path, device: torch.device, dtype) -> AceStepConditionGenerationModel:
    from safetensors.torch import load_file

    sd = load_file(str(path), device=str(device))
    hidden = None
    try:
        hidden = int(sd["decoder.layers.0.scale_shift_table"].shape[-1])
    except Exception:
        hidden = None

    enc_hidden = None
    try:
        enc_hidden = int(sd["decoder.condition_embedder.weight"].shape[1])
    except Exception:
        pass

    # Infer number of DiT layers from checkpoint (XL turbo has 32, base 24)
    num_layers = None
    try:
        dec_indices = {int(k.split(".")[2]) for k in sd if k.startswith("decoder.layers.")}
        if dec_indices:
            num_layers = max(dec_indices) + 1
    except Exception:
        num_layers = None

    if hidden == 2560:
        kwargs = dict(
            hidden_size=2560,
            intermediate_size=9728,
            encoder_hidden_size=2048 if (enc_hidden is None or enc_hidden == 2048) else enc_hidden,
            encoder_intermediate_size=6144,
            num_heads=32,
            encoder_num_heads=16,
            head_dim=128,
            dtype=dtype,
            device=device,
        )
        if num_layers is not None:
            kwargs["num_dit_layers"] = num_layers
        model = AceStepConditionGenerationModel(**kwargs)
    elif hidden == 2048:
        kwargs = dict(dtype=dtype, device=device)
        if num_layers is not None:
            kwargs["num_dit_layers"] = num_layers
        model = AceStepConditionGenerationModel(**kwargs)
    else:
        model = AceStepConditionGenerationModel(dtype=dtype, device=device)
        try:
            missing, _ = model.load_state_dict(sd, strict=False)
            if len([k for k in missing if "decoder.layers" in k]) > 50:
                kwargs = dict(
                    hidden_size=2560,
                    intermediate_size=9728,
                    encoder_hidden_size=2048,
                    encoder_intermediate_size=6144,
                    num_heads=32,
                    encoder_num_heads=16,
                    head_dim=128,
                    dtype=dtype,
                    device=device,
                )
                if num_layers is not None:
                    kwargs["num_dit_layers"] = num_layers
                model = AceStepConditionGenerationModel(**kwargs)
        except Exception:
            pass

    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[ACE] DiT missing {len(missing)} keys (e.g. {missing[:3]})", file=sys.stderr)
    if unexpected:
        # Filter tokenizer/detokenizer keys that are expected extra
        unexpected = [k for k in unexpected if not k.startswith("tokenizer.") and not k.startswith("detokenizer.")]
        if unexpected:
            print(f"[ACE] DiT unexpected {len(unexpected)} keys: {unexpected[:5]}", file=sys.stderr)
    model.to(device=device, dtype=dtype).eval()
    return model


def _load_dcae(path: Path, device: torch.device, dtype):
    from safetensors.torch import load_file

    sd = load_file(str(path), device=str(device))
    is_oobleck = "decoder.layers.1.layers.0.beta" in sd or "encoder.layers.0.weight_g" in sd
    if is_oobleck:
        strides = [2, 4, 4, 8, 8]
        for k in ("decoder.layers.2.layers.1.weight_v", "decoder.layers.2.layers.1.parametrizations.weight.original1", "encoder.layers.4.layers.4.parametrizations.weight.original1"):
            if k in sd and sd[k].shape[-1] == 12:
                strides = [2, 4, 4, 6, 10]
                break
        vae = AudioOobleckVAE(strides=strides)
        vae.load_state_dict(sd, strict=False)
        vae.to(device=device, dtype=dtype).eval()

        class _OobleckWrapper(torch.nn.Module):
            def __init__(self, v):
                super().__init__()
                self.v = v
                self.is_oobleck = True

            def decode(self, latents, audio_lengths=None, sr=None):
                with torch.no_grad():
                    wav = self.v.decode(latents)  # [B, 2, T*1920] @48k
                    if sr is not None and sr != 48000:
                        try:
                            import torchaudio

                            if hasattr(torchaudio.functional, "resample"):
                                test = torchaudio.functional.resample(wav[:, :, :10], 48000, sr)
                                if test.shape[-1] != wav.shape[-1]:
                                    wav = torchaudio.functional.resample(wav, 48000, sr)
                                else:
                                    raise RuntimeError("dummy resample")
                            else:
                                raise RuntimeError("no resample")
                        except Exception:
                            # No proper resampler — keep 48k and warn
                            print(f"[ACE] torchaudio resample unavailable, wav stays at 48k (requested {sr})", file=sys.stderr)
                    return wav

            def encode(self, *a, **kw):
                return self.v.encode(*a, **kw)

        return _OobleckWrapper(vae)
    dcae = MusicDCAE()
    missing, unexpected = dcae.load_state_dict(sd, strict=False)
    if len(missing) > 100:
        stripped = {}
        for k, v in sd.items():
            for pref in ("dcae.", "vae.", "model."):
                if k.startswith(pref):
                    k = k[len(pref):]
                    break
            stripped[k] = v
        dcae.load_state_dict(stripped, strict=False)
    dcae.to(device=device, dtype=dtype).eval()
    dcae.is_oobleck = False  # type: ignore[attr-defined]
    return dcae


def _encode_text(
    prompt: str,
    lyrics: str,
    device: torch.device,
    dtype,
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
):
    """Return (text_hidden, lyric_hidden, audio_codes). Tries ComfyUI Qwen, falls back to dummy."""
    try:
        from .comfy_subprocess import encode_via_comfy

        data = encode_via_comfy(
            prompt,
            lyrics,
            duration=duration,
            seed=seed,
            bpm=bpm,
            timesignature=timesignature,
            language=language,
            keyscale=keyscale,
            generate_audio_codes=generate_audio_codes,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
        )
        if data is not None:
            text_hidden = data["text_hidden"].to(device=device, dtype=dtype)
            lyric_hidden = data["lyric_hidden"]
            if lyric_hidden is not None:
                lyric_hidden = lyric_hidden.to(device=device, dtype=dtype)
            audio_codes = data.get("audio_codes")
            return text_hidden, lyric_hidden, audio_codes
        else:
            print("[ACE] Comfy Qwen unavailable (missing models or COMFY_PYTHON), using dummy conditioning", file=sys.stderr)
    except Exception as e:
        print(f"[ACE] encode_via_comfy failed: {e} — using dummy", file=sys.stderr)

    import hashlib

    def _structured_rand(seed_str: str, shape, scale=0.6):
        # include musical metas in hash so dummy is duration/bpm sensitive
        h = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
        g = torch.Generator(device=device)
        g.manual_seed(h % (2**31))
        base = torch.randn(shape, device=device, dtype=dtype, generator=g) * scale
        seq_len = shape[1]
        pos = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        pos_emb = torch.sin(pos * 0.8 + h * 0.05) * 0.15 + torch.cos(pos * 0.3 + h * 0.02) * 0.08
        return base + pos_emb

    meta_tag = f"|bpm{bpm}|ts{timesignature}|lang{language}|key{keyscale}|dur{duration}|cs{cfg_scale}"
    seq_len_text = min(256, max(8, len(prompt.split()) * 2))
    seq_len_lyric = min(512, max(8, len(lyrics.split()) * 2)) if lyrics else 16
    text_hidden = _structured_rand((prompt or "empty") + meta_tag, (1, seq_len_text, 1024))
    lyric_hidden = _structured_rand((lyrics or "empty_lyric") + meta_tag, (1, seq_len_lyric, 1024))
    return text_hidden, lyric_hidden, None


# ------------------------------------------------------------------ reference audio

def _encode_reference_audio(
    path: Path,
    dcae_model,
    device: torch.device,
    dtype,
) -> torch.Tensor:
    """Encode reference wav to latents [1, 64, T]. Hard error on failure."""
    import torch.nn.functional as F  # noqa: F401

    data, sr = sf.read(str(path), always_2d=True)  # [T, C]
    if data.shape[0] == 0:
        raise ValueError(f"reference_audio empty: {path}")
    wav = torch.from_numpy(data.T).float()  # [C, T]
    # mono -> stereo duplicate; >2ch -> mean to stereo
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)
    elif wav.shape[0] > 2:
        wav = wav[:2]
    wav = wav.unsqueeze(0).to(device=device, dtype=dtype)  # [1, 2, T]

    is_oobleck = getattr(dcae_model, "is_oobleck", False) or hasattr(dcae_model, "v")
    target_sr = 48000 if is_oobleck else 44100
    if sr != target_sr:
        try:
            import torchaudio

            if hasattr(torchaudio.functional, "resample") and torchaudio is not None:
                wav = torchaudio.functional.resample(wav, sr, target_sr)
                # detect dummy returning same shape
                if wav.shape[-1] == data.shape[0] and sr != target_sr:
                    raise RuntimeError("dummy resample")
            else:
                raise RuntimeError("no torchaudio resample")
        except Exception:
            # linear fallback (aliasing but better than error for reference)
            ratio = target_sr / sr
            new_len = int(wav.shape[-1] * ratio)
            wav = torch.nn.functional.interpolate(wav, size=new_len, mode="linear", align_corners=False)

    with torch.no_grad():
        if is_oobleck:
            inner = dcae_model.v if hasattr(dcae_model, "v") else dcae_model  # type: ignore[attr-defined]
            # AudioOobleckVAE.encode expects [B, 2, T]
            latents = inner.encode(wav)
            # inner.encode returns [B, 64, T/1920] already
        else:
            # MusicDCAE.encode(audios, sr=target_sr) -> latents
            latents = dcae_model.encode(wav, sr=target_sr)  # type: ignore[attr-defined]
        if isinstance(latents, tuple):
            latents = latents[0]
        latents = latents.to(device=device, dtype=dtype)
        # Ensure [B, 64, T]
        if latents.ndim == 3 and latents.shape[1] != 64 and latents.shape[2] == 64:
            latents = latents.transpose(1, 2)
        return latents


# ------------------------------------------------------------------ sampling

def _flow_euler(
    model: AceStepConditionGenerationModel,
    latents: torch.Tensor,
    text_hidden: torch.Tensor,
    lyric_hidden: torch.Tensor,
    device: torch.device,
    steps: int = 25,
    cfg_scale: float = 2.0,
    audio_codes=None,
    reference_latent: torch.Tensor | None = None,
) -> torch.Tensor:
    """Flow-matching Euler sampler replicating ACEStep15.extra_conds logic."""
    B, C, T = latents.shape
    # ---- refer_audio / is_covers exactly as model_base.ACEStep15.extra_conds ----
    if reference_latent is not None:
        refer_audio = reference_latent[:, :, :T].to(device=device, dtype=latents.dtype)
        is_covers_val: bool | None = True
        pass_audio_codes = False
    else:
        refer_audio = get_silence_latent(T, device).to(device=device, dtype=latents.dtype)
        refer_audio = refer_audio.repeat(B, 1, 1)[:, :, :T]
        pass_audio_codes = True
        is_covers_val = None  # determined below

    has_audio_codes = audio_codes is not None
    if pass_audio_codes:
        if has_audio_codes:
            # truncate silence to 750 as Comfy does when audio_codes present
            refer_audio = refer_audio[:, :, :750]
            is_covers_val = None
        else:
            is_covers_val = False
    # pad if shorter than noise (as in extra_conds)
    if refer_audio.shape[2] < T:
        pad = get_silence_latent(T, device).to(device=device, dtype=latents.dtype)
        # pad is [1,64,T]; repeat for batch
        if pad.shape[0] == 1 and B > 1:
            pad = pad.repeat(B, 1, 1)
        refer_audio = torch.cat([refer_audio, pad[:, :, refer_audio.shape[2]:]], dim=2)
    elif refer_audio.shape[2] > T:
        refer_audio = refer_audio[:, :, :T]

    null_text = torch.zeros_like(text_hidden)
    null_lyric = torch.zeros_like(lyric_hidden)

    x = latents
    shift = 3.0
    for i in range(steps):
        t = 1 - i / steps
        t_next = 1 - (i + 1) / steps
        t_shifted = shift * t / (1 + (shift - 1) * t)
        t_next_shifted = shift * t_next / (1 + (shift - 1) * t_next) if t_next > 0 else 0
        dt = t_next_shifted - t_shifted
        timestep = torch.tensor([t_shifted * 1000], device=device, dtype=x.dtype).repeat(B)

        with torch.no_grad():
            ac = audio_codes
            if ac is not None and isinstance(ac, list):
                # Comfy returns [audio_codes] list[list[int]]; unwrap
                inner = ac[0] if len(ac) == 1 and isinstance(ac[0], list) else ac
                ac = torch.tensor(inner, device=device, dtype=torch.long).unsqueeze(0)
            elif ac is not None and isinstance(ac, torch.Tensor) and ac.ndim == 1:
                ac = ac.unsqueeze(0)
            # is_covers: None (use lm_hints) / True (cover) / False (silence)
            is_covers = is_covers_val
            v_cond = model(
                x, timestep, context=text_hidden, lyric_embed=lyric_hidden, refer_audio=refer_audio, audio_codes=ac, is_covers=is_covers
            )
            if cfg_scale != 1.0:
                v_uncond = model(
                    x,
                    timestep,
                    context=null_text,
                    lyric_embed=null_lyric,
                    refer_audio=refer_audio,
                    audio_codes=ac,
                    is_covers=is_covers,
                    replace_with_null_embeds=True,
                )
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                v = v_cond
        x = x + v * dt
    return x


# ------------------------------------------------------------------ public API

def generate(
    prompt: str,
    lyrics: str = "",
    diffusion_model: str | Path | None = None,
    vae: str | Path | None = None,
    text_encoder: str | Path | None = None,
    llm: str | Path | None = None,
    output: str | Path | None = None,
    *,
    duration: float = 30.0,
    steps: int = 25,
    cfg_scale: float = 2.0,
    seed: int = 0,
    bpm: int = 120,
    timesignature: str = "4",
    language: str = "en",
    keyscale: str = "C major",
    generate_audio_codes: bool = True,
    temperature: float = 0.85,
    top_p: float = 0.9,
    top_k: int = 0,
    min_p: float = 0.0,
    audio_codes: list[int] | None = None,
    reference_audio: str | Path | None = None,
) -> Path:
    """Generate music with ACE-Step 1.5.

    Args:
        prompt: Style/instrumentation tags (e.g. "warm piano, strings").
        lyrics: Full lyrics with structure tags like [Verse], [Chorus].
        diffusion_model: Path to acestep diffusion .safetensors.
        vae: Path to ace_1.5_vae.safetensors (DCAE).
        text_encoder: Path to qwen_0.6b_ace15.safetensors (validated, not loaded directly).
        llm: Path to qwen_4b_ace15.safetensors (optional, for audio codes — not directly loaded; used for validation).
        output: Where to write wav. Defaults to <cwd>/ace_output.wav.
        duration: Seconds of audio to generate.
        steps: Diffusion steps.
        cfg_scale: Classifier-free guidance (Comfy 1.5 default 2.0).
        seed: Random seed.
        bpm: Beats per minute (forwarded to Qwen tokenizer).
        timesignature: Time signature (e.g. "4", "3", "6", "4/4" — stripped to single digit).
        language: Language code (en, ja, zh, ...).
        keyscale: Key/scale (e.g. "C major", "A minor").
        generate_audio_codes: If True, sample LLM semantic tokens (requires qwen 4b). If False, uses silence/VAE path.
        temperature: LLM sampling temperature.
        top_p: LLM nucleus sampling p.
        top_k: LLM top-k (0 = disabled).
        min_p: LLM min-p.
        audio_codes: Precomputed audio codes (overrides generate_audio_codes).
        reference_audio: Path to reference wav for timbre cloning.

    Returns:
        Resolved output path.
    """
    diffusion_model = _resolve(diffusion_model, "models", "diffusion_models", "acestep_v1.5_xl_turbo_bf16.safetensors", required=True)
    vae = _resolve(vae, "models", "vae", "ace_1.5_vae.safetensors", required=True)
    if text_encoder is not None:
        _resolve(text_encoder, "models", "text_encoders", "qwen_0.6b_ace15.safetensors", required=True)
        _resolve(text_encoder, "models", "clip", "qwen_0.6b_ace15.safetensors", required=False)
    if llm is not None:
        _resolve(llm, "models", "text_encoders", "qwen_4b_ace15.safetensors", required=True)

    if output is None:
        output = Path("ace_output.wav")
    else:
        output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    device = _device()
    dtype = _dtype(device)
    torch.manual_seed(seed)

    latent_len = round(duration * 48000 / 1920)
    latents = torch.randn(1, 64, latent_len, device=device, dtype=dtype)

    # Load models — hard error if missing (no sine fallback)
    dit = _load_dit(diffusion_model, device, dtype)
    dcae_model = _load_dcae(vae, device, dtype)

    # Reference latent (requires DCAE)
    reference_latent = None
    if reference_audio is not None:
        ref_path = Path(reference_audio)
        if not ref_path.exists():
            raise FileNotFoundError(f"reference_audio not found: {ref_path}")
        reference_latent = _encode_reference_audio(ref_path, dcae_model, device, dtype)

    # Text conditioning (full Comfy kwargs)
    text_hidden, lyric_hidden, audio_codes_comfy = _encode_text(
        prompt,
        lyrics,
        device,
        dtype,
        duration=duration,
        seed=seed,
        bpm=bpm,
        timesignature=timesignature,
        language=language,
        keyscale=keyscale,
        generate_audio_codes=generate_audio_codes,
        cfg_scale=cfg_scale,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
    )
    if audio_codes is not None:
        audio_codes_comfy = audio_codes

    # Normalize audio_codes: Comfy returns [list[int]]; pipeline accepts list[int] or tensor
    audio_codes_tensor = None
    if audio_codes_comfy is not None:
        if isinstance(audio_codes_comfy, list):
            # Could be [[int,...]] or [int,...]
            inner = audio_codes_comfy[0] if len(audio_codes_comfy) == 1 and isinstance(audio_codes_comfy[0], list) else audio_codes_comfy
            audio_codes_tensor = inner  # keep as list for _flow_euler to convert
            # need list form for truncation check in _flow_euler, so pass list
            audio_codes_tensor = inner if isinstance(inner, list) else audio_codes_comfy
            # _flow_euler handles list->tensor
            # keep as list for now to preserve [ [codes] ] unwrapping logic
            audio_codes_tensor = audio_codes_comfy if isinstance(audio_codes_comfy[0], list) else [audio_codes_comfy]  # type: ignore
            # Actually _flow_euler expects list[int] or list[list[int]]; pass as-is
            audio_codes_tensor = audio_codes_comfy
        elif isinstance(audio_codes_comfy, torch.Tensor):
            audio_codes_tensor = audio_codes_comfy.to(device=device)
        else:
            audio_codes_tensor = audio_codes_comfy

    # Diffusion sampling
    latents = _flow_euler(
        dit,
        latents,
        text_hidden,
        lyric_hidden,
        device,
        steps=steps,
        cfg_scale=cfg_scale,
        audio_codes=audio_codes_tensor,
        reference_latent=reference_latent,
    )

    # Decode: Oobleck VAE returns [B, 2, T*1920] stereo @48k; MusicDCAE @44.1k
    target_sr = 44100
    is_oobleck = getattr(dcae_model, "is_oobleck", False)
    decode_sr = 44100
    # Oobleck wrapper handles resample internally; for non-Oobleck, MusicDCAE already at 44100
    with torch.no_grad():
        wav_tensor = dcae_model.decode(latents, sr=decode_sr)  # [B, C, T] or [B, T]
        # Peak normalize: boost quiet (<0.08) and tame loud (>0.95)
        peak = float(wav_tensor.abs().max().item()) if wav_tensor.numel() > 0 else 0.0
        if 0 < peak < 0.08:
            wav_tensor = wav_tensor * (0.35 / (peak + 1e-6))
        elif peak > 0.95:
            wav_tensor = wav_tensor * (0.89 / peak)
        wav_tensor = wav_tensor.clamp(-1.0, 1.0)
        # To numpy for soundfile: [T] mono or [T, C] stereo
        if wav_tensor.ndim == 3:
            w = wav_tensor[0]  # [C, T]
            if w.shape[0] == 1:
                wav = w.squeeze(0).float().cpu().numpy()
            elif w.shape[0] == 2:
                wav = w.float().cpu().numpy().T  # [T, 2]
            else:
                wav = w.mean(dim=0).float().cpu().numpy()
        elif wav_tensor.ndim == 2:
            wav = wav_tensor[0].float().cpu().numpy()
        else:
            wav = wav_tensor.float().cpu().numpy()
        target_samples = int(target_sr * duration)
        if wav.ndim == 1:
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                wav = torch.nn.functional.pad(torch.from_numpy(wav), (0, target_samples - wav.shape[0])).numpy()
        else:
            if wav.shape[0] > target_samples:
                wav = wav[:target_samples]
            elif wav.shape[0] < target_samples:
                pad = target_samples - wav.shape[0]
                wav = torch.nn.functional.pad(torch.from_numpy(wav.T), (0, pad)).numpy().T

    # Oobleck decode was at 48k then resampled to 44.1k inside wrapper; if wrapper didn't resample (no torchaudio), wav is 48k
    # Detect: if is_oobleck and wav shape corresponds to 48k length, we wrote target_sr=44100 but wav length is ~48k*duration
    # Wrapper already attempted resample; if it failed, wav length will be 48k*duration, we should not silently keep it at 44100 sr
    actual_sr = target_sr
    if is_oobleck:
        expected_48k = int(48000 * duration)
        # if wav length close to 48k not 44.1k, use 48k sr
        if abs(wav.shape[0] - expected_48k) < abs(wav.shape[0] - target_samples):
            actual_sr = 48000

    sf.write(str(output), wav, actual_sr)
    return output


def generate_with_reference(
    prompt: str,
    lyrics: str,
    reference_audio: str | Path,
    diffusion_model: str | Path | None = None,
    vae: str | Path | None = None,
    text_encoder: str | Path | None = None,
    output: str | Path | None = None,
    **kwargs,
) -> Path:
    """Timbre cloning: generate conditioned on reference audio."""
    return generate(
        prompt=prompt,
        lyrics=lyrics,
        diffusion_model=diffusion_model,
        vae=vae,
        text_encoder=text_encoder,
        output=output,
        reference_audio=reference_audio,
        **kwargs,
    )
