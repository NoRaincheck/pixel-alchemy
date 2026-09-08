"""LLM orchestrator for Sonic Pi generation via tau-ai OpenAI-compatible provider.

Ports App/services/agent.py::GPTAgent (discussion / execute_composition_chain)
but:
- uses tau_ai.OpenAICompatibleProvider instead of openai.Anthropic clients
- supports local qwen3.6-35b-a3b-mtp via LMStudio base_url http://localhost:1234/v1
- falls back to openai python client if tau not desired
- simple keyword sample selection (no FAISS)

Example::

    from pixel_alchemy.music_agent.agent import MusicAgent, generate_sonic_pi_from_audio

    agent = MusicAgent(model="qwen3.6-35b-a3b-mtp", base_url="http://localhost:1234/v1")
    code = await agent.generate_async(genre="LoFi", duration=120, prompt="chill cafe vibe")

    # one-shot from audio
    code = generate_sonic_pi_from_audio("input.wav", genre="LoFi", duration=90)
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

from .analyze import analyze_sample
from .config import get_tau_config, load_json_config
from .embed import load_metadata, select_samples
from .song import Song
from .song_data import SongCreationData
from .sonic_pi import fix_sample_paths_for_llm, fix_sonic_pi_notes

DEFAULT_SYSTEM_CODER = (
    "As a songwriter skilled in Sonic Pi, you create and program songs using this tool. "
    "You understand the basic song structure: intro, verse, chorus, verse, chorus, bridge, chorus, outro. "
    "When having clear arrangements, rhythm, structure, and sections defined, you are able to translate them into functional Sonic Pi code."
)

# reuse artist.json prompts at runtime


def _load_assistant_system(role: str) -> str:
    cfg = load_json_config("artist.json")
    for a in cfg.get("assistants", []):
        if a["name"] == role:
            return "\n".join(a.get("system_instruction", []))
    return DEFAULT_SYSTEM_CODER


def _render_phase(phase_name: str, data: SongCreationData) -> tuple[str, str, dict]:
    """Return (system, prompt, phase_cfg) for a chat phase."""
    phases = load_json_config("phases.json")
    cfg = phases.get(phase_name)
    if not cfg:
        raise KeyError(f"phase {phase_name!r} not found in phases.json")
    role = cfg.get("assistant_role_name", "Sonic PI coder")
    system = _load_assistant_system(role)
    prompt = "".join(cfg.get("phase_prompt", []))
    for key, param_name in cfg.get("input", {}).items():
        val = data.get_parameter(param_name)
        if val is not None:
            prompt = prompt.replace(f"{{{key}}}", str(val))
    return system, prompt, cfg


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return json.loads(text)


def _strip_think(text: str) -> str:
    # Qwen thinking may be in <think> blocks
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


# ── Low-level tau call ─────────────────────────────────────────────────────────

async def _call_tau(
    system: str,
    prompt: str,
    model: str,
    base_url: str,
    api_key: str,
    timeout: float = 120,
    max_tokens: int = 6000,
) -> str:
    from tau_agent.messages import UserMessage
    from tau_ai import OpenAICompatibleProvider
    from tau_ai.env import OpenAICompatibleConfig

    cfg = OpenAICompatibleConfig(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout_seconds=timeout,
        supports_images=False,
        max_tokens=max_tokens,
    )
    provider = OpenAICompatibleProvider(cfg)
    msg = UserMessage(content=prompt)
    full = ""
    try:
        async for ev in provider.stream_response(
            model=model, system=system, messages=[msg], tools=[]
        ):
            if ev.__class__.__name__ == "TextDeltaEvent":
                full += ev.delta  # type: ignore[attr-defined]
            elif ev.__class__.__name__ == "ThinkingDeltaEvent":
                # ignore thinking deltas for final code
                pass
    finally:
        await provider.aclose()
    return full


def _call_tau_sync(system: str, prompt: str, model: str, base_url: str, api_key: str, timeout: float = 120) -> str:
    return asyncio.run(_call_tau(system, prompt, model, base_url, api_key, timeout))


def _call_openai_compat(system: str, prompt: str, model: str, base_url: str, api_key: str, timeout: float = 120) -> str:
    """Fallback via openai python client (also works for LMStudio)."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=8000,
    )
    return resp.choices[0].message.content or ""


# ── High-level MusicAgent ─────────────────────────────────────────────────────

class MusicAgent:
    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        use_tau: bool = True,
        song_name: str = "generated",
        timeout: float = 120,
    ):
        tau = get_tau_config()
        self.model = model or tau["model"]
        self.base_url = base_url or tau["base_url"]
        self.api_key = api_key or tau["api_key"]
        self.use_tau = use_tau
        self.song_name = song_name
        self.timeout = timeout
        self.data = SongCreationData()

    def _chat(self, system: str, prompt: str) -> str:
        if self.use_tau:
            try:
                return _call_tau_sync(system, prompt, self.model, self.base_url, self.api_key, self.timeout)
            except Exception as e:
                print(f"[warn] tau call failed ({e}), falling back to openai client")
                return _call_openai_compat(system, prompt, self.model, self.base_url, self.api_key, self.timeout)
        return _call_openai_compat(system, prompt, self.model, self.base_url, self.api_key, self.timeout)

    async def _chat_async(self, system: str, prompt: str) -> str:
        return await _call_tau(system, prompt, self.model, self.base_url, self.api_key, self.timeout)

    # ── Chain ────────────────────────────────────────────────────────────────

    def _prepare_data(
        self,
        genre: str = "LoFi",
        duration: int = 120,
        prompt_extra: str = "",
        audio_meta: dict | None = None,
        samples_pool: str | Path | None = None,
        samples_k: int = 5,
    ):
        d = self.data
        # song_description from genre + extra + audio vibe
        audio_vibe = ""
        if audio_meta and "Vibe" in audio_meta:
            audio_vibe = f" Reference audio: {audio_meta.get('Vibe')} Key {audio_meta.get('Key')} BPM {audio_meta.get('BPM')} Tags {audio_meta.get('Tags')}."
        d.set_parameter("song_description", f"I want a {genre} song. {prompt_extra}{audio_vibe}".strip())
        d.set_parameter("total_duration", str(duration))
        # placeholder defaults if LLM chain skipped
        if not d.theme:
            d.theme = f"{genre} mood"
        if not d.melody:
            d.melody = f"{genre} melodic, {audio_meta.get('Key','C major') if audio_meta else 'C major'} tonality" if audio_meta else f"{genre} melodic"
        if not d.rhythm:
            d.rhythm = f"{audio_meta.get('BPM','90')} BPM, {audio_meta.get('Tags',[''])[0] if audio_meta else 'moderate'} groove" if audio_meta else "moderate tempo"
        # samples json
        if samples_pool is not None:
            metas = load_metadata(samples_pool)
            q = f"{d.theme} {d.melody} {d.rhythm} {prompt_extra} {genre}"
            chosen = select_samples(metas, q, k=samples_k)
            d.samples = json.dumps(chosen, separators=(",", ":"))
        elif audio_meta:
            d.samples = json.dumps([audio_meta], separators=(",", ":"))

    def generate(
        self,
        genre: str = "LoFi",
        duration: int = 120,
        prompt_extra: str = "",
        audio_meta: dict | None = None,
        samples_pool: str | Path | None = None,
        do_review: bool = False,
        do_mix: bool = True,
    ) -> str:
        """Run composition chain synchronously, return sonicpi_code."""
        self._prepare_data(genre, duration, prompt_extra, audio_meta, samples_pool)

        # fast path if LMStudio not reachable: deterministic fallback via incidental
        if not self._is_reachable():
            print("[info] LLM not reachable — using deterministic incidental fallback")
            return self._fallback_generate(genre, duration)

        # Conceptualization
        try:
            system, prompt, _ = _render_phase("Conceptualization", self.data)
            raw = _strip_think(self._chat(system, prompt))
            parsed = _extract_json(raw)
            self.data.update_parameters_from_response(parsed)
        except Exception as e:
            print(f"[warn] Conceptualization failed: {e}")

        # Songwriting
        try:
            system, prompt, _ = _render_phase("Songwriting", self.data)
            raw = _strip_think(self._chat(system, prompt))
            parsed = _extract_json(raw)
            self.data.update_parameters_from_response(parsed)
        except Exception as e:
            print(f"[warn] Songwriting failed: {e}")

        # Segmentation
        try:
            system, prompt, _ = _render_phase("Segmentation", self.data)
            raw = _strip_think(self._chat(system, prompt))
            parsed = _extract_json(raw)
            self.data.update_parameters_from_response(parsed)
        except Exception as e:
            print(f"[warn] Segmentation failed: {e}")

        # Arrangements
        try:
            system, prompt, _ = _render_phase("Arrangements", self.data)
            raw = _strip_think(self._chat(system, prompt))
            parsed = _extract_json(raw)
            self.data.update_parameters_from_response(parsed)
        except Exception as e:
            print(f"[warn] Arrangements failed: {e}")

        # Sampling already done via embed.py in _prepare_data

        # Initial Song Coding
        code = ""
        try:
            system, prompt, _ = _render_phase("Initial Song Coding", self.data)
            raw = _strip_think(self._chat(system, prompt))
            parsed = _extract_json(raw)
            if "sonicpi_code" in parsed:
                c = parsed["sonicpi_code"]
                code = "\n".join(c) if isinstance(c, list) else str(c)
                code = fix_sonic_pi_notes(code)
                self.data.set_parameter("sonicpi_code", code)
        except Exception as e:
            print(f"[warn] Initial coding failed: {e}")

        if not code:
            return self._fallback_generate(genre, duration)

        # Optional review cycle
        if do_review:
            try:
                system, prompt, _ = _render_phase("Code Review", self.data)
                raw = _strip_think(self._chat(system, prompt))
                parsed = _extract_json(raw)
                review = parsed.get("review", "")
                if review and "no further code changes are required" not in str(review).lower():
                    self.data.set_parameter("review", review)
                    system2, prompt2, _ = _render_phase("Code Modification", self.data)
                    raw2 = _strip_think(self._chat(system2, prompt2))
                    parsed2 = _extract_json(raw2)
                    if "sonicpi_code" in parsed2:
                        c2 = parsed2["sonicpi_code"]
                        code = "\n".join(c2) if isinstance(c2, list) else str(c2)
                        code = fix_sonic_pi_notes(code)
                        self.data.set_parameter("sonicpi_code", code)
            except Exception as e:
                print(f"[warn] Review cycle failed: {e}")

        # Song mixing
        if do_mix:
            try:
                system, prompt, _ = _render_phase("Song mixing", self.data)
                raw = _strip_think(self._chat(system, prompt))
                parsed = _extract_json(raw)
                if "sonicpi_code" in parsed:
                    c = parsed["sonicpi_code"]
                    code = "\n".join(c) if isinstance(c, list) else str(c)
                    code = fix_sonic_pi_notes(code)
                    self.data.set_parameter("sonicpi_code", code)
            except Exception as e:
                print(f"[warn] Mixing failed: {e}")

        # write file
        Song(self.song_name).create_song_file(self.data)
        return code

    def _is_reachable(self, timeout: float = 1.5) -> bool:
        import urllib.request

        url = self.base_url.rstrip("/") + "/models"
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status < 500
        except Exception:
            return False

    def _fallback_generate(self, genre: str, duration: int) -> str:
        """Deterministic fallback via incidental generator."""
        try:
            from pixel_alchemy.incidental.generator import generate_piece, to_sonic_pi

            # map duration to bars (4 beats each, ~52 bpm)
            bpm_guess = 52
            bars = max(4, round(duration / (60 / bpm_guess * 4)))
            seed = hash(f"{genre}{duration}") % 999999
            piece = generate_piece(seed=seed, bars=bars, instrument="piano")
            code = to_sonic_pi(piece)
            self.data.set_parameter("sonicpi_code", code)
            Song(self.song_name).create_song_file(self.data)
            return code
        except Exception as e:
            return f"# fallback generation failed: {e}\n# genre={genre} duration={duration}"

    async def generate_async(self, **kw) -> str:
        # simple async wrapper — runs sync in thread pool for now
        import concurrent.futures

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as pool:
            return await loop.run_in_executor(pool, lambda: self.generate(**kw))


# ── Convenience one-shot from audio file ────────────────────────────────────

def generate_sonic_pi_from_audio(
    audio_path: str | Path,
    genre: str = "LoFi",
    duration: int = 120,
    prompt_extra: str = "",
    samples_pool: str | Path | None = None,
    song_name: str = "generated",
    **agent_kw,
) -> str:
    meta = analyze_sample(audio_path)
    if "Error" in meta:
        raise ValueError(f"analyze failed: {meta['Error']}")
    agent = MusicAgent(song_name=song_name, **agent_kw)
    return agent.generate(genre=genre, duration=duration, prompt_extra=prompt_extra, audio_meta=meta, samples_pool=samples_pool)


def fix_code_notes(code: str) -> str:
    return fix_sonic_pi_notes(code)
