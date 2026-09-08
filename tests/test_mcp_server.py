from __future__ import annotations

import base64
import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image as PILImage

from pixel_alchemy.mcp.server import (
    _decode_base64_to_file,
    _strip_data_uri,
    mcp,
)

# helpers -----------------------------------------------------------------


def _make_base64_image(size=(8, 8), color="red", fmt="PNG") -> str:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def _mock_generate(prompt, diffusion_model, vae, llm, output, **kw):
    # write a tiny valid PNG to output path
    PILImage.new("RGB", (kw.get("width", 512), kw.get("height", 512)), "blue").save(output, "PNG")
    return Path(output)


def _mock_birefnet(input_path, output_path, fp16=True, foreground=False):
    # mimic real birefnet: save either mask or foreground
    img = PILImage.open(input_path)
    if foreground:
        img = img.convert("RGBA")
        img.save(output_path, "PNG")
    else:
        mask = PILImage.new("L", img.size, 128)
        mask.save(output_path, "PNG")
    return Path(output_path)


def _mock_upscayl_pipeline(input_path, output_path, target_width=2048, blur_multipliers=None, model="upscayl-standard-4x", scale=4):
    # just resize to target_width for test, save PNG
    img = PILImage.open(input_path)
    w, h = img.size
    new_h = round(target_width * h / w) if w else target_width
    img.resize((target_width, new_h), PILImage.LANCZOS).save(output_path, "PNG")
    return Path(output_path)


# helper tests -------------------------------------------------------------


def test_strip_data_uri_plain():
    b64 = _make_base64_image()
    raw, mime = _strip_data_uri(b64)
    assert raw == b64.strip()
    assert mime is None


def test_strip_data_uri_with_prefix():
    b64 = _make_base64_image()
    uri = f"data:image/png;base64,{b64}"
    raw, mime = _strip_data_uri(uri)
    assert raw == b64
    assert mime == "image/png"


def test_decode_base64_to_file(tmp_path: Path):
    b64 = _make_base64_image(size=(4, 4))
    p = _decode_base64_to_file(b64, str(tmp_path))
    assert p.exists()
    assert PILImage.open(p).size == (4, 4)


def test_decode_data_uri(tmp_path: Path):
    b64 = _make_base64_image()
    uri = f"data:image/jpeg;base64,{b64}"
    p = _decode_base64_to_file(uri, str(tmp_path))
    assert p.exists()
    assert p.suffix == ".jpg"


def test_decode_invalid_base64(tmp_path: Path):
    with pytest.raises(ValueError, match="Invalid base64"):
        _decode_base64_to_file("!!!not base64!!!", str(tmp_path))


# mcp tool listing ---------------------------------------------------------


def test_list_tools_contains_three():
    import anyio

    async def _run():
        tools = await mcp.list_tools()
        names = {t.name for t in tools}
        assert "generate_image" in names
        assert "remove_background" in names
        assert "upscale_image" in names

    anyio.run(_run)


# generate_image -----------------------------------------------------------


def test_generate_image_calls_defaults(tmp_path: Path):
    # create fake model files so exists() passes
    dm = tmp_path / "z_image.gguf"
    vae = tmp_path / "vae.safetensors"
    llm = tmp_path / "llm.gguf"
    dm.touch()
    vae.touch()
    llm.touch()

    import anyio

    with patch("pixel_alchemy.mcp.server._DEFAULT_DIFFUSION", dm), patch("pixel_alchemy.mcp.server._DEFAULT_VAE", vae), patch(
        "pixel_alchemy.mcp.server._DEFAULT_LLM", llm
    ), patch("pixel_alchemy.generation.sd_cli.generate", side_effect=_mock_generate) as mock_gen:
        async def _run():
            res = await mcp.call_tool("generate_image", {"prompt": "a cat"})
            assert res.content
            # should be ImageContent
            assert res.content[0].type == "image"
            # data decodes to PNG
            data = base64.b64decode(res.content[0].data)
            assert data[:8] == b"\x89PNG\r\n\x1a\n"
            mock_gen.assert_called_once()
            kwargs = mock_gen.call_args.kwargs
            # check defaults were used
            assert kwargs["width"] == 1024
            assert kwargs["height"] == 1024

        anyio.run(_run)


def test_generate_image_custom_dimensions(tmp_path: Path):
    dm = tmp_path / "dm.gguf"
    vae = tmp_path / "vae.safetensors"
    llm = tmp_path / "llm.gguf"
    for p in (dm, vae, llm):
        p.touch()

    import anyio

    with patch("pixel_alchemy.mcp.server._DEFAULT_DIFFUSION", dm), patch("pixel_alchemy.mcp.server._DEFAULT_VAE", vae), patch(
        "pixel_alchemy.mcp.server._DEFAULT_LLM", llm
    ), patch("pixel_alchemy.generation.sd_cli.generate", side_effect=_mock_generate) as mock_gen:
        async def _run():
            res = await mcp.call_tool(
                "generate_image",
                {"prompt": "hello", "width": 768, "height": 1365, "steps": 9, "sampling_method": "euler"},
            )
            assert res.content[0].type == "image"
            assert mock_gen.call_args.kwargs["width"] == 768
            assert mock_gen.call_args.kwargs["height"] == 1365
            assert mock_gen.call_args.kwargs["steps"] == 9

        anyio.run(_run)


def test_generate_image_empty_prompt_fails(tmp_path: Path):
    import anyio

    async def _run():
        with pytest.raises(Exception):  # ToolError wrapped
            await mcp.call_tool("generate_image", {"prompt": "   "})

    anyio.run(_run)


def test_generate_image_missing_model_raises(tmp_path: Path):
    import anyio

    dm = tmp_path / "missing.gguf"
    vae = tmp_path / "vae.safetensors"
    llm = tmp_path / "llm.gguf"
    vae.touch()
    llm.touch()
    with patch("pixel_alchemy.mcp.server._DEFAULT_DIFFUSION", dm), patch("pixel_alchemy.mcp.server._DEFAULT_VAE", vae), patch(
        "pixel_alchemy.mcp.server._DEFAULT_LLM", llm
    ):
        async def _run():
            with pytest.raises(Exception):
                await mcp.call_tool("generate_image", {"prompt": "a cat"})

        anyio.run(_run)


# remove_background --------------------------------------------------------


def test_remove_background_foreground(tmp_path: Path):
    b64 = _make_base64_image(size=(10, 10))

    import anyio

    with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet):
        async def _run():
            res = await mcp.call_tool("remove_background", {"image_base64": b64, "foreground": True})
            assert res.content[0].type == "image"
            # decode and check RGBA
            data = base64.b64decode(res.content[0].data)
            img = PILImage.open(io.BytesIO(data))
            assert img.mode == "RGBA"
            assert img.size == (10, 10)
            assert res.content[0].mime_type == "image/png"

        anyio.run(_run)


def test_remove_background_mask(tmp_path: Path):
    b64 = _make_base64_image(size=(6, 4))

    import anyio

    with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet):
        async def _run():
            res = await mcp.call_tool("remove_background", {"image_base64": b64, "foreground": False})
            assert res.content[0].type == "image"
            data = base64.b64decode(res.content[0].data)
            img = PILImage.open(io.BytesIO(data))
            assert img.mode == "L"

        anyio.run(_run)


def test_remove_background_data_uri(tmp_path: Path):
    raw = _make_base64_image()
    uri = f"data:image/png;base64,{raw}"
    import anyio

    with patch("pixel_alchemy.background_removal.birefnet.birefnet", side_effect=_mock_birefnet) as mock:
        async def _run():
            res = await mcp.call_tool("remove_background", {"image_base64": uri})
            assert res.content[0].type == "image"
            mock.assert_called_once()

        anyio.run(_run)


def test_remove_background_empty_fails():
    import anyio

    async def _run():
        with pytest.raises(Exception):
            await mcp.call_tool("remove_background", {"image_base64": ""})

    anyio.run(_run)


# upscale_image ------------------------------------------------------------


def test_upscale_image_default(tmp_path: Path):
    b64 = _make_base64_image(size=(4, 4))

    import anyio

    with patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl_pipeline", side_effect=_mock_upscayl_pipeline) as mock:
        async def _run():
            res = await mcp.call_tool("upscale_image", {"image_base64": b64})
            assert res.content[0].type == "image"
            data = base64.b64decode(res.content[0].data)
            img = PILImage.open(io.BytesIO(data))
            assert img.size[0] == 2048  # default target_width
            mock.assert_called_once()
            assert mock.call_args.kwargs["target_width"] == 2048

        anyio.run(_run)


def test_upscale_image_custom_params(tmp_path: Path):
    b64 = _make_base64_image(size=(8, 4))

    import anyio

    with patch("pixel_alchemy.pipeline.upscayl_pipeline.upscayl_pipeline", side_effect=_mock_upscayl_pipeline) as mock:
        async def _run():
            res = await mcp.call_tool(
                "upscale_image",
                {"image_base64": b64, "target_width": 1024, "blur_multipliers": [3, 1], "model": "high-fidelity-4x", "scale": 2},
            )
            assert res.content[0].type == "image"
            data = base64.b64decode(res.content[0].data)
            img = PILImage.open(io.BytesIO(data))
            assert img.size[0] == 1024
            assert mock.call_args.kwargs["blur_multipliers"] == [3, 1]
            assert mock.call_args.kwargs["model"] == "high-fidelity-4x"
            assert mock.call_args.kwargs["scale"] == 2

        anyio.run(_run)


def test_upscale_image_invalid_width():
    b64 = _make_base64_image()
    import anyio

    async def _run():
        with pytest.raises(Exception):
            await mcp.call_tool("upscale_image", {"image_base64": b64, "target_width": 0})

    anyio.run(_run)


def test_upscale_image_invalid_scale():
    b64 = _make_base64_image()
    import anyio

    async def _run():
        with pytest.raises(Exception):
            await mcp.call_tool("upscale_image", {"image_base64": b64, "scale": 5})

    anyio.run(_run)
