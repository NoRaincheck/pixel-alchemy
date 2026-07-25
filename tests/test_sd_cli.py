from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pixel_alchemy.generation.sd_cli import _find_sd_cli, edit, generate, inpaint


@pytest.fixture()
def model_files(tmp_path: Path) -> dict[str, Path]:
    dm = tmp_path / "model.gguf"
    vae = tmp_path / "vae.safetensors"
    llm = tmp_path / "llm.gguf"
    dm.touch()
    vae.touch()
    llm.touch()
    return {"diffusion_model": dm, "vae": vae, "llm": llm}


@pytest.fixture()
def image_files(tmp_path: Path) -> dict[str, Path]:
    ref = tmp_path / "ref.png"
    mask = tmp_path / "mask.png"
    ref.touch()
    mask.touch()
    return {"reference": ref, "mask": mask}


@patch("pixel_alchemy.generation.sd_cli.shutil.which", return_value="/usr/bin/sd-cli")
def test_find_sd_cli_found(mock_which: MagicMock) -> None:
    with patch("pixel_alchemy.generation.sd_cli.Path.exists", return_value=True):
        result = _find_sd_cli()
        assert result == Path("/usr/bin/sd-cli")


@patch("pixel_alchemy.generation.sd_cli.shutil.which", return_value=None)
def test_find_sd_cli_not_found(mock_which: MagicMock) -> None:
    with pytest.raises(FileNotFoundError, match="sd-cli not found on PATH"):
        _find_sd_cli()


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_default_output(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    result = generate("a cat", **model_files)
    assert result == Path("generated.png")
    args = mock_run.call_args[0][0]
    assert "-p" in args
    assert "a cat" in args
    assert "--output" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_custom_output(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    result = generate("a cat", output="custom.png", **model_files)
    assert result == Path("custom.png")


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_passes_model_paths(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", **model_files)
    args = mock_run.call_args[0][0]
    assert "--diffusion-model" in args
    assert str(model_files["diffusion_model"]) in args
    assert "--vae" in args
    assert str(model_files["vae"]) in args
    assert "--llm" in args
    assert str(model_files["llm"]) in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_offload_and_fa_flags(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", **model_files)
    args = mock_run.call_args[0][0]
    assert "--offload-to-cpu" in args
    assert "--diffusion-fa" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_no_offload(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", offload_to_cpu=False, diffusion_fa=False, **model_files)
    args = mock_run.call_args[0][0]
    assert "--offload-to-cpu" not in args
    assert "--diffusion-fa" not in args


def test_generate_missing_model(model_files: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError, match="diffusion_model"):
        generate("a cat", diffusion_model="/no/such/file.gguf", vae=model_files["vae"], llm=model_files["llm"])


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_edit_uses_reference_flag(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    edit(reference=image_files["reference"], prompt="change it", **model_files)
    args = mock_run.call_args[0][0]
    assert "-r" in args
    assert str(image_files["reference"]) in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_edit_default_output(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    result = edit(reference=image_files["reference"], prompt="change it", **model_files)
    assert result == Path("edited.png")


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_edit_custom_output(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    result = edit(reference=image_files["reference"], prompt="change it", output="my-edit.png", **model_files)
    assert result == Path("my-edit.png")


def test_edit_missing_reference(model_files: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError, match="reference"):
        edit(reference="/no/such/file.png", prompt="change it", **model_files)


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_uses_init_img_and_mask(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", **model_files)
    args = mock_run.call_args[0][0]
    assert "--init-img" in args
    assert str(image_files["reference"]) in args
    assert "--mask" in args
    assert str(image_files["mask"]) in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_default_output(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    result = inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", **model_files)
    assert result == Path("inpainted.png")


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_vae_tiling_flags(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", **model_files)
    args = mock_run.call_args[0][0]
    assert "--vae-tiling" in args
    assert "--vae-tile-overlap" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_no_vae_tiling(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", vae_tiling=False, **model_files)
    args = mock_run.call_args[0][0]
    assert "--vae-tiling" not in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_color_flag(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", **model_files)
    args = mock_run.call_args[0][0]
    assert "--color" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_no_color(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", color=False, **model_files)
    args = mock_run.call_args[0][0]
    assert "--color" not in args


def test_inpaint_missing_init_image(model_files: dict[str, Path]) -> None:
    with pytest.raises(FileNotFoundError, match="init_image"):
        inpaint(init_image="/no/such/file.png", mask="/no/mask.png", prompt="a dog", **model_files)


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_sampling_method(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", sampling_method="dpm++2s_a", **model_files)
    args = mock_run.call_args[0][0]
    assert "--sampling-method" in args
    assert "dpm++2s_a" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_custom_dimensions(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", width=768, height=1024, **model_files)
    args = mock_run.call_args[0][0]
    assert "-H" in args
    assert "1024" in args
    assert "-W" in args
    assert "768" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_verbose_flag(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", verbose=True, **model_files)
    args = mock_run.call_args[0][0]
    assert "-v" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_no_verbose(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", **model_files)
    args = mock_run.call_args[0][0]
    assert "-v" not in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_inpaint_threads(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path], image_files: dict[str, Path]) -> None:
    inpaint(init_image=image_files["reference"], mask=image_files["mask"], prompt="a dog", threads=16, **model_files)
    args = mock_run.call_args[0][0]
    assert "-t" in args
    assert "16" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_steps(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", steps=20, **model_files)
    args = mock_run.call_args[0][0]
    assert "--steps" in args
    assert "20" in args


@patch("pixel_alchemy.generation.sd_cli.subprocess.run")
@patch("pixel_alchemy.generation.sd_cli._find_sd_cli", return_value=Path("/fake/sd-cli"))
def test_generate_cfg_scale(mock_find: MagicMock, mock_run: MagicMock, model_files: dict[str, Path]) -> None:
    generate("a cat", cfg_scale=7.5, **model_files)
    args = mock_run.call_args[0][0]
    assert "--cfg-scale" in args
    assert "7.5" in args
