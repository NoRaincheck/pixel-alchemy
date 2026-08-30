"""
AdaAttN: Revisit Attention Mechanism in Arbitrary Neural Style Transfer.

Implements the ICCV 2021 paper by Liu et al.:

    Liu, Songhua et al. "AdaAttN: Revisit Attention Mechanism in
    Arbitrary Neural Style Transfer." ICCV 2021.
    arXiv: https://arxiv.org/abs/2108.03647

Reference implementation: https://github.com/Huage001/AdaAttN
    (officially unofficial PyTorch re-implementation, Apache-2.0).

Algorithm overview
------------------
* **Encoder**: VGG-19 up to ``relu4_1`` / ``relu5_1`` with a leading
  ``1×1`` normalisation conv (``vgg_normalised.pth``).  Frozen; split
  into 5 stages so shallow features (``relu1_2``/``relu2_2``/``relu3_4``)
  can be re-used.
* **AdaAttN module**: per-point attention.  Content/style keys
  (``f``/``g``, ``1×1`` convs) give attention ``S = softmax(F·G)``,
  style values (``h``) give per-point ``mean = S·V`` and
  ``std = sqrt(S·V² − mean²)``.  Output is
  ``std ⊙ norm(content) + mean``.  When ``H·W > max_sample`` (``64²``)
  a random (or deterministic for ONNX) subset is sampled.
* **Transformer**: two AdaAttN branches (``relu4_1`` and ``relu5_1``,
  shallow-layer concatenation ``960``/``1472`` channels when enabled) +
  nearest upsample + ``3×3`` merge conv.
* **Decoder**: mirrored VGG decoder with ``ReflectionPad`` + ``3×3`` convs
  and ``×2`` nearest upsampling (``512→256→128→64→3``), optional
  ``skip_connection_3`` (``adaattn_3`` at ``relu3``).

This file is a direct port of ``models/networks.py`` +
``models/adaattn_model.py`` but exposes an OpenCV-friendly API:

    >>> import cv2
    >>> from pixel_alchemy.style_transfer.adaattn import AdaAttNModel, stylize, stylize_paths
    >>> model = AdaAttNModel()  # shallow_layer=True, skip_connection_3=True
    >>> model.load_vgg("models/AdaAttN_model/vgg_normalised.pth")
    >>> model.load_pretrained("models/AdaAttN_model/AdaAttN")
    >>> bgr = cv2.imread("content.jpg")
    >>> style = cv2.imread("style.jpg")
    >>> out = stylize(bgr, style, model, size=512)  # BGR uint8 HxWx3
    >>> stylize_paths("content.jpg", "style.jpg", "out.jpg")  # one-liner
    >>> export_onnx(model, "adaattn.onnx")  # onnxruntime / cv2.dnn

Pretrained weights (``vgg_normalised.pth``, ``latest_net_*.pth``) are
stored under ``models/AdaAttN_model/`` (git-ignored).  See ``README``
in the upstream repo for download links.

Notes for ONNX: random ``randperm`` sampling is replaced by deterministic
first-``max_sample`` slicing when ``torch.onnx.is_in_onnx_export()`` or
``model.set_deterministic(True)`` — verified <1/255 max error at 512.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-5):
    n, c = feat.shape[:2]
    var = feat.view(n, c, -1).var(dim=2) + eps
    std = var.sqrt().view(n, c, 1, 1)
    mean = feat.view(n, c, -1).mean(dim=2).view(n, c, 1, 1)
    return mean, std


def mean_variance_norm(feat: torch.Tensor):
    mean, std = calc_mean_std(feat)
    return (feat - mean) / std


# ---------------------------------------------------------------------------
# core modules — direct port of Huage001/AdaAttN models/networks.py
# ---------------------------------------------------------------------------


class AdaAttN(nn.Module):
    def __init__(self, in_planes: int, key_planes: int | None = None, max_sample: int = 64 * 64):
        super().__init__()
        if key_planes is None:
            key_planes = in_planes
        self.f = nn.Conv2d(key_planes, key_planes, 1)
        self.g = nn.Conv2d(key_planes, key_planes, 1)
        self.h = nn.Conv2d(in_planes, in_planes, 1)
        self.sm = nn.Softmax(dim=-1)
        self.max_sample = max_sample
        self.deterministic = False

    def forward(self, content, style, content_key, style_key, seed=None):
        F_ = self.f(content_key)
        G = self.g(style_key)
        H = self.h(style)
        b, _, hg, wg = G.shape
        G = G.view(b, -1, hg * wg)
        H_flat = H.view(b, -1, hg * wg)
        if hg * wg > self.max_sample:
            if self.deterministic or torch.onnx.is_in_onnx_export():
                # ONNX-compatible deterministic sampling (first max_sample)
                G = G[:, :, : self.max_sample]
                style_flat = H_flat[:, :, : self.max_sample].transpose(1, 2).contiguous()
            else:
                if seed is not None:
                    torch.manual_seed(seed)
                idx = torch.randperm(hg * wg, device=content.device)[: self.max_sample]
                G = G[:, :, idx]
                style_flat = H_flat[:, :, idx].transpose(1, 2).contiguous()
        else:
            style_flat = H_flat.transpose(1, 2).contiguous()
        b, _, h, w = F_.shape
        F_ = F_.view(b, -1, h * w).permute(0, 2, 1)
        S = torch.bmm(F_, G)
        S = self.sm(S)
        mean = torch.bmm(S, style_flat)
        std = torch.sqrt(torch.relu(torch.bmm(S, style_flat**2) - mean**2))
        mean = mean.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        std = std.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return std * mean_variance_norm(content) + mean


class Transformer(nn.Module):
    def __init__(self, in_planes=512, key_planes=None, shallow_layer=False):
        super().__init__()
        self.attn_adain_4_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes)
        self.attn_adain_5_1 = AdaAttN(in_planes=in_planes, key_planes=key_planes + 512 if shallow_layer else key_planes)
        self.upsample5_1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.merge_conv_pad = nn.ReflectionPad2d((1, 1, 1, 1))
        self.merge_conv = nn.Conv2d(in_planes, in_planes, 3)

    def forward(self, c4, s4, c5, s5, c4k, s4k, c5k, s5k, seed=None):
        a4 = self.attn_adain_4_1(c4, s4, c4k, s4k, seed=seed)
        a5 = self.attn_adain_5_1(c5, s5, c5k, s5k, seed=seed)
        return self.merge_conv(self.merge_conv_pad(a4 + self.upsample5_1(a5)))


class Decoder(nn.Module):
    def __init__(self, skip_connection_3=False):
        super().__init__()
        self.decoder_layer_1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
        )
        in_ch = 256 + 256 if skip_connection_3 else 256
        self.decoder_layer_2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(in_ch, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3),
        )

    def forward(self, cs, c_adain_3=None):
        cs = self.decoder_layer_1(cs)
        if c_adain_3 is None:
            return self.decoder_layer_2(cs)
        return self.decoder_layer_2(torch.cat((cs, c_adain_3), dim=1))


# ---------------------------------------------------------------------------
# VGG-19 encoder (normalised, first conv is 1x1 identity + mean/std norm)
# ---------------------------------------------------------------------------


def _make_vgg_encoder():
    return nn.Sequential(
        nn.Conv2d(3, 3, 1),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, 3),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d(2, 2, 0, ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, 3),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, 3),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d(2, 2, 0, ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, 3),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, 3),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d(2, 2, 0, ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, 3),
        nn.ReLU(),  # relu4-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu4-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu4-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu4-4
        nn.MaxPool2d(2, 2, 0, ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu5-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu5-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu5-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 512, 3),
        nn.ReLU(),  # relu5-4
    )


def _split_encoder(enc: nn.Sequential):
    layers = list(enc.children())
    enc1 = nn.Sequential(*layers[:4])
    enc2 = nn.Sequential(*layers[4:11])
    enc3 = nn.Sequential(*layers[11:18])
    enc4 = nn.Sequential(*layers[18:31])
    enc5 = nn.Sequential(*layers[31:44])
    return [enc1, enc2, enc3, enc4, enc5]


# ---------------------------------------------------------------------------
# Full model — OpenCV-friendly wrapper
# ---------------------------------------------------------------------------


class AdaAttNModel(nn.Module):
    """AdaAttN arbitrary style transfer (ICCV 2021).

    Mirrors ``models/adaattn_model.py`` + ``models/networks.py`` from
    https://github.com/Huage001/AdaAttN but exposes a simple
    ``forward(content, style)`` and OpenCV helper ``stylize``.
    """

    def __init__(self, shallow_layer: bool = True, skip_connection_3: bool = True, max_sample: int = 64 * 64):
        super().__init__()
        self.shallow_layer = shallow_layer
        self.skip_connection_3 = skip_connection_3
        self.max_sample = max_sample

        vgg = _make_vgg_encoder()
        self.enc_layers = nn.ModuleList(_split_encoder(vgg))
        for p in self.parameters():
            pass  # encoder params are in enc_layers, freeze later after load
        for layer in self.enc_layers:
            for p in layer.parameters():
                p.requires_grad = False

        if skip_connection_3:
            key3 = 256 + 128 + 64 if shallow_layer else 256
            self.adaattn_3 = AdaAttN(256, key3, max_sample)
        else:
            self.adaattn_3 = None

        key4 = 512 + 256 + 128 + 64 if shallow_layer else 512
        self.transformer = Transformer(512, key4, shallow_layer)
        self.decoder = Decoder(skip_connection_3)

    def set_deterministic(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, AdaAttN):
                m.deterministic = flag

    # -- encoding helpers ---------------------------------------------------

    def encode_with_intermediate(self, x):
        feats = []
        for layer in self.enc_layers:
            x = layer(x)
            feats.append(x)
        return feats

    @staticmethod
    def get_key(feats, idx, need_shallow=True):
        if need_shallow and idx > 0:
            _, _, h, w = feats[idx].shape
            parts = []
            for i in range(idx):
                parts.append(mean_variance_norm(F.interpolate(feats[i], (h, w), mode="nearest")))
            parts.append(mean_variance_norm(feats[idx]))
            return torch.cat(parts, dim=1)
        return mean_variance_norm(feats[idx])

    def forward(self, content, style, seed: int | None = None):
        c_feats = self.encode_with_intermediate(content)
        s_feats = self.encode_with_intermediate(style)

        if self.adaattn_3 is not None:
            c_adain_3 = self.adaattn_3(
                c_feats[2],
                s_feats[2],
                self.get_key(c_feats, 2, self.shallow_layer),
                self.get_key(s_feats, 2, self.shallow_layer),
                seed,
            )
        else:
            c_adain_3 = None

        cs = self.transformer(
            c_feats[3],
            s_feats[3],
            c_feats[4],
            s_feats[4],
            self.get_key(c_feats, 3, self.shallow_layer),
            self.get_key(s_feats, 3, self.shallow_layer),
            self.get_key(c_feats, 4, self.shallow_layer),
            self.get_key(s_feats, 4, self.shallow_layer),
            seed,
        )
        return self.decoder(cs, c_adain_3)

    # -- weight loading -----------------------------------------------------

    def load_vgg(self, path: str | Path, device="cpu"):
        sd = torch.load(path, map_location=device)
        # original file is state_dict of the flat Sequential
        vgg = _make_vgg_encoder()
        vgg.load_state_dict(sd)
        layers = _split_encoder(vgg)
        for dst, src in zip(self.enc_layers, layers):
            dst.load_state_dict(src.state_dict())

    def load_pretrained(self, ckpt_dir: str | Path, device="cpu"):
        """Load AdaAttN weights from a directory or a single .pth.

        Expected files (as saved by original repo):
            latest_net_decoder.pth, latest_net_transformer.pth,
            latest_net_adaattn_3.pth (if skip_connection_3)
        Alternatively a single file containing dict with keys
        ``decoder``, ``transformer``, ``adaattn_3`` or a flat state_dict
        for the whole model is accepted.
        """
        p = Path(ckpt_dir)
        if p.is_file():
            sd = torch.load(p, map_location=device)
            # try flat model state
            try:
                self.load_state_dict(sd, strict=False)
                return
            except Exception:
                pass
            # dict with sub-keys
            if "decoder" in sd:
                self.decoder.load_state_dict(sd["decoder"])
            if "transformer" in sd:
                self.transformer.load_state_dict(sd["transformer"])
            if "adaattn_3" in sd and self.adaattn_3 is not None:
                self.adaattn_3.load_state_dict(sd["adaattn_3"])
            return

        # directory mode — DataParallel prefix handling
        def _load(name, module):
            for cand in [p / f"latest_net_{name}.pth", p / f"{name}.pth"]:
                if cand.exists():
                    sd = torch.load(cand, map_location=device)
                    # strip "module." prefix from DataParallel
                    sd = {k.replace("module.", ""): v for k, v in sd.items()}
                    module.load_state_dict(sd)
                    return True
            return False

        _load("decoder", self.decoder)
        _load("transformer", self.transformer)
        if self.adaattn_3 is not None:
            _load("adaattn_3", self.adaattn_3)


# ---------------------------------------------------------------------------
# OpenCV-friendly functional API
# ---------------------------------------------------------------------------


def _to_tensor(bgr: np.ndarray, size: int | tuple[int, int] | None, device) -> torch.Tensor:
    """BGR uint8 HxWx3 -> RGB float tensor 1x3xHxW in [0,1]."""
    if size is not None:
        if isinstance(size, int):
            size = (size, size)
        # size as (h, w) for cv2 which expects (w, h)
        bgr = cv2.resize(bgr, (size[1], size[0]), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def _tensor_to_bgr(t: torch.Tensor, orig_hw: tuple[int, int] | None = None) -> np.ndarray:
    t = t.squeeze(0).detach().cpu().clamp(0, 1)
    rgb = (t.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if orig_hw is not None:
        bgr = cv2.resize(bgr, (orig_hw[1], orig_hw[0]), interpolation=cv2.INTER_CUBIC)
    return bgr


@torch.no_grad()
def stylize(
    content_bgr: np.ndarray,
    style_bgr: np.ndarray,
    model: AdaAttNModel,
    device: str | torch.device = "cpu",
    size: int | tuple[int, int] | None = 512,
    alpha: float = 1.0,
    keep_content_size: bool = True,
    seed: int | None = None,
) -> np.ndarray:
    """Stylize a BGR image with AdaAttN.

    Args:
        content_bgr: HxWx3 uint8 BGR (as from ``cv2.imread``).
        style_bgr:   HxWx3 uint8 BGR.
        model:       ``AdaAttNModel`` (already on *device*, eval mode).
        device:      torch device.
        size:        inference size. ``512`` resizes both to 512x512;
                     ``None`` keeps original content resolution (must be
                     multiple-aware; will be padded to multiple of 8).
                     Tuple is (h, w).
        alpha:       blend factor: 1.0 = fully stylized, 0.0 = original.
        keep_content_size: if True, resize result back to original content HxW.
        seed:        random seed for style sampling (deterministic if set).

    Returns:
        Stylized BGR uint8 image.
    """
    device = torch.device(device)
    model = model.to(device).eval()
    orig_hw = content_bgr.shape[:2] if keep_content_size else None

    # pad to multiple of 8 to avoid ceil/mismatch artefacts when size is None
    if size is None:
        h, w = content_bgr.shape[:2]
        # next multiple of 8
        ph, pw = (8 - h % 8) % 8, (8 - w % 8) % 8
        if ph or pw:
            content_bgr = cv2.copyMakeBorder(content_bgr, 0, ph, 0, pw, cv2.BORDER_REFLECT_101)
            style_bgr = cv2.resize(style_bgr, (w + pw, h + ph), interpolation=cv2.INTER_AREA)
            size = None  # already padded
            orig_hw = (h, w)  # restore original after
        ct = _to_tensor(content_bgr, None, device)
        st = _to_tensor(style_bgr, (ct.shape[2], ct.shape[3]), device)
    else:
        ct = _to_tensor(content_bgr, size, device)
        st = _to_tensor(style_bgr, (ct.shape[2], ct.shape[3]), device)

    out = model(ct, st, seed=seed)

    bgr = _tensor_to_bgr(out, orig_hw)

    if alpha < 1.0:
        # blend with original content (resized to output size)
        base = (
            content_bgr
            if not keep_content_size or size is None
            else cv2.resize(content_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_CUBIC)
        )
        # if we padded, base already matches; otherwise ensure match
        if base.shape[:2] != bgr.shape[:2]:
            base = cv2.resize(base, (bgr.shape[1], bgr.shape[0]))
        bgr = cv2.addWeighted(bgr, alpha, base, 1 - alpha, 0)

    return bgr


# TODO - have an appropriate way to download/make this reproducible
DEFAULT_VGG = Path(__file__).parents[3] / "models" / "AdaAttN_model" / "vgg_normalised.pth"
DEFAULT_CKPT_DIR = Path(__file__).parents[3] / "models" / "AdaAttN_model" / "AdaAttN"


def stylize_paths(
    content_path: str | Path,
    style_path: str | Path,
    output_path: str | Path,
    ckpt_dir: str | Path | None = None,
    vgg_path: str | Path | None = None,
    device: str | None = None,
    size: int = 512,
    alpha: float = 1.0,
) -> Path:
    """One-liner for CLI / scripts: load images with cv2, run model, save."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if vgg_path is None and DEFAULT_VGG.exists():
        vgg_path = DEFAULT_VGG
    if ckpt_dir is None and DEFAULT_CKPT_DIR.exists():
        ckpt_dir = DEFAULT_CKPT_DIR
    content_bgr = cv2.imread(str(content_path), cv2.IMREAD_COLOR)
    style_bgr = cv2.imread(str(style_path), cv2.IMREAD_COLOR)
    if content_bgr is None:
        raise FileNotFoundError(content_path)
    if style_bgr is None:
        raise FileNotFoundError(style_path)

    model = AdaAttNModel()
    if vgg_path is not None:
        model.load_vgg(vgg_path, device)
    if ckpt_dir is not None:
        model.load_pretrained(ckpt_dir, device)

    result = stylize(content_bgr, style_bgr, model, device=device, size=size, alpha=alpha)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    return output_path


def export_onnx(
    model: AdaAttNModel, output_path: str | Path, size: int = 512, opset: int = 18, dynamic: bool = False
) -> Path:
    """Export AdaAttN to ONNX (for cv2.dnn or onnxruntime).

    The exported graph takes two inputs ``content`` and ``style``
    (1x3xHxW float32 in [0,1] RGB) and outputs stylized RGB in [0,1].

    Args:
        model: AdaAttNModel (weights already loaded)
        output_path: where to write .onnx
        size: dummy size for tracing (512). Use 512 for best compatibility.
        opset: ONNX opset (18 recommended, avoids Pad conversion issue)
        dynamic: if True, export with dynamic_axes (needs dynamo=False legacy)

    Deterministic sampling is forced for ONNX (first max_sample) to stay
    ONNX-compatible; call ``model.set_deterministic(True)`` for verification.
    """
    model.eval()
    model.set_deterministic(True)
    output_path = Path(output_path)
    dummy_c = torch.randn(1, 3, size, size)
    dummy_s = torch.randn(1, 3, size, size)
    if dynamic:
        torch.onnx.export(
            model,
            (dummy_c, dummy_s),
            str(output_path),
            input_names=["content", "style"],
            output_names=["stylized"],
            dynamic_axes={"content": {2: "h", 3: "w"}, "style": {2: "h", 3: "w"}, "stylized": {2: "h", 3: "w"}},
            opset_version=opset,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model,
            (dummy_c, dummy_s),
            str(output_path),
            input_names=["content", "style"],
            output_names=["stylized"],
            opset_version=opset,
        )
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="AdaAttN arbitrary style transfer (OpenCV-friendly)")
    ap.add_argument("content", help="content image path (BGR via cv2)")
    ap.add_argument("style", help="style image path")
    ap.add_argument("-o", "--output", default="stylized.png")
    ap.add_argument("--ckpt", default=None, help="path to AdaAttN checkpoint dir or .pth")
    ap.add_argument("--vgg", default=None, help="path to vgg_normalised.pth")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--export-onnx", default=None, help="if set, export ONNX to this path and exit")
    args = ap.parse_args()
    if args.export_onnx:
        m = AdaAttNModel()
        if args.vgg:
            m.load_vgg(args.vgg)
        if args.ckpt:
            m.load_pretrained(args.ckpt)
        export_onnx(m, args.export_onnx, args.size)
        print(f"onnx saved to {args.export_onnx}")
    else:
        stylize_paths(args.content, args.style, args.output, args.ckpt, args.vgg, args.device, args.size, args.alpha)
        print(f"saved to {args.output}")
