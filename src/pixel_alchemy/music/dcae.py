# Vendored ACE DCAE + Vocoder (without ComfyUI deps)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List
import math, logging
try:
    import torchaudio
    from torchaudio.transforms import MelScale
except Exception:
    logging.warning('torchaudio missing, MusicDCAE broken')
    torchaudio = None

    class MelScale(nn.Module):  # type: ignore
        def __init__(self, *a, **kw):
            super().__init__()

        def forward(self, x):
            return x

try:
    import torchvision.transforms as transforms
except Exception:
    transforms = None
    class _DummyTransforms:
        @staticmethod
        def Compose(x):
            return nn.Identity()

        class Normalize(nn.Module):
            def __init__(self, *a, **kw):
                super().__init__()

            def forward(self, x):
                return (x - 0.5) / 0.5

    transforms = _DummyTransforms  # type: ignore
from functools import partial
from math import prod
from typing import Callable
import numpy as np
from torch.nn.utils.parametrize import remove_parametrizations as remove_weight_norm

def _cast(x, dtype=None, device=None):
    kwargs={}
    if dtype is not None:
        kwargs['dtype']=dtype
    if device is not None:
        kwargs['device']=device
    if kwargs:
        return x.to(**kwargs)
    return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, bias=False):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        else:
            self.weight = None
            self.bias = None
    def forward(self, x):
        t = x.float()
        mean_sq = t.pow(2).mean(-1, keepdim=True)
        t = t * torch.rsqrt(mean_sq + self.eps)
        t = t.to(x.dtype)
        if self.elementwise_affine and self.weight is not None:
            t = t * self.weight
            if self.bias is not None:
                t = t + self.bias
        return t

# --- from autoencoder_dc.py ---
# Rewritten from diffusers
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union



class RMSNorm(RMSNorm):
    def __init__(self, dim, eps=1e-5, elementwise_affine=True, bias=False):
        super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)
        if elementwise_affine:
            self.bias = nn.Parameter(torch.empty(dim)) if bias else None

    def forward(self, x):
        x = super().forward(x)
        if self.elementwise_affine:
            if self.bias is not None:
                x = x + _cast(self.bias, dtype=x.dtype, device=x.device)
        return x


def get_normalization(norm_type, num_features, num_groups=32, eps=1e-5):
    if norm_type == "batch_norm":
        return nn.BatchNorm2d(num_features)
    elif norm_type == "group_norm":
        return nn.GroupNorm(num_groups, num_features)
    elif norm_type == "layer_norm":
        return nn.LayerNorm(num_features)
    elif norm_type == "rms_norm":
        return RMSNorm(num_features, eps=eps, elementwise_affine=True, bias=True)
    else:
        raise ValueError(f"Unknown normalization type: {norm_type}")


def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "silu":
        return nn.SiLU()
    elif activation_type == "leaky_relu":
        return nn.LeakyReLU(0.2)
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch_norm",
        act_fn: str = "relu6",
    ) -> None:
        super().__init__()

        self.norm_type = norm_type
        self.nonlinearity = get_activation(act_fn) if act_fn is not None else nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.norm = get_normalization(norm_type, out_channels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm(hidden_states)

        return hidden_states + residual

class SanaMultiscaleAttentionProjection(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_attention_heads: int,
        kernel_size: int,
    ) -> None:
        super().__init__()

        channels = 3 * in_channels
        self.proj_in = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
            bias=False,
        )
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0, groups=3 * num_attention_heads, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj_in(hidden_states)
        hidden_states = self.proj_out(hidden_states)
        return hidden_states

class SanaMultiscaleLinearAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_attention_heads: int = None,
        attention_head_dim: int = 8,
        mult: float = 1.0,
        norm_type: str = "batch_norm",
        kernel_sizes: tuple = (5,),
        eps: float = 1e-15,
        residual_connection: bool = False,
    ):
        super().__init__()

        self.eps = eps
        self.attention_head_dim = attention_head_dim
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        num_attention_heads = (
            int(in_channels // attention_head_dim * mult)
            if num_attention_heads is None
            else num_attention_heads
        )
        inner_dim = num_attention_heads * attention_head_dim

        self.to_q = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, inner_dim, bias=False)

        self.to_qkv_multiscale = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.to_qkv_multiscale.append(
                SanaMultiscaleAttentionProjection(inner_dim, num_attention_heads, kernel_size)
            )

        self.nonlinearity = nn.ReLU()
        self.to_out = nn.Linear(inner_dim * (1 + len(kernel_sizes)), out_channels, bias=False)
        self.norm_out = get_normalization(norm_type, out_channels)

    def apply_linear_attention(self, query, key, value):
        value = F.pad(value, (0, 0, 0, 1), mode="constant", value=1)
        scores = torch.matmul(value, key.transpose(-1, -2))
        hidden_states = torch.matmul(scores, query)

        hidden_states = hidden_states.to(dtype=torch.float32)
        hidden_states = hidden_states[:, :, :-1] / (hidden_states[:, :, -1:] + self.eps)
        return hidden_states

    def apply_quadratic_attention(self, query, key, value):
        scores = torch.matmul(key.transpose(-1, -2), query)
        scores = scores.to(dtype=torch.float32)
        scores = scores / (torch.sum(scores, dim=2, keepdim=True) + self.eps)
        hidden_states = torch.matmul(value, scores.to(value.dtype))
        return hidden_states

    def forward(self, hidden_states):
        height, width = hidden_states.shape[-2:]
        if height * width > self.attention_head_dim:
            use_linear_attention = True
        else:
            use_linear_attention = False

        residual = hidden_states

        batch_size, _, height, width = list(hidden_states.size())
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.movedim(1, -1)
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        hidden_states = torch.cat([query, key, value], dim=3)
        hidden_states = hidden_states.movedim(-1, 1)

        multi_scale_qkv = [hidden_states]
        for block in self.to_qkv_multiscale:
            multi_scale_qkv.append(block(hidden_states))

        hidden_states = torch.cat(multi_scale_qkv, dim=1)

        if use_linear_attention:
            # for linear attention upcast hidden_states to float32
            hidden_states = hidden_states.to(dtype=torch.float32)

        hidden_states = hidden_states.reshape(batch_size, -1, 3 * self.attention_head_dim, height * width)

        query, key, value = hidden_states.chunk(3, dim=2)
        query = self.nonlinearity(query)
        key = self.nonlinearity(key)

        if use_linear_attention:
            hidden_states = self.apply_linear_attention(query, key, value)
            hidden_states = hidden_states.to(dtype=original_dtype)
        else:
            hidden_states = self.apply_quadratic_attention(query, key, value)

        hidden_states = torch.reshape(hidden_states, (batch_size, -1, height, width))
        hidden_states = self.to_out(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.norm_type == "rms_norm":
            hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        else:
            hidden_states = self.norm_out(hidden_states)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mult: float = 1.0,
        attention_head_dim: int = 32,
        qkv_multiscales: tuple = (5,),
        norm_type: str = "batch_norm",
    ) -> None:
        super().__init__()

        self.attn = SanaMultiscaleLinearAttention(
            in_channels=in_channels,
            out_channels=in_channels,
            mult=mult,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            kernel_sizes=qkv_multiscales,
            residual_connection=True,
        )

        self.conv_out = GLUMBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            norm_type="rms_norm",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        x = self.conv_out(x)
        return x


class GLUMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 4,
        norm_type: str = None,
        residual_connection: bool = True,
    ) -> None:
        super().__init__()

        hidden_channels = int(expand_ratio * in_channels)
        self.norm_type = norm_type
        self.residual_connection = residual_connection

        self.nonlinearity = nn.SiLU()
        self.conv_inverted = nn.Conv2d(in_channels, hidden_channels * 2, 1, 1, 0)
        self.conv_depth = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, 3, 1, 1, groups=hidden_channels * 2)
        self.conv_point = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0, bias=False)

        self.norm = None
        if norm_type == "rms_norm":
            self.norm = RMSNorm(out_channels, eps=1e-5, elementwise_affine=True, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.residual_connection:
            residual = hidden_states

        hidden_states = self.conv_inverted(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv_depth(hidden_states)
        hidden_states, gate = torch.chunk(hidden_states, 2, dim=1)
        hidden_states = hidden_states * self.nonlinearity(gate)

        hidden_states = self.conv_point(hidden_states)

        if self.norm_type == "rms_norm":
            # move channel to the last dimension so we apply RMSnorm across channel dimension
            hidden_states = self.norm(hidden_states.movedim(1, -1)).movedim(-1, 1)

        if self.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states


def get_block(
    block_type: str,
    in_channels: int,
    out_channels: int,
    attention_head_dim: int,
    norm_type: str,
    act_fn: str,
    qkv_mutliscales: tuple = (),
):
    if block_type == "ResBlock":
        block = ResBlock(in_channels, out_channels, norm_type, act_fn)
    elif block_type == "EfficientViTBlock":
        block = EfficientViTBlock(
            in_channels,
            attention_head_dim=attention_head_dim,
            norm_type=norm_type,
            qkv_multiscales=qkv_mutliscales
        )
    else:
        raise ValueError(f"Block with {block_type=} is not supported.")

    return block


class DCDownBlock2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, downsample: bool = False, shortcut: bool = True) -> None:
        super().__init__()

        self.downsample = downsample
        self.factor = 2
        self.stride = 1 if downsample else 2
        self.group_size = in_channels * self.factor**2 // out_channels
        self.shortcut = shortcut

        out_ratio = self.factor**2
        if downsample:
            assert out_channels % out_ratio == 0
            out_channels = out_channels // out_ratio

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.conv(hidden_states)
        if self.downsample:
            x = F.pixel_unshuffle(x, self.factor)

        if self.shortcut:
            y = F.pixel_unshuffle(hidden_states, self.factor)
            y = y.unflatten(1, (-1, self.group_size))
            y = y.mean(dim=2)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class DCUpBlock2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        interpolate: bool = False,
        shortcut: bool = True,
        interpolation_mode: str = "nearest",
    ) -> None:
        super().__init__()

        self.interpolate = interpolate
        self.interpolation_mode = interpolation_mode
        self.shortcut = shortcut
        self.factor = 2
        self.repeats = out_channels * self.factor**2 // in_channels

        out_ratio = self.factor**2
        if not interpolate:
            out_channels = out_channels * out_ratio

        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.interpolate:
            x = F.interpolate(hidden_states, scale_factor=self.factor, mode=self.interpolation_mode)
            x = self.conv(x)
        else:
            x = self.conv(hidden_states)
            x = F.pixel_shuffle(x, self.factor)

        if self.shortcut:
            y = hidden_states.repeat_interleave(self.repeats, dim=1, output_size=hidden_states.shape[1] * self.repeats)
            y = F.pixel_shuffle(y, self.factor)
            hidden_states = x + y
        else:
            hidden_states = x

        return hidden_states


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: str or tuple = "ResBlock",
        block_out_channels: tuple = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: tuple = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: tuple = ((), (), (), (5,), (5,), (5,)),
        downsample_block_type: str = "pixel_unshuffle",
        out_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks

        if layers_per_block[0] > 0:
            self.conv_in = nn.Conv2d(
                in_channels,
                block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                kernel_size=3,
                stride=1,
                padding=1,
            )
        else:
            self.conv_in = DCDownBlock2d(
                in_channels=in_channels,
                out_channels=block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1],
                downsample=downsample_block_type == "pixel_unshuffle",
                shortcut=False,
            )

        down_blocks = []
        for i, (out_channel, num_layers) in enumerate(zip(block_out_channels, layers_per_block)):
            down_block_list = []

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type="rms_norm",
                    act_fn="silu",
                    qkv_mutliscales=qkv_multiscales[i],
                )
                down_block_list.append(block)

            if i < num_blocks - 1 and num_layers > 0:
                downsample_block = DCDownBlock2d(
                    in_channels=out_channel,
                    out_channels=block_out_channels[i + 1],
                    downsample=downsample_block_type == "pixel_unshuffle",
                    shortcut=True,
                )
                down_block_list.append(downsample_block)

            down_blocks.append(nn.Sequential(*down_block_list))

        self.down_blocks = nn.ModuleList(down_blocks)

        self.conv_out = nn.Conv2d(block_out_channels[-1], latent_channels, 3, 1, 1)

        self.out_shortcut = out_shortcut
        if out_shortcut:
            self.out_shortcut_average_group_size = block_out_channels[-1] // latent_channels

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        if self.out_shortcut:
            x = hidden_states.unflatten(1, (-1, self.out_shortcut_average_group_size))
            x = x.mean(dim=2)
            hidden_states = self.conv_out(hidden_states) + x
        else:
            hidden_states = self.conv_out(hidden_states)

        return hidden_states


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        latent_channels: int,
        attention_head_dim: int = 32,
        block_type: str or tuple = "ResBlock",
        block_out_channels: tuple = (128, 256, 512, 512, 1024, 1024),
        layers_per_block: tuple = (2, 2, 2, 2, 2, 2),
        qkv_multiscales: tuple = ((), (), (), (5,), (5,), (5,)),
        norm_type: str or tuple = "rms_norm",
        act_fn: str or tuple = "silu",
        upsample_block_type: str = "pixel_shuffle",
        in_shortcut: bool = True,
    ):
        super().__init__()

        num_blocks = len(block_out_channels)

        if isinstance(block_type, str):
            block_type = (block_type,) * num_blocks
        if isinstance(norm_type, str):
            norm_type = (norm_type,) * num_blocks
        if isinstance(act_fn, str):
            act_fn = (act_fn,) * num_blocks

        self.conv_in = nn.Conv2d(latent_channels, block_out_channels[-1], 3, 1, 1)

        self.in_shortcut = in_shortcut
        if in_shortcut:
            self.in_shortcut_repeats = block_out_channels[-1] // latent_channels

        up_blocks = []
        for i, (out_channel, num_layers) in reversed(list(enumerate(zip(block_out_channels, layers_per_block)))):
            up_block_list = []

            if i < num_blocks - 1 and num_layers > 0:
                upsample_block = DCUpBlock2d(
                    block_out_channels[i + 1],
                    out_channel,
                    interpolate=upsample_block_type == "interpolate",
                    shortcut=True,
                )
                up_block_list.append(upsample_block)

            for _ in range(num_layers):
                block = get_block(
                    block_type[i],
                    out_channel,
                    out_channel,
                    attention_head_dim=attention_head_dim,
                    norm_type=norm_type[i],
                    act_fn=act_fn[i],
                    qkv_mutliscales=qkv_multiscales[i],
                )
                up_block_list.append(block)

            up_blocks.insert(0, nn.Sequential(*up_block_list))

        self.up_blocks = nn.ModuleList(up_blocks)

        channels = block_out_channels[0] if layers_per_block[0] > 0 else block_out_channels[1]

        self.norm_out = RMSNorm(channels, 1e-5, elementwise_affine=True, bias=True)
        self.conv_act = nn.ReLU()
        self.conv_out = None

        if layers_per_block[0] > 0:
            self.conv_out = nn.Conv2d(channels, in_channels, 3, 1, 1)
        else:
            self.conv_out = DCUpBlock2d(
                channels, in_channels, interpolate=upsample_block_type == "interpolate", shortcut=False
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.in_shortcut:
            x = hidden_states.repeat_interleave(
                self.in_shortcut_repeats, dim=1, output_size=hidden_states.shape[1] * self.in_shortcut_repeats
            )
            hidden_states = self.conv_in(hidden_states) + x
        else:
            hidden_states = self.conv_in(hidden_states)

        for up_block in reversed(self.up_blocks):
            hidden_states = up_block(hidden_states)

        hidden_states = self.norm_out(hidden_states.movedim(1, -1)).movedim(-1, 1)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)
        return hidden_states


class AutoencoderDC(nn.Module):
    def __init__(
        self,
        in_channels: int = 2,
        latent_channels: int = 8,
        attention_head_dim: int = 32,
        encoder_block_types: Union[str, Tuple[str]] = ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
        decoder_block_types: Union[str, Tuple[str]] = ["ResBlock", "ResBlock", "ResBlock", "EfficientViTBlock"],
        encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 1024),
        encoder_layers_per_block: Tuple[int] = (2, 2, 3, 3),
        decoder_layers_per_block: Tuple[int] = (3, 3, 3, 3),
        encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,), (5,)),
        decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = ((), (), (5,), (5,)),
        upsample_block_type: str = "interpolate",
        downsample_block_type: str = "Conv",
        decoder_norm_types: Union[str, Tuple[str]] = "rms_norm",
        decoder_act_fns: Union[str, Tuple[str]] = "silu",
        scaling_factor: float = 0.41407,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=encoder_block_types,
            block_out_channels=encoder_block_out_channels,
            layers_per_block=encoder_layers_per_block,
            qkv_multiscales=encoder_qkv_multiscales,
            downsample_block_type=downsample_block_type,
        )

        self.decoder = Decoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            attention_head_dim=attention_head_dim,
            block_type=decoder_block_types,
            block_out_channels=decoder_block_out_channels,
            layers_per_block=decoder_layers_per_block,
            qkv_multiscales=decoder_qkv_multiscales,
            norm_type=decoder_norm_types,
            act_fn=decoder_act_fns,
            upsample_block_type=upsample_block_type,
        )

        self.scaling_factor = scaling_factor
        self.spatial_compression_ratio = 2 ** (len(encoder_block_out_channels) - 1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Internal encoding function."""
        encoded = self.encoder(x)
        return encoded * self.scaling_factor

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Scale the latents back
        z = z / self.scaling_factor
        decoded = self.decoder(z)
        return decoded

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# --- from music_log_mel.py ---
# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_log_mel.py
import torch.nn as nn
import logging



class LinearSpectrogram(nn.Module):
    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        mode="pow2_sqrt",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode

        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, y: Tensor) -> Tensor:
        if y.ndim == 3:
            y = y.squeeze(1)

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                (self.win_length - self.hop_length) // 2,
                (self.win_length - self.hop_length + 1) // 2,
            ),
            mode="reflect",
        ).squeeze(1)
        dtype = y.dtype
        spec = torch.stft(
            y.float(),
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=_cast(self.window, dtype=torch.float32, device=y.device),
            center=self.center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)

        if self.mode == "pow2_sqrt":
            spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
        spec = spec.to(dtype)
        return spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or sample_rate // 2

        self.spectrogram = LinearSpectrogram(n_fft, win_length, hop_length, center)
        self.mel_scale = MelScale(
            self.n_mels,
            self.sample_rate,
            self.f_min,
            self.f_max,
            self.n_fft // 2 + 1,
            "slaney",
            "slaney",
        )

    def compress(self, x: Tensor) -> Tensor:
        return torch.log(torch.clamp(x, min=1e-5))

    def decompress(self, x: Tensor) -> Tensor:
        return torch.exp(x)

    def forward(self, x: Tensor, return_linear: bool = False) -> Tensor:
        linear = self.spectrogram(x)
        x = self.mel_scale(linear)
        x = self.compress(x)
        # print(x.shape)
        if return_linear:
            return x, self.compress(linear)

        return x

# --- from music_vocoder.py ---
# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_vocoder.py

from functools import partial
from math import prod
from typing import Callable, Tuple, List

import numpy as np
import torch.nn.functional as F




def drop_path(
    x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True
):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """  # noqa: E501

    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""  # noqa: E501

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


class LayerNorm(nn.Module):
    r"""LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """  # noqa: E501

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x, self.normalized_shape, _cast(self.weight, dtype=x.dtype, device=x.device), _cast(self.bias, dtype=x.dtype, device=x.device), self.eps
            )
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = _cast(self.weight[:, None], dtype=x.dtype, device=x.device) * x + _cast(self.bias[:, None], dtype=x.dtype, device=x.device)
            return x


class ConvNeXtBlock(nn.Module):
    r"""ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        kernel_size (int): Kernel size for depthwise conv. Default: 7.
        dilation (int): Dilation for depthwise conv. Default: 1.
    """  # noqa: E501

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        mlp_ratio: float = 4.0,
        kernel_size: int = 7,
        dilation: int = 1,
    ):
        super().__init__()

        self.dwconv = nn.Conv1d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=int(dilation * (kernel_size - 1) / 2),
            groups=dim,
        )  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(
            dim, int(mlp_ratio * dim)
        )  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(int(mlp_ratio * dim), dim)
        self.gamma = (
            nn.Parameter(torch.empty((dim)), requires_grad=False)
            if layer_scale_init_value > 0
            else None
        )
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, apply_residual: bool = True):
        input = x

        x = self.dwconv(x)
        x = x.permute(0, 2, 1)  # (N, C, L) -> (N, L, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if self.gamma is not None:
            x = _cast(self.gamma, dtype=x.dtype, device=x.device) * x

        x = x.permute(0, 2, 1)  # (N, L, C) -> (N, C, L)
        x = self.drop_path(x)

        if apply_residual:
            x = input + x

        return x


class ParallelConvNeXtBlock(nn.Module):
    def __init__(self, kernel_sizes: List[int], *args, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                ConvNeXtBlock(kernel_size=kernel_size, *args, **kwargs)
                for kernel_size in kernel_sizes
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [block(x, apply_residual=False) for block in self.blocks] + [x],
            dim=1,
        ).sum(dim=1)


class ConvNeXtEncoder(nn.Module):
    def __init__(
        self,
        input_channels=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.0,
        layer_scale_init_value=1e-6,
        kernel_sizes: Tuple[int] = (7,),
    ):
        super().__init__()
        assert len(depths) == len(dims)

        self.channel_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(
                input_channels,
                dims[0],
                kernel_size=7,
                padding=3,
                padding_mode="replicate",
            ),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.channel_layers.append(stem)

        for i in range(len(depths) - 1):
            mid_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=1),
            )
            self.channel_layers.append(mid_layer)

        block_fn = (
            partial(ConvNeXtBlock, kernel_size=kernel_sizes[0])
            if len(kernel_sizes) == 1
            else partial(ParallelConvNeXtBlock, kernel_sizes=kernel_sizes)
        )

        self.stages = nn.ModuleList()
        drop_path_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(
                *[
                    block_fn(
                        dim=dims[i],
                        drop_path=drop_path_rates[cur + j],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                    for j in range(depths[i])
                ]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = LayerNorm(dims[-1], eps=1e-6, data_format="channels_first")

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        for channel_layer, stage in zip(self.channel_layers, self.stages):
            x = channel_layer(x)
            x = stage(x)

        return self.norm(x)


def get_padding(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


class ResBlock1(torch.nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()

        self.convs1 = nn.ModuleList(
            [
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[0],
                        padding=get_padding(kernel_size, dilation[0]),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[1],
                        padding=get_padding(kernel_size, dilation[1]),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation[2],
                        padding=get_padding(kernel_size, dilation[2]),
                    )
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
                torch.nn.utils.parametrizations.weight_norm(
                    nn.Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                ),
            ]
        )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.silu(x)
            xt = c1(xt)
            xt = F.silu(xt)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for conv in self.convs1:
            remove_weight_norm(conv)
        for conv in self.convs2:
            remove_weight_norm(conv)


class HiFiGANGenerator(nn.Module):
    def __init__(
        self,
        *,
        hop_length: int = 512,
        upsample_rates: Tuple[int] = (8, 8, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (16, 16, 8, 2, 2),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11),
        resblock_dilation_sizes: Tuple[Tuple[int]] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 128,
        upsample_initial_channel: int = 512,
        use_template: bool = True,
        pre_conv_kernel_size: int = 7,
        post_conv_kernel_size: int = 7,
        post_activation: Callable = partial(nn.SiLU, inplace=True),
    ):
        super().__init__()

        assert (
            prod(upsample_rates) == hop_length
        ), f"hop_length must be {prod(upsample_rates)}"

        self.conv_pre = torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                num_mels,
                upsample_initial_channel,
                pre_conv_kernel_size,
                1,
                padding=get_padding(pre_conv_kernel_size),
            )
        )

        self.num_upsamples = len(upsample_rates)
        self.num_kernels = len(resblock_kernel_sizes)

        self.noise_convs = nn.ModuleList()
        self.use_template = use_template
        self.ups = nn.ModuleList()

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c_cur = upsample_initial_channel // (2 ** (i + 1))
            self.ups.append(
                torch.nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

            if not use_template:
                continue

            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1:])
                self.noise_convs.append(
                    nn.Conv1d(
                        1,
                        c_cur,
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                )
            else:
                self.noise_convs.append(nn.Conv1d(1, c_cur, kernel_size=1))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock1(ch, k, d))

        self.activation_post = post_activation()
        self.conv_post = torch.nn.utils.parametrizations.weight_norm(
            nn.Conv1d(
                ch,
                1,
                post_conv_kernel_size,
                1,
                padding=get_padding(post_conv_kernel_size),
            )
        )

    def forward(self, x, template=None):
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.silu(x, inplace=True)
            x = self.ups[i](x)

            if self.use_template:
                x = x + self.noise_convs[i](template)

            xs = None

            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)

            x = xs / self.num_kernels

        x = self.activation_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.conv_post)


class ADaMoSHiFiGANV1(nn.Module):
    def __init__(
        self,
        input_channels: int = 128,
        depths: List[int] = [3, 3, 9, 3],
        dims: List[int] = [128, 256, 384, 512],
        drop_path_rate: float = 0.0,
        kernel_sizes: Tuple[int] = (7,),
        upsample_rates: Tuple[int] = (4, 4, 2, 2, 2, 2, 2),
        upsample_kernel_sizes: Tuple[int] = (8, 8, 4, 4, 4, 4, 4),
        resblock_kernel_sizes: Tuple[int] = (3, 7, 11, 13),
        resblock_dilation_sizes: Tuple[Tuple[int]] = (
            (1, 3, 5), (1, 3, 5), (1, 3, 5), (1, 3, 5)),
        num_mels: int = 512,
        upsample_initial_channel: int = 1024,
        use_template: bool = False,
        pre_conv_kernel_size: int = 13,
        post_conv_kernel_size: int = 13,
        sampling_rate: int = 44100,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: int = 512,
        f_min: int = 40,
        f_max: int = 16000,
        n_mels: int = 128,
    ):
        super().__init__()

        self.backbone = ConvNeXtEncoder(
            input_channels=input_channels,
            depths=depths,
            dims=dims,
            drop_path_rate=drop_path_rate,
            kernel_sizes=kernel_sizes,
        )

        self.head = HiFiGANGenerator(
            hop_length=hop_length,
            upsample_rates=upsample_rates,
            upsample_kernel_sizes=upsample_kernel_sizes,
            resblock_kernel_sizes=resblock_kernel_sizes,
            resblock_dilation_sizes=resblock_dilation_sizes,
            num_mels=num_mels,
            upsample_initial_channel=upsample_initial_channel,
            use_template=use_template,
            pre_conv_kernel_size=pre_conv_kernel_size,
            post_conv_kernel_size=post_conv_kernel_size,
        )
        self.sampling_rate = sampling_rate
        self.mel_transform = LogMelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
        )
        self.eval()

    @torch.no_grad()
    def decode(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

    @torch.no_grad()
    def encode(self, x):
        return self.mel_transform(x)

    def forward(self, mel):
        y = self.backbone(mel)
        y = self.head(y)
        return y

# --- from music_dcae_pipeline.py ---
# Original from: https://github.com/ace-step/ACE-Step/blob/main/music_dcae/music_dcae_pipeline.py
import logging
try:
    import torchaudio
except Exception:
    logging.warning("torchaudio missing, ACE model will be broken")

    class _DummyTorchaudio:
        class functional:
            @staticmethod
            def resample(x, orig, new):
                return x

    if torchaudio is None:  # type: ignore
        torchaudio = _DummyTorchaudio()  # type: ignore




class MusicDCAE(torch.nn.Module):
    def __init__(self, source_sample_rate=None, dcae_config={}, vocoder_config={}):
        super(MusicDCAE, self).__init__()

        self.dcae = AutoencoderDC(**dcae_config)
        self.vocoder = ADaMoSHiFiGANV1(**vocoder_config)

        if source_sample_rate is None:
            self.source_sample_rate = 48000
        else:
            self.source_sample_rate = source_sample_rate

        self.transform = transforms.Compose([
            transforms.Normalize(0.5, 0.5),
        ])
        self.min_mel_value = -11.0
        self.max_mel_value = 3.0
        self.audio_chunk_size = int(round((1024 * 512 / 44100 * 48000)))
        self.mel_chunk_size = 1024
        self.time_dimention_multiple = 8
        self.latent_chunk_size = self.mel_chunk_size // self.time_dimention_multiple
        self.scale_factor = 0.1786
        self.shift_factor = -1.9091

    def forward_mel(self, audios):
        mels = []
        for i in range(len(audios)):
            image = self.vocoder.mel_transform(audios[i])
            mels.append(image)
        mels = torch.stack(mels)
        return mels

    @torch.no_grad()
    def encode(self, audios, audio_lengths=None, sr=None):
        if audio_lengths is None:
            audio_lengths = torch.tensor([audios.shape[2]] * audios.shape[0])
            audio_lengths = audio_lengths.to(audios.device)

        if sr is None:
            sr = self.source_sample_rate

        if sr != 44100:
            audios = torchaudio.functional.resample(audios, sr, 44100)

        max_audio_len = audios.shape[-1]
        if max_audio_len % (8 * 512) != 0:
            audios = torch.nn.functional.pad(audios, (0, 8 * 512 - max_audio_len % (8 * 512)))

        mels = self.forward_mel(audios)
        mels = (mels - self.min_mel_value) / (self.max_mel_value - self.min_mel_value)
        mels = self.transform(mels)
        latents = []
        for mel in mels:
            latent = self.dcae.encoder(mel.unsqueeze(0))
            latents.append(latent)
        latents = torch.cat(latents, dim=0)
        latents = (latents - self.shift_factor) * self.scale_factor
        return latents

    @torch.no_grad()
    def decode(self, latents, audio_lengths=None, sr=None):
        latents = latents / self.scale_factor + self.shift_factor

        pred_wavs = []

        for latent in latents:
            mels = self.dcae.decoder(latent.unsqueeze(0))
            mels = mels * 0.5 + 0.5
            mels = mels * (self.max_mel_value - self.min_mel_value) + self.min_mel_value
            wav = self.vocoder.decode(mels[0]).squeeze(1)

            if sr is not None:
                wav = torchaudio.functional.resample(wav, 44100, sr)
            else:
                sr = 44100
            pred_wavs.append(wav)

        if audio_lengths is not None:
            pred_wavs = [wav[:, :length].cpu() for wav, length in zip(pred_wavs, audio_lengths)]
        return torch.stack(pred_wavs)

    def forward(self, audios, audio_lengths=None, sr=None):
        latents, latent_lengths = self.encode(audios=audios, audio_lengths=audio_lengths, sr=sr)
        sr, pred_wavs = self.decode(latents=latents, audio_lengths=audio_lengths, sr=sr)
        return sr, pred_wavs, latents, latent_lengths


# --- Vendored AudioOobleckVAE for ACE 1.5 (from comfy/ldm/audio/autoencoder.py) ---
# code adapted from: https://github.com/Stability-AI/stable-audio-tools

import torch
from torch import nn
from typing import Literal
import math


def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

class VAEBottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.is_discrete = False

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x


def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * pow(torch.sin(x * alpha), 2)

# Adapted from https://github.com/NVIDIA/BigVGAN/blob/main/activations.py under MIT license
class SnakeBeta(nn.Module):

    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        # self.alpha.requires_grad = alpha_trainable
        # self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1).to(x.device) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1).to(x.device)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x

def WNConv1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.Conv1d(*args, **kwargs))

def WNConvTranspose1d(*args, **kwargs):
    return torch.nn.utils.parametrizations.weight_norm(nn.ConvTranspose1d(*args, **kwargs))

def get_activation(activation, antialias=False, channels=None) -> nn.Module:
    # handle both DCAE (relu/silu) and Oobleck (elu/snake) activations
    if activation == "elu":
        act = torch.nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = torch.nn.Identity()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "relu6":
        return nn.ReLU6()
    elif activation == "leaky_relu":
        return nn.LeakyReLU(0.2)
    else:
        raise ValueError(f"Unknown activation {activation}")

    if antialias:
        act = act

    return act


class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()

        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        res = x

        #x = checkpoint(self.layers, x)
        x = self.layers(x)

        return x + res

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False, use_nearest_upsample=False):
        super().__init__()

        if use_nearest_upsample:
            upsample_layer = nn.Sequential(
                nn.Upsample(scale_factor=stride, mode="nearest"),
                WNConv1d(in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=2*stride,
                        stride=1,
                        bias=False,
                        padding='same')
            )
        else:
            upsample_layer = WNConvTranspose1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2))

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            upsample_layer,
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels,
                         dilation=9, use_snake=use_snake),
        )

    def forward(self, x):
        return self.layers(x)

class OobleckEncoder(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False
        ):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]

        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class OobleckDecoder(nn.Module):
    def __init__(self,
                 out_channels=2,
                 channels=128,
                 latent_dim=32,
                 c_mults = [1, 2, 4, 8],
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=True):
        super().__init__()

        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=latent_dim, out_channels=c_mults[-1]*channels, kernel_size=7, padding=3),
        ]

        for i in range(self.depth-1, 0, -1):
            layers += [DecoderBlock(
                in_channels=c_mults[i]*channels,
                out_channels=c_mults[i-1]*channels,
                stride=strides[i-1],
                use_snake=use_snake,
                antialias_activation=antialias_activation,
                use_nearest_upsample=use_nearest_upsample
                )
            ]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[0] * channels),
            WNConv1d(in_channels=c_mults[0] * channels, out_channels=out_channels, kernel_size=7, padding=3, bias=False),
            nn.Tanh() if final_tanh else nn.Identity()
        ]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AudioOobleckVAE(nn.Module):
    def __init__(self,
                 in_channels=2,
                 channels=128,
                 latent_dim=64,
                 c_mults = [1, 2, 4, 8, 16],
                 strides = [2, 4, 4, 8, 8],
                 use_snake=True,
                 antialias_activation=False,
                 use_nearest_upsample=False,
                 final_tanh=False):
        super().__init__()
        self.encoder = OobleckEncoder(in_channels, channels, latent_dim * 2, c_mults, strides, use_snake, antialias_activation)
        self.decoder = OobleckDecoder(in_channels, channels, latent_dim, c_mults, strides, use_snake, antialias_activation,
                                      use_nearest_upsample=use_nearest_upsample, final_tanh=final_tanh)
        self.bottleneck = VAEBottleneck()

    def encode(self, x):
        return self.bottleneck.encode(self.encoder(x))

    def decode(self, x):
        return self.decoder(self.bottleneck.decode(x))

