import math
import functools
from typing import Callable

import torch
from torch import nn


class BatchNormAct(nn.Module):
    def __init__(self,
                 num_features: int,
                 bn_layer: Callable = nn.BatchNorm3d,
                 act_layer: Callable = nn.ReLU,
                 apply_act: bool = True):
        super().__init__()
        self.bn = bn_layer(num_features)
        self.act = act_layer() if apply_act else nn.Identity()

    def forward(self, x):
        x = self.bn(x)
        x = self.act(x)
        return x


class SqueezeExcite3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 reduce_ratio: int = 16,
                 act_layer: Callable = nn.ReLU,
                 gate_layer: Callable = nn.Sigmoid):
        super().__init__()
        rd_channels = in_features // reduce_ratio
        self.conv_reduce = nn.Conv3d(in_features, rd_channels, (1, 1, 1), bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv3d(rd_channels, in_features, (1, 1, 1), bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0. and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class InvertedResidual3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 stride: tuple[int, int, int] = (1, 1, 1),
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 act_layer: Callable = nn.ReLU,
                 drop_path_rate: float = 0.,
                 bias: bool = False):
        super().__init__()
        mid_features = in_features * expansion_ratio
        bn_layer = nn.BatchNorm3d

        # Point-wise expansion
        self.conv_pw = nn.Conv3d(in_features, mid_features, (1, 1, 1), bias=bias)
        self.bn1 = BatchNormAct(mid_features, bn_layer=bn_layer, act_layer=act_layer)

        # Depth-wise convolution
        self.conv_dw = nn.Conv3d(mid_features, mid_features, (3, 3, 3), stride=stride,
                                 padding=(1, 1, 1), groups=mid_features, bias=bias)
        self.bn2 = BatchNormAct(mid_features, bn_layer=bn_layer, act_layer=act_layer)

        # Squeeze-and-excitation
        self.se = SqueezeExcite3d(mid_features, act_layer=act_layer, reduce_ratio=se_reduce_ratio)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv3d(mid_features, out_features, (1, 1, 1), bias=bias)
        self.bn3 = BatchNormAct(out_features, bn_layer=bn_layer, apply_act=False)

        # Projection shortcut
        self.drop_path = DropPath(drop_prob=drop_path_rate)
        self.proj_sc = nn.Sequential(
            nn.Conv3d(in_features, out_features, kernel_size=(1, 1, 1), stride=stride,
                      groups=math.gcd(in_features, out_features), bias=bias),
            BatchNormAct(out_features, bn_layer=bn_layer, apply_act=False),
        )

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        x = self.drop_path(x) + self.proj_sc(shortcut)
        return x


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.orig_channels = channels
        channels = math.ceil(channels / 6) * 2
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_encoding", None, persistent=False)

    def get_emb(self, sin_inp):
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=0)
        return torch.flatten(emb, 0, 1)

    def create_cached_encoding(self, tensor):
        _, orig_ch, x, y, z = tensor.shape
        assert orig_ch == self.orig_channels
        self.cached_encoding = None
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", self.inv_freq, pos_x)
        sin_inp_y = torch.einsum("i,j->ij", self.inv_freq, pos_y)
        sin_inp_z = torch.einsum("i,j->ij", self.inv_freq, pos_z)
        emb_x = self.get_emb(sin_inp_x).unsqueeze(-1).unsqueeze(-1)
        emb_y = self.get_emb(sin_inp_y).unsqueeze(1).unsqueeze(-1)
        emb_z = self.get_emb(sin_inp_z).unsqueeze(1).unsqueeze(1)
        emb = torch.zeros((self.channels * 3, x, y, z), dtype=tensor.dtype, device=tensor.device)
        emb[:self.channels] = emb_x
        emb[self.channels: 2 * self.channels] = emb_y
        emb[2 * self.channels:] = emb_z
        emb = emb[None, :self.orig_channels].contiguous()
        self.cached_encoding = emb
        return emb

    def forward(self, x):
        if len(x.shape) != 5:
            raise RuntimeError("The input tensor has to be 5D")

        cached_encoding = self.cached_encoding
        if cached_encoding is None or cached_encoding.shape[1:] != x.shape[1:]:
            cached_encoding = self.create_cached_encoding(x)

        return x + cached_encoding.expand_as(x)


class ChannelShuffle(nn.Module):
    def __init__(self, groups: int):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        if self.groups > 1:
            b, c, t = x.shape
            x = x.view(b, -1, self.groups, t)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(b, -1, t)
        return x


class Readout(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 layers_features: tuple[int, ...],
                 layers_groups: tuple[int, ...],
                 final_groups: int,
                 kernel_size: int = 1,
                 act_layer: Callable = nn.ReLU,
                 drop_rate: float = 0.):
        super().__init__()
        self.out_features = out_features

        prev_num_features = in_features
        layers = []
        for layer_index, (features, groups) in enumerate(zip(layers_features, layers_groups)):
            layer_drop_rate = drop_rate * layer_index / len(layers_features)
            layers += [
                nn.Sequential(
                    nn.Dropout1d(p=layer_drop_rate),
                    nn.Conv1d(prev_num_features, features, (kernel_size,),
                              padding="same", groups=groups, bias=False),
                    BatchNormAct(features, bn_layer=nn.BatchNorm1d, act_layer=act_layer),
                    ChannelShuffle(groups),
                )
            ]
            prev_num_features = features
        self.layers = nn.Sequential(*layers)

        self.final_layer = nn.Sequential(
            nn.Dropout1d(p=drop_rate),
            nn.Conv1d(prev_num_features,
                      math.ceil(out_features / final_groups) * final_groups, (1,),
                      groups=final_groups, bias=True),
        )
        self.gate = nn.Softplus()

    def forward(self, x):
        x = self.layers(x)
        x = self.final_layer(x)
        x = x[:, :self.out_features]
        x = self.gate(x)
        return x


class DwiseNeuro(nn.Module):
    def __init__(self,
                 readout_outputs: tuple[int, ...],
                 in_channels: int = 1,
                 stem_features: int = 32,
                 block_features: tuple[int, ...] = (64, 128, 256, 512),
                 block_strides: tuple[int, ...] = (2, 2, 2, 2),
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 readout_features: tuple[int, ...] = (2048, 4096),
                 readout_groups: tuple[int, ...] = (2, 4),
                 readout_final_groups: int = 8,
                 readout_kernel_size: int = 1,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        act_layer = functools.partial(nn.SiLU, inplace=True)
        self.conv1 = nn.Conv3d(in_channels, stem_features, (1, 3, 3),
                               stride=(1, 2, 2), padding=(0, 1, 1), bias=False)
        self.bn1 = BatchNormAct(stem_features, bn_layer=nn.BatchNorm3d, act_layer=act_layer)

        prev_num_features = stem_features
        blocks = []
        for block_index, (num_features, stride) in enumerate(zip(block_features, block_strides)):
            block_drop_path_rate = drop_path_rate * block_index / len(block_features)
            blocks += [
                PositionalEncoding3D(prev_num_features),
                InvertedResidual3d(
                    prev_num_features,
                    num_features,
                    stride=(1, stride, stride),
                    expansion_ratio=expansion_ratio,
                    se_reduce_ratio=se_reduce_ratio,
                    act_layer=act_layer,
                    drop_path_rate=block_drop_path_rate,
                    bias=False,
                )
            ]
            prev_num_features = num_features
        self.blocks = nn.Sequential(*blocks)

        self.pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        self.readouts = nn.ModuleList()
        for readout_output in readout_outputs:
            self.readouts += [
                Readout(prev_num_features, readout_output, readout_features,
                        readout_groups, readout_final_groups,
                        kernel_size=readout_kernel_size,
                        act_layer=act_layer, drop_rate=drop_rate)
            ]

    def forward(self, x: torch.Tensor, index: int | None = None) -> list[torch.Tensor] | torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)

        if index is None:
            return [readout(x) for readout in self.readouts]
        else:
            return self.readouts[index](x)
