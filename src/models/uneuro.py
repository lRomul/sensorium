from torch import nn

from timm.layers import (
    DropPath,
    get_act_layer,
)


class BatchNormAct3d(nn.Module):
    def __init__(self,
                 num_features: int,
                 act_layer=nn.ReLU,
                 apply_act: bool = True):
        super().__init__()
        self.bn3d = nn.BatchNorm3d(num_features)
        if apply_act:
            self.act = act_layer(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x):
        x = self.bn3d(x)
        x = self.act(x)
        return x


class ConvNormAct3d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int | tuple[int, int, int],
                 stride: int | tuple[int, int, int] = 1,
                 padding: int | tuple[int, int, int] = 1,
                 dilation: int | tuple[int, int, int] = 1,
                 groups: int = 1,
                 bias: bool = False,
                 apply_act: bool = True,
                 act_layer=nn.ReLU):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias)
        self.bn = BatchNormAct3d(out_channels, act_layer=act_layer, apply_act=apply_act)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class SqueezeExcite(nn.Module):
    def __init__(self,
                 in_features: int,
                 reduce_ratio: int = 16,
                 act_layer=nn.ReLU,
                 gate_layer=nn.Sigmoid):
        super().__init__()
        rd_channels = in_features // reduce_ratio
        self.conv_reduce = nn.Conv3d(in_features, rd_channels, (1, 1, 1), bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv3d(rd_channels, in_features, (1, 1, 1), bias=True)
        self.gate = gate_layer()

    def forward(self, x):
        x_se = x.mean((2, 3, 4), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class InvertedResidual3d(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 stride: int = 1,
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 act_layer=nn.ReLU,
                 drop_path_rate: float = 0.,
                 bias: bool = False,
                 no_skip: bool = False):
        super().__init__()
        self.has_skip = not no_skip and (in_features == out_features and stride == 1)

        mid_features = in_features * expansion_ratio

        # Point-wise expansion
        self.conv_pw = nn.Conv3d(in_features, mid_features, (1, 1, 1), bias=bias)
        self.bn1 = BatchNormAct3d(mid_features, act_layer=act_layer)

        # Depth-wise convolution
        self.conv_dw = nn.Conv3d(mid_features, mid_features,
                                 kernel_size=(3, 3, 3), stride=(1, stride, stride),
                                 dilation=(1, 1, 1), padding=(1, 1, 1),
                                 groups=mid_features, bias=bias)
        self.bn2 = BatchNormAct3d(mid_features, act_layer=act_layer)

        # Squeeze-and-excitation
        self.se = SqueezeExcite(mid_features, act_layer=act_layer, reduce_ratio=se_reduce_ratio)

        # Point-wise linear projection
        self.conv_pwl = nn.Conv3d(mid_features, out_features, (1, 1, 1), bias=bias)
        self.bn3 = BatchNormAct3d(out_features, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class UNeuro(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 1,
                 num_stem_features: int = 32,
                 num_block_features: tuple[int, ...] = (64, 128, 256, 512),
                 block_strides: tuple[int, ...] = (2, 2, 2, 2),
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 drop_rate: bool = 0.,
                 drop_path_rate: float = 0.,
                 act_layer: str = "silu"):
        super().__init__()
        self.drop_rate = drop_rate

        act_layer = get_act_layer(act_layer)

        self.conv1 = nn.Conv3d(in_channels, num_stem_features,
                               kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               dilation=(1, 1, 1), padding=(1, 1, 1))
        self.bn1 = BatchNormAct3d(num_stem_features, act_layer=act_layer)

        blocks = []
        prev_num_features = num_stem_features
        for num_features, stride in zip(num_block_features, block_strides):
            blocks += [
                InvertedResidual3d(
                    prev_num_features,
                    num_features,
                    stride=stride,
                    expansion_ratio=expansion_ratio,
                    se_reduce_ratio=se_reduce_ratio,
                    act_layer=act_layer,
                    drop_path_rate=drop_path_rate,
                )
            ]
            prev_num_features = num_features
        self.blocks = nn.Sequential(*blocks)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(prev_num_features, num_classes, bias=True)
        self.activation = nn.Softplus(beta=1, threshold=20)

    def forward_3d(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        b, _, t, h, w = x.shape
        x = self.blocks(x)
        x = x[:, :, -1]
        return x

    def forward_head(self, x):
        x = self.global_pool(x)
        x = x.view(x.shape[0], -1)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        x = self.activation(x)
        return x

    def forward(self, x):
        x = self.forward_3d(x)
        x = self.forward_head(x)
        return x
