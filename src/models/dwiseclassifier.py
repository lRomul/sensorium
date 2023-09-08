import functools

from torch import nn

from src.models.dwiseneuro import DepthwiseCore


class DwiseClassifier(nn.Module):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 1,
                 features: tuple[int, ...] = (64, 128, 256, 512),
                 spatial_strides: tuple[int, ...] = (2, 2, 2, 2),
                 spatial_kernel: int = 3,
                 temporal_kernel: int = 3,
                 expansion_ratio: int = 3,
                 se_reduce_ratio: int = 16,
                 drop_rate: float = 0.,
                 drop_path_rate: float = 0.):
        super().__init__()
        self.drop_rate = drop_rate
        act_layer = functools.partial(nn.SiLU, inplace=True)

        self.core = DepthwiseCore(
            in_channels=in_channels,
            features=features,
            spatial_strides=spatial_strides,
            spatial_kernel=spatial_kernel,
            temporal_kernel=temporal_kernel,
            expansion_ratio=expansion_ratio,
            se_reduce_ratio=se_reduce_ratio,
            act_layer=act_layer,
            drop_path_rate=drop_path_rate,
        )

        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Linear(features[-1], num_classes, bias=True)

    def forward(self, x):
        x = self.core(x)
        x = self.pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
        if self.drop_rate > 0.:
            x = nn.functional.dropout(x, p=self.drop_rate, training=self.training)
        x = self.classifier(x)
        return x
