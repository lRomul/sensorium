from src.utils import get_lr
from src.kinetics import constants

image_size = (64, 64)
batch_size = 32
base_lr = 3e-4
frame_stack_size = 16
config = dict(
    task="kinetics",
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    ema_decay=0.999,
    train_epoch_size=64000,
    num_epochs=[3, 18],
    stages=["warmup", "train"],
    init_weights=True,
    argus_params={
        "nn_module": ("dwiseclassifier", {
            "num_classes": constants.num_classes,
            "in_channels": 5,
            "features": (64, 64, 64, 64,
                         128, 128, 128,
                         256, 256),
            "spatial_strides": (2, 1, 1, 1,
                                2, 1, 1,
                                2, 1),
            "spatial_kernel": 3,
            "temporal_kernel": 5,
            "expansion_ratio": 6,
            "se_reduce_ratio": 32,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
        }),
        "loss": "CrossEntropyLoss",
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
            "weight_decay": 0.05,
        }),
        "device": "cuda:0",
        "frame_stack": {
            "size": frame_stack_size,
            "step": 2,
            "position": "last",
        },
        "frame_size": 64,
        "amp": True,
        "iter_size": 2,
    },
)
