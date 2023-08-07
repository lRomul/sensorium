from src.utils import get_lr
from src import constants


image_size = (64, 64)
batch_size = 16
base_lr = 3e-4
frame_stack_size = 16
config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    ema_decay=0.999,
    train_epoch_size=18000,
    num_epochs=[3, 18],
    stages=["warmup", "train"],
    num_dataloader_workers=8,
    init_weights=True,
    argus_params={
        "nn_module": ("dwiseneuro", {
            "readout_outputs": constants.num_neurons,
            "in_channels": 5,
            "stem_features": 64,
            "block_features": (128, 256, 512, 1024),
            "block_strides": (2, 2, 2, 2),
            "expansion_ratio": 3,
            "se_reduce_ratio": 32,
            "readout_features": 1024 * 16,
            "readout_groups": 4,
            "dropout": 0.2,
        }),
        "loss": ("mice_poisson", {
            "log_input": False,
            "full": False,
            "eps": 1e-8,
        }),
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
        "inputs_processor": ("stack_inputs", {
            "size": image_size,
            "pad_fill_value": 0.,
        }),
        "responses_processor": ("identity", {}),
        "amp": False,
        "iter_size": 1,
    },
    mixup={
        "alpha": 0.4,
        "prob": 0.5,
    },
)
