from src.utils import get_lr


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
    train_epoch_size=6000,
    num_epochs=[3, 18],
    stages=["warmup", "train"],
    num_dataloader_workers=8,
    argus_params={
        "nn_module": ("uneuro", {
            "num_classes": None,
            "in_channels": 5,
            "num_stem_features": 64,
            "num_block_features": (128, 256, 512, 1024),
            "block_strides": (2, 2, 2, 2),
            "expansion_ratio": 3,
            "se_reduce_ratio": 32,
            "drop_rate": 0.2,
            "num_readout_features": 2048,
        }),
        "loss": ("PoissonNLLLoss", {
            "log_input": False,
            "full": False,
            "reduction": "mean",
        }),
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
            "weight_decay": 0.01,
        }),
        "prediction_transform": "identity",
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
