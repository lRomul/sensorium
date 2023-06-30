from src.utils import get_lr


image_size = (64, 64)
batch_size = 8
base_lr = 3e-4
frame_stack_size = 15
config = dict(
    image_size=image_size,
    batch_size=batch_size,
    base_lr=base_lr,
    min_base_lr=base_lr * 0.01,
    train_epoch_size=6000,
    num_epochs=[4, 20],
    stages=["warmup", "train"],
    num_dataloader_workers=8,
    argus_params={
        "nn_module": ("timm", {
            "model_name": "tf_efficientnetv2_b0",
            "num_classes": None,
            "in_chans": frame_stack_size,
            "drop_rate": 0.2,
            "drop_path_rate": 0.2,
            "pretrained": True,
            "use_stem_stride": False,
        }),
        "loss": "MSELoss",
        "optimizer": ("AdamW", {
            "lr": get_lr(base_lr, batch_size),
            "weight_decay": 0.01,
        }),
        "prediction_transform": "identity",
        "device": "cuda:0",
        "frame_stack": {
            "size": frame_stack_size,
            "step": 2
        },
        "frames_processor": ("normalize_pad", {
            "size": image_size,
            "pad_mode": "constant",
            "fill_value": 0,
        }),
        "responses_processor": ("indexing", {
            "index": -1,
        })
    },
)
