import copy
import json
import argparse
from pathlib import Path
from pprint import pprint
from importlib.machinery import SourceFileLoader

from torch.utils.data import DataLoader

from argus.callbacks import (
    LoggingToFile,
    LoggingToCSV,
    CosineAnnealingLR,
    Checkpoint,
    LambdaLR,
)

from src.datasets import TrainMouseVideoDataset, ValMouseVideoDataset, ConcatMiceVideoDataset
from src.responses import get_responses_processor
from src.indexes import StackIndexesGenerator
from src.ema import ModelEma, EmaCheckpoint
from src.inputs import get_inputs_processor
from src.metrics import CorrelationMetric
from src.argus_models import MouseModel
from src.utils import get_lr
from src.mixup import Mixup
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    return parser.parse_args()


def train_mouse(config: dict, save_dir: Path):
    config = copy.deepcopy(config)
    argus_params = config["argus_params"]

    model = MouseModel(argus_params)

    nn_module_params = model.params["nn_module"][1]
    if "pretrained" in model.params["nn_module"][1]:
        nn_module_params["pretrained"] = False

    if config["ema_decay"]:
        print("EMA decay:", config["ema_decay"])
        model.model_ema = ModelEma(model.nn_module, decay=config["ema_decay"])
        checkpoint_class = EmaCheckpoint
    else:
        checkpoint_class = Checkpoint

    indexes_generator = StackIndexesGenerator(**argus_params["frame_stack"])
    inputs_processor = get_inputs_processor(*argus_params["inputs_processor"])
    responses_processor = get_responses_processor(*argus_params["responses_processor"])

    mixup = Mixup(**config["mixup"]) if config["mixup"]["prob"] else None
    train_datasets = []
    mouse_epoch_size = config["train_epoch_size"] // constants.num_mice
    for mouse in constants.mice:
        train_datasets += [
            TrainMouseVideoDataset(
                mouse=mouse,
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
                epoch_size=mouse_epoch_size,
                mixup=mixup,
            )
        ]
    train_dataset = ConcatMiceVideoDataset(train_datasets)
    print("Train dataset len:", len(train_dataset))
    val_datasets = []
    for mouse in constants.mice:
        val_datasets += [
            ValMouseVideoDataset(
                mouse=mouse,
                indexes_generator=indexes_generator,
                inputs_processor=inputs_processor,
                responses_processor=responses_processor,
            )
        ]
    val_dataset = ConcatMiceVideoDataset(val_datasets)
    print("Val dataset len:", len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_dataloader_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(config["batch_size"] * 2),
        num_workers=config["num_dataloader_workers"],
        shuffle=False,
    )

    for num_epochs, stage in zip(config["num_epochs"], config["stages"]):
        callbacks = [
            LoggingToFile(save_dir / "log.txt", append=True),
            LoggingToCSV(save_dir / "log.csv", append=True),
        ]

        num_iterations = (len(train_dataset) // config["batch_size"]) * num_epochs
        if stage == "warmup":
            callbacks += [
                LambdaLR(lambda x: x / num_iterations,
                         step_on_iteration=True),
            ]
        elif stage == "train":
            checkpoint_format = "model-{epoch:03d}-{val_corr:.6f}.pth"
            callbacks += [
                checkpoint_class(save_dir, file_format=checkpoint_format, max_saves=1),
                CosineAnnealingLR(
                    T_max=num_iterations,
                    eta_min=get_lr(config["min_base_lr"], config["batch_size"]),
                    step_on_iteration=True,
                ),
            ]

        metrics = [
            CorrelationMetric(),
        ]

        model.fit(train_loader,
                  val_loader=val_loader,
                  num_epochs=num_epochs,
                  callbacks=callbacks,
                  metrics=metrics)


if __name__ == "__main__":
    args = parse_arguments()
    print("Experiment:", args.experiment)

    config_path = constants.configs_dir / f"{args.experiment}.py"
    if not config_path.exists():
        raise RuntimeError(f"Config '{config_path}' is not exists")

    config = SourceFileLoader(args.experiment, str(config_path)).load_module().config
    print("Experiment config:")
    pprint(config, sort_dicts=False)

    experiments_dir = constants.experiments_dir / args.experiment
    print("Experiment dir:", experiments_dir)
    if not experiments_dir.exists():
        experiments_dir.mkdir(parents=True, exist_ok=True)
    else:
        print(f"Folder '{experiments_dir}' already exists.")

    with open(experiments_dir / "train.py", "w") as outfile:
        outfile.write(open(__file__).read())

    with open(experiments_dir / "config.json", "w") as outfile:
        json.dump(config, outfile, indent=4)

    train_mouse(config, experiments_dir)
