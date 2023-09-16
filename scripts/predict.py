import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np

from src.submission import evaluate_folds_predictions, make_submission
from src.utils import get_best_model_path
from src.predictors import Predictor
from src.data import get_mouse_data
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-s", "--split", required=True,
                        choices=["folds"] + constants.unlabeled_splits, type=str)
    parser.add_argument("-d", "--dataset", default="new", choices=["new", "old"], type=str)
    parser.add_argument("--device", default="cuda:0", type=str)
    return parser.parse_args()


def predict_trial(trial_data: dict, predictor: Predictor, mouse_index: int):
    length = trial_data["length"]
    video = np.load(trial_data["video_path"])[..., :length]
    behavior = np.load(trial_data["behavior_path"])[..., :length]
    pupil_center = np.load(trial_data["pupil_center_path"])[..., :length]
    responses = predictor.predict_trial(
        video=video,
        behavior=behavior,
        pupil_center=pupil_center,
        mouse_index=mouse_index,
    )
    return responses


def predict_mouse_split(mouse: str, split: str,
                        predictors: list[Predictor], save_dir: Path):
    mouse_index = constants.mouse2index[mouse]
    print(f"Predict mouse split: {mouse=} {split=} {len(predictors)=} {str(save_dir)=}")
    mouse_data = get_mouse_data(mouse=mouse, splits=[split])

    for trial_data in tqdm(mouse_data["trials"]):
        responses_lst = []
        for predictor in predictors:
            responses = predict_trial(trial_data, predictor, mouse_index)
            responses_lst.append(responses)
        blend_responses = np.mean(responses_lst, axis=0)
        np.save(str(save_dir / f"{trial_data['trial_id']}.npy"), blend_responses)


def predict_folds(experiment: str, dataset: str, device: str):
    print(f"Predict folds: {experiment=}, {dataset=}, {device=}")
    for mouse in constants.dataset2mice[dataset]:
        mouse_prediction_dir = constants.predictions_dir / experiment / "out-of-fold" / mouse
        mouse_prediction_dir.mkdir(parents=True, exist_ok=True)
        for fold_split in constants.folds_splits:
            model_path = get_best_model_path(constants.experiments_dir / experiment / fold_split)
            print("Model path:", str(model_path))
            predictor = Predictor(model_path=model_path, device=device, blend_weights="ones")
            predict_mouse_split(mouse, fold_split, [predictor], mouse_prediction_dir)


def predict_unlabeled_split(experiment: str, split: str, dataset: str, device: str):
    print(f"Predict unlabeled split: {experiment=}, {split=}, {dataset=}, {device=}")
    predictors = []
    for fold_split in constants.folds_splits:
        model_path = get_best_model_path(constants.experiments_dir / experiment / fold_split)
        print("Model path:", str(model_path))
        predictor = Predictor(model_path=model_path, device=device, blend_weights="ones")
        predictors.append(predictor)
    for mouse in constants.dataset2mice[dataset]:
        mouse_prediction_dir = constants.predictions_dir / experiment / split / mouse
        mouse_prediction_dir.mkdir(parents=True, exist_ok=True)
        predict_mouse_split(mouse, split, predictors, mouse_prediction_dir)


if __name__ == "__main__":
    args = parse_arguments()

    if args.split == "folds":
        predict_folds(args.experiment, args.dataset, args.device)
        evaluate_folds_predictions(args.experiment, args.dataset)
    elif args.dataset == "new":
        predict_unlabeled_split(args.experiment, args.split, args.dataset, args.device)
        make_submission(args.experiment, args.split)
