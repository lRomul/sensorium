import json
import argparse
from pathlib import Path

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.responses import ResponseNormalizer
from src.utils import get_best_model_path
from src.predictors import Predictor
from src.data import get_mouse_data
from src.metrics import corr
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-s", "--split", required=True,
                        choices=["folds"] + constants.unlabeled_splits, type=str)
    parser.add_argument("-d", "--dataset", default="new", type=str)
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


def cut_responses_for_submission(prediction: np.ndarray):
    prediction = prediction[..., constants.submission_skip_first:]
    if constants.submission_skip_last:
        prediction = prediction[..., :-constants.submission_skip_last]
    return prediction


def evaluate_folds_predictions(experiment: str, dataset: str):
    prediction_dir = constants.predictions_dir / experiment / "out-of-fold"
    correlations = dict()
    for mouse in constants.dataset2mice[dataset]:
        mouse_data = get_mouse_data(mouse=mouse, splits=constants.folds_splits)
        mouse_prediction_dir = prediction_dir / mouse
        predictions = []
        targets = []
        for trial_data in mouse_data["trials"]:
            trial_id = trial_data['trial_id']
            prediction = np.load(str(mouse_prediction_dir / f"{trial_id}.npy"))
            target = np.load(trial_data["response_path"])[..., :trial_data["length"]]
            prediction = cut_responses_for_submission(prediction)
            target = cut_responses_for_submission(target)
            predictions.append(prediction)
            targets.append(target)
        correlation = corr(
            np.concatenate(predictions, axis=1),
            np.concatenate(targets, axis=1),
            axis=1
        ).mean()
        print(f"Mouse {mouse} correlation: {correlation}")
        correlations[mouse] = correlation
    mean_correlation = np.mean(list(correlations.values()))
    print("Mean correlation:", mean_correlation)

    evaluate_result = {"correlations": correlations, "mean_correlation": mean_correlation}
    with open(prediction_dir / f"evaluate_{dataset}.json", "w") as outfile:
        json.dump(evaluate_result, outfile, indent=4)


def make_submission(experiment: str, split: str):
    prediction_dir = constants.predictions_dir / experiment / split
    data = []
    for mouse in constants.new_mice:
        normalizer = ResponseNormalizer(mouse)
        mouse_data = get_mouse_data(mouse=mouse, splits=[split])
        neuron_ids = mouse_data["neuron_ids"].tolist()
        mouse_prediction_dir = prediction_dir / mouse
        for trial_data in mouse_data["trials"]:
            trial_id = trial_data['trial_id']
            prediction = np.load(str(mouse_prediction_dir / f"{trial_id}.npy"))
            prediction = normalizer(prediction)
            prediction = cut_responses_for_submission(prediction)
            data.append((mouse, trial_id, prediction.tolist(), neuron_ids))
    submission_df = pd.DataFrame.from_records(
        data,
        columns=['mouse', 'trial_indices', 'prediction', 'neuron_ids']
    )
    split = split.replace('_test_', '_').replace('bonus', 'ood')
    submission_path = prediction_dir / f"predictions_{split}.parquet.brotli"
    submission_df.to_parquet(submission_path, compression='brotli', engine='pyarrow', index=False)
    print(f"Submission saved to '{submission_path}'")


if __name__ == "__main__":
    args = parse_arguments()

    if args.split == "folds":
        predict_folds(args.experiment, args.dataset, args.device)
        evaluate_folds_predictions(args.experiment, args.dataset)
    elif args.dataset == "new":
        predict_unlabeled_split(args.experiment, args.split, args.dataset, args.device)
        make_submission(args.experiment, args.split)
