import json
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils import get_best_model_path
from src.predictors import Predictor
from src.data import get_mouse_data
from src.metrics import corr
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-s", "--split", required=True, type=str)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
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


def predict_mouse(experiment: str, split: str, predictor: Predictor, mouse_index: int):
    mouse = constants.index2mouse[mouse_index]
    print(f"Predict mouse: {mouse_index=}, {mouse=}")
    mouse_data = get_mouse_data(mouse=mouse, split=split)
    mouse_prediction_dir = constants.predictions_dir / experiment / split / f"mouse_{mouse_index}"
    mouse_prediction_dir.mkdir(parents=True, exist_ok=True)

    for trial_data in tqdm(mouse_data["trials"]):
        responses = predict_trial(trial_data, predictor, mouse_index)
        np.save(str(mouse_prediction_dir / f"{trial_data['trial_id']}.npy"), responses)


def predict_mice(experiment: str, split: str, device: str):
    model_path = get_best_model_path(constants.experiments_dir / experiment)
    print(f"Predict mice: {experiment=}, {split=}, {model_path=}")
    predictor = Predictor(model_path=model_path, device=device, blend_weights="ones")

    for mouse_index in constants.mice_indexes:
        predict_mouse(experiment, split, predictor, mouse_index)


def cut_responses(prediction: np.ndarray):
    prediction = prediction[..., constants.submission_skip_first:]
    if constants.submission_skip_last:
        prediction = prediction[..., :-constants.submission_skip_last]
    return prediction


def evaluate_predictions(experiment: str, split: str):
    prediction_dir = constants.predictions_dir / experiment / split
    correlations = dict()
    for mouse_index in constants.mice_indexes:
        mouse = constants.index2mouse[mouse_index]
        mouse_data = get_mouse_data(mouse=mouse, split=split)
        mouse_prediction_dir = prediction_dir / f"mouse_{mouse_index}"
        predictions = []
        targets = []
        for trial_data in mouse_data["trials"]:
            trial_id = trial_data['trial_id']
            prediction = np.load(str(mouse_prediction_dir / f"{trial_id}.npy"))
            target = np.load(trial_data["response_path"])[..., :trial_data["length"]]
            prediction, target = cut_responses(prediction), cut_responses(target)
            predictions.append(prediction)
            targets.append(target)
        correlation = corr(
            np.concatenate(predictions, axis=1),
            np.concatenate(targets, axis=1),
            axis=1
        ).mean()
        print(f"Mouse {mouse_index} {mouse} correlation: {correlation}")
        correlations[mouse] = correlation
    mean_correlation = np.mean(list(correlations.values()))
    print("Mean correlation:", mean_correlation)

    evaluate_result = {"correlations": correlations, "mean_correlation": mean_correlation}
    with open(prediction_dir / "evaluate.json", "w") as outfile:
        json.dump(evaluate_result, outfile, indent=4)


def make_submission(experiment: str, split: str):
    prediction_dir = constants.predictions_dir / experiment / split
    data = []
    for mouse_index in constants.mice_indexes:
        mouse = constants.index2mouse[mouse_index]
        mouse_data = get_mouse_data(mouse=mouse, split=split)
        neuron_ids = mouse_data["neuron_ids"].tolist()
        mouse_prediction_dir = prediction_dir / f"mouse_{mouse_index}"
        for trial_data in mouse_data["trials"]:
            trial_id = trial_data['trial_id']
            prediction = np.load(str(mouse_prediction_dir / f"{trial_id}.npy"))
            prediction = cut_responses(prediction)
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
    predict_mice(args.experiment, args.split, args.device)

    if args.split in constants.labeled_splits:
        evaluate_predictions(args.experiment, args.split)
    else:
        make_submission(args.experiment, args.split)
