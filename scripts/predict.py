import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd

from src.utils import get_best_model_path
from src.predictors import Predictor
from src.data import get_mouse_data
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiment", required=True, type=str)
    parser.add_argument("-s", "--split", required=True, type=str)
    parser.add_argument("-m", "--mice", default="all", type=str)
    parser.add_argument("-d", "--device", default="cuda:0", type=str)
    return parser.parse_args()


def predict_trial(trial_data: dict, predictor: Predictor):
    length = trial_data["length"]
    video = np.load(trial_data["video_path"])[..., :length]
    responses = predictor.predict_trial(video=video)
    return responses


def predict_mouse(experiment: str, split: str, mouse_index: int, device: str):
    mouse = constants.index2mouse[mouse_index]
    print(f"Predict: {experiment=}, {split=}, {mouse_index=}, {mouse=}, {device=}")
    mouse_data = get_mouse_data(mouse=mouse, split=split)
    model_path = get_best_model_path(constants.experiments_dir / experiment / f"mouse_{mouse_index}")
    print("Model path:", model_path)
    predictor = Predictor(model_path=model_path, device=device)
    mouse_prediction_dir = constants.predictions_dir / experiment / split / f"mouse_{mouse_index}"
    mouse_prediction_dir.mkdir(parents=True, exist_ok=True)

    for trial_data in tqdm(mouse_data["trials"]):
        responses = predict_trial(trial_data, predictor)
        np.save(str(mouse_prediction_dir / f"{trial_data['trial_id']}.npy"), responses)


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
            prediction = prediction[..., constants.submission_skip_first:]
            if constants.submission_skip_last:
                prediction = prediction[..., :-constants.submission_skip_last]
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

    if args.mice == "all":
        mice_indexes = constants.mice_indexes
    else:
        mice_indexes = [int(index) for index in args.mice.split(",")]

    for mouse_index in mice_indexes:
        predict_mouse(args.experiment, args.split, mouse_index, args.device)

    make_submission(args.experiment, args.split)
