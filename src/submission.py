import json

import numpy as np
import pandas as pd

from src.responses import ResponseNormalizer
from src.data import get_mouse_data
from src.metrics import corr
from src import constants


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
        correlation = float(corr(
            np.concatenate(predictions, axis=1),
            np.concatenate(targets, axis=1),
            axis=1
        ).mean())
        print(f"Mouse {mouse} correlation: {correlation}")
        correlations[mouse] = correlation
    mean_correlation = float(np.mean(list(correlations.values())))
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
    del data
    split = split.replace('_test_', '_').replace('bonus', 'test_bonus_ood')
    submission_path = prediction_dir / f"predictions_{split}.parquet.brotli"
    submission_df.to_parquet(submission_path, compression='brotli', engine='pyarrow', index=False)
    print(f"Submission saved to '{submission_path}'")
