import argparse

from tqdm import tqdm
import numpy as np

from src.submission import evaluate_folds_predictions, make_submission
from src.data import get_mouse_data
from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--experiments", required=True, type=str)
    parser.add_argument("-s", "--split", required=True,
                        choices=["folds"] + constants.unlabeled_splits, type=str)
    parser.add_argument("-d", "--dataset", default="new", choices=["new", "old"], type=str)
    return parser.parse_args()


def ensemble_experiments(experiments: list[str], split: str, dataset: str):
    assert len(experiments) > 1
    print(f"Ensemble experiments: {experiments=}, {split=}, {dataset=}")
    split_dir_name = "out-of-fold" if split == "folds" else split
    splits = constants.folds_splits if split == "folds" else [split]
    ensemble_dir = constants.predictions_dir / ",".join(experiments) / split_dir_name
    for mouse in constants.dataset2mice[dataset]:
        ensemble_mouse_dir = ensemble_dir / mouse
        print(f"Ensemble mouse: {mouse=}, {str(ensemble_mouse_dir)=}")
        ensemble_mouse_dir.mkdir(parents=True, exist_ok=True)
        mouse_data = get_mouse_data(mouse=mouse, splits=splits)

        for trial_data in tqdm(mouse_data["trials"]):
            pred_filename = f"{trial_data['trial_id']}.npy"
            responses_lst = []
            for experiment in experiments:
                responses = np.load(
                    str(constants.predictions_dir / experiment / split_dir_name / mouse / pred_filename)
                )
                responses_lst.append(responses)
            blend_responses = np.mean(responses_lst, axis=0)
            np.save(str(ensemble_mouse_dir / pred_filename), blend_responses)


if __name__ == "__main__":
    args = parse_arguments()
    experiments_lst = sorted(args.experiments.split(','))
    experiment_name = ",".join(experiments_lst)
    ensemble_experiments(experiments_lst, args.split, args.dataset)
    if args.split == "folds":
        evaluate_folds_predictions(experiment_name, args.dataset)
    elif args.dataset == "new":
        make_submission(experiment_name, args.split)
