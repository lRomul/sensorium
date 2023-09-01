import numpy as np

from src import constants


def get_length_without_nan(array: np.ndarray):
    nan_indexes = np.argwhere(np.isnan(array)).ravel()
    if nan_indexes.shape[0]:
        return nan_indexes[0]
    else:
        return array.shape[0]


def get_folds_tiers(mouse: str, num_folds: int):
    tiers = np.load(str(constants.sensorium_dir / mouse / "meta" / "trials" / "tiers.npy"))
    if mouse in constants.new_mice:
        folds_ids = np.argwhere((tiers == "train") | (tiers == "oracle")).ravel()
    else:
        folds_ids = np.argwhere(tiers != "none").ravel()
    generator = np.random.default_rng(seed=12)
    generator.shuffle(folds_ids)
    for fold, fold_ids in enumerate(np.array_split(folds_ids, num_folds)):
        tiers[fold_ids] = f"fold_{fold}"
    return tiers


def get_mouse_data(mouse: str, splits: list[str]) -> dict:
    assert mouse in constants.mice
    tiers = get_folds_tiers(mouse, constants.num_folds)
    mouse_dir = constants.sensorium_dir / mouse
    neuron_ids = np.load(str(mouse_dir / "meta" / "neurons" / "unit_ids.npy"))
    cell_motor_coords = np.load(str(mouse_dir / "meta" / "neurons" / "cell_motor_coordinates.npy"))

    mouse_data = {
        "mouse": mouse,
        "splits": splits,
        "neuron_ids": neuron_ids,
        "num_neurons": neuron_ids.shape[0],
        "cell_motor_coordinates": cell_motor_coords,
        "trials": [],
    }

    for split in splits:
        if split in constants.folds_splits:
            labeled_split = True
        elif split in constants.unlabeled_splits:
            labeled_split = False
        else:
            raise ValueError(f"Unknown data split '{split}'")
        trial_ids = np.argwhere(tiers == split).ravel().tolist()

        for trial_id in trial_ids:
            behavior_path = str(mouse_dir / "data" / "behavior" / f"{trial_id}.npy")
            trial_data = {
                "trial_id": trial_id,
                "length": get_length_without_nan(np.load(behavior_path)[0]),
                "video_path": str(mouse_dir / "data" / "videos" / f"{trial_id}.npy"),
                "behavior_path": behavior_path,
                "pupil_center_path": str(mouse_dir / "data" / "pupil_center" / f"{trial_id}.npy"),
            }
            if labeled_split:
                response_path = str(mouse_dir / "data" / "responses" / f"{trial_id}.npy")
                trial_data["response_path"] = response_path
                trial_data["length"] = get_length_without_nan(np.load(response_path)[0])
            mouse_data["trials"].append(trial_data)

    return mouse_data
