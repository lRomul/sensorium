import numpy as np

from src.phash import calculate_video_phash
from src.utils import get_length_without_nan
from src import constants


def create_videos_phashes(mouse: str) -> np.ndarray:
    mouse_dir = constants.sensorium_dir / mouse
    tiers = np.load(str(mouse_dir / "meta" / "trials" / "tiers.npy"))
    phashes = np.zeros(tiers.shape[0], dtype=np.uint64)
    for trial_id, tier in enumerate(tiers):
        if tier == "none":
            continue
        video = np.load(str(mouse_dir / "data" / "videos" / f"{trial_id}.npy"))
        phashes[trial_id] = calculate_video_phash(video)
    return phashes


def get_folds_tiers(mouse: str, num_folds: int):
    tiers = np.load(str(constants.sensorium_dir / mouse / "meta" / "trials" / "tiers.npy"))
    phashes = create_videos_phashes(mouse)
    trial_ids = np.argwhere((tiers == "train") | (tiers == "oracle")).ravel()
    for trial_id in trial_ids:
        fold = int(phashes[trial_id]) % num_folds  # group k-fold by video hash
        tiers[trial_id] = f"fold_{fold}"
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
