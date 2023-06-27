from pathlib import Path

work_dir = Path("/workdir")
data_dir = work_dir / "data"
deeplake_dir = data_dir / "deeplake"

configs_dir = work_dir / "configs"
experiments_dir = work_dir / "experiments"
predictions_dir = work_dir / "predictions"

deeplake_path_format = "hub://sinzlab/Sensorium_2023_{mouse}_{split}"

mice = [
    "dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce",
]

num_responses = [7440, 7928, 8285, 7671, 7495]

index2mouse: dict[int, str] = {index: mouse for index, mouse in enumerate(mice)}
mouse2index: dict[str, int] = {mouse: index for index, mouse in enumerate(mice)}
mouse2num_responses: dict[str, int] = {mouse: num for mouse, num in zip(mice, num_responses)}

labeled_splits = ["train", "val"]
unlabeled_splits = ["live_test_main", "live_test_bonus", "final_test_main", "final_test_bonus"]
