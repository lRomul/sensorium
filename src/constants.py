from pathlib import Path

work_dir = Path("/workdir")
data_dir = work_dir / "data"
sensorium_dir = data_dir / "sensorium_2023"

configs_dir = work_dir / "configs"
experiments_dir = data_dir / "experiments"
predictions_dir = data_dir / "predictions"

new_mice = [
    "dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20",
    "dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20",
]
num_new_mice = len(new_mice)
new_num_neurons = [7863, 7908, 8202, 7939, 8122]
old_mice = [
    "dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce",
    "dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce",
]
num_old_mice = len(old_mice)
old_num_neurons = [7440, 7928, 8285, 7671, 7495]
dataset2mice = {
    "new": new_mice,
    "old": old_mice,
}
mouse2dataset = {m: d for d, mc in dataset2mice.items() for m in mc}
dataset2url_format = {
    "new": "https://gin.g-node.org/pollytur/sensorium_2023_dataset/raw/master/{file_name}",
    "old": "https://gin.g-node.org/pollytur/Sensorium2023Data/raw/master/{file_name}",
}

mice = new_mice + old_mice
num_neurons = new_num_neurons + old_num_neurons

num_mice = len(mice)
index2mouse: dict[int, str] = {index: mouse for index, mouse in enumerate(mice)}
mouse2index: dict[str, int] = {mouse: index for index, mouse in enumerate(mice)}
mouse2num_neurons: dict[str, int] = {mouse: num for mouse, num in zip(mice, num_neurons)}
mice_indexes = list(range(num_mice))

labeled_splits = ["train", "val"]
unlabeled_splits = ["live_test_main", "live_test_bonus", "final_test_main", "final_test_bonus"]

submission_skip_first = 50
submission_skip_last = 1
