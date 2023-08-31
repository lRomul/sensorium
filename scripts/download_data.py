import shutil
import zipfile
import argparse
from pathlib import Path

import deeplake
import numpy as np
import requests
from tqdm import tqdm

from src import constants


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default=constants.sensorium_dir, type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    sensorium_dir = args.path
    sensorium_dir.mkdir(parents=True, exist_ok=True)

    for mouse in constants.mice:
        file_name = f"{mouse}.zip"
        dataset = constants.mouse2dataset[mouse]
        url = constants.dataset2url_format[dataset].format(file_name=file_name)
        zip_path = sensorium_dir / file_name
        mouse_dir = sensorium_dir / mouse

        if mouse_dir.exists():
            print(f"Folder '{str(mouse_dir)}' already exists, skip download")
            continue

        print(f"Download '{url}' to '{zip_path}'")
        zip_path.unlink(missing_ok=True)
        with requests.get(url, stream=True) as r:
            total_length = int(r.headers.get("Content-Length"))
            with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as raw:
                with open(zip_path, 'wb') as output:
                    shutil.copyfileobj(raw, output)

        print("Unzip", zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_file:
            zip_file.extractall(sensorium_dir)

        print("Delete", zip_path)
        zip_path.unlink()
        shutil.rmtree(sensorium_dir / "__MACOSX", ignore_errors=True)

        if mouse in constants.new_mice:
            continue
        for split in constants.unlabeled_splits:
            dataset = deeplake.load(f"hub://sinzlab/Sensorium_2023_{mouse}_{split}")
            trials_ids = dataset.id.numpy().astype(int).ravel().tolist()
            for index, trial_id in enumerate(trials_ids):
                responses_path = mouse_dir / "data" / "responses" / f"{trial_id}.npy"
                responses = dataset.responses[index].numpy()
                np.save(str(responses_path), responses)
