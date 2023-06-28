import shutil
import zipfile
import argparse
from pathlib import Path

import requests
from tqdm import tqdm

from src import constants

DOWNLOAD_URL_FORMAT = "https://gin.g-node.org/pollytur/Sensorium2023Data/raw/master/{file_name}"


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", default="data/sensorium", type=Path)
    parser.add_argument("-m", "--mice", default="all", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    if args.mice == "all":
        mice = constants.mice
    else:
        mice = {constants.mice[int(i)] for i in args.mice.split(",")}

    sensorium_dir = args.path
    sensorium_dir.mkdir(parents=True, exist_ok=True)

    for mouse in mice:
        file_name = f"{mouse}.zip"
        url = DOWNLOAD_URL_FORMAT.format(file_name=file_name)
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
