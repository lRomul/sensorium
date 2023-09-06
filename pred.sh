#!/bin/bash

exp="{$1}"

git fetch --all
git checkout --force "$exp"

python scripts/predict.py -e "$exp" -s folds
python scripts/predict.py -e "$exp" -s live_test_main
python scripts/predict.py -e "$exp" -s final_test_main
