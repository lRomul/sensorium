#!/bin/bash

exp="$1"

python scripts/train.py -e "$exp"
python scripts/predict.py -e "$exp" -s val
