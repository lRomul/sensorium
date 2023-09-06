#!/bin/bash

exp="{$1}"
folds="{$2:-all}"

git fetch --all
git checkout --force "$exp"

python scripts/train.py -e "$exp" -f "$folds"
