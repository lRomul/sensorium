#!/bin/bash

exp="{$1}"
fold="{$2:-all}"

git checkout --force "$exp"
python scripts/train.py -e "$exp" -f "$fold"
