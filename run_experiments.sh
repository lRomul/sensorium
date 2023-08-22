#!/bin/bash

git fetch --all

for exp in "$@"
do
    git checkout --force "$exp"
    make COMMAND="./train_val.sh $exp"
done
