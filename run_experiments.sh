#!/bin/bash

git fetch --all

for exp in "$@"
do
    git reset --hard "$exp"
    make COMMAND="./train_val.sh $exp"
done
