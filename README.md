# Solution for Sensorium 2023 Competition

![header](data/readme_images/header.png)

This repository contains the code to reproduce the winning solution to the Sensorium 2023, part of the NeurIPS 2023 competition track.
The competition aims to find the best model that can predict the activity of thousands of neurons in the primary visual cortex of mice in response to videos.
Using dynamic stimuli (videos) instead of static stimuli (images) like in the previous Sensorium competition added a temporal component, making the task much more difficult.

See more about data and the challenge in the [whitepaper](https://arxiv.org/abs/2305.19654).  
One important note to the paper is that additional data for five mice appeared during competition, doubling the dataset's size ([old](https://gin.g-node.org/pollytur/Sensorium2023Data) and [new](https://gin.g-node.org/pollytur/sensorium_2023_dataset) data).

## Solution

Key points:
* DwiseNeuro - novel model architecture for prediction neural activity in the mouse primary visual cortex
* Solid cross-validation strategy with splitting folds by perceptual video hash
* Training on all mice with an option to fill unlabeled samples via distillation

## Architecture

During the competition, I dedicated most of my time to designing the model architecture since it significantly impacted the solution's outcome compared to other components.
A good starting point for model development was a core made up of blocks similar to blocks from EfficientNet but rewritten in 3D layers. 
I iteratively tested various computer vision and deep learning techniques, integrating them into the architecture as the correlation metric improved.

The diagram below illustrates the final architecture, which I named DwiseNeuro:

![architecture](data/readme_images/architecture.png)

DwiseNeuro consists of three main parts: core, cortex, and readouts.
The core consumes sequences of video frames and mouse behavior activity in separate channels, processing temporal and spatial features.
Produced features pass through global pooling over spatial dimensions to aggregate them.
The cortex processes the pooled features independently for each timestep, significantly increasing the channels.
Finally, each readout predicts the activation of neurons for the corresponding mouse.

In the following sections, we will delve deeper into each part of the architecture.

### Core
* Stem
* Inverted Residual 3D Block
  * Factorized depth-wise convolution (R(2+1)D, CSN)
  * SqueezeExcite3d
  * Drop path  
  * Shortcut (interpolate spatial, tile channels, batch norm)
* EfficientNet scaling
* PositionalEncoding3d before each block  
* Global pooling of spatial dimensions

### Cortex
* Conv1d with kernel size 1 is a parallel linear layers
* Channel Shuffle
* Drop path  
* Shortcut (tile channels, batch norm)

### Readout
* Dropout1d
* One layers conv1d with kernel size 1
* Softplus, tuning beta, trainable

## Training
* CutMix
* MicePoissonLoss, all mice in batch
* Fill unlabeled samples via distillation
* 7k fold Cross-validation  
* Split folds by videos perceptual hashes 

## Prediction
* Predict each frame, mean blending overlaps
* Mean blend folds

## Quick setup and start

### Requirements

* Linux (tested on Ubuntu 20.04 and 22.04)
* NVIDIA GPU (models trained on RTX A6000)
* NVIDIA Drivers >= 520, CUDA >= 11.8
* [Docker](https://docs.docker.com/engine/install/)
* [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

Pipeline tuned for training on a single RTX A6000 with 48 GB.
In the case of GPU with less memory, you can use gradient accumulation by increasing the `iter_size` parameter in training configs.
It will worsen the result (by a 0.002 score for `"iter_size": 2`), but it has less than the effect of reducing the batch size.

### Run

Clone the repo and enter the folder.

```bash
git clone git@github.com:lRomul/sensorium.git
cd sensorium
```

Build a Docker image and run a container.

<details><summary>Here is a small guide on how to use the provided Makefile</summary>

```bash
make  # stop, build, run

# do the same
make stop
make build
make run

make  # by default all GPUs passed
make GPUS=all  # do the same
make GPUS=none  # without GPUs

make run GPUS=2  # pass the first two GPUs
make run GPUS='\"device=1,2\"'  # pass GPUs numbered 1 and 2

make logs
make exec  # run a new command in a running container
make exec COMMAND="bash"  # do the same
make stop
```

</details>

```bash
make
```

From now on, you should run all commands inside the docker container.

If you already have the Sensorium 2023 dataset (148 GB), copy it to the folder `./data/sensorium_all_2023/`.
Otherwise, use the script for downloading:

```bash
python scripts/download_data.py
```

You can now reproduce the final results of the solution using the following commands:
```bash
# Train
# The training time is 3.5 days (12 hours per fold) for each experiment on a single A6000
# You can speed up the process by using the --folds argument to train folds in parallel
# Or just download trained weights in the section below
python scripts/train.py -e true_batch_001
python scripts/train.py -e distillation_001

# Predict
# Any GPU with more than 6 GB memory will be enough
python scripts/predict.py -e true_batch_001 -s live_test_main
python scripts/predict.py -e true_batch_001 -s live_test_bonus
python scripts/predict.py -e true_batch_001 -s final_test_main
python scripts/predict.py -e true_batch_001 -s final_test_bonus
python scripts/predict.py -e distillation_001 -s live_test_main
python scripts/predict.py -e distillation_001 -s live_test_bonus
python scripts/predict.py -e distillation_001 -s final_test_main
python scripts/predict.py -e distillation_001 -s final_test_bonus

# Ensemble predictions of two experiments
python scripts/ensemble.py -e distillation_001,true_batch_001 -s live_test_main
python scripts/ensemble.py -e distillation_001,true_batch_001 -s live_test_bonus
python scripts/ensemble.py -e distillation_001,true_batch_001 -s final_test_main
python scripts/ensemble.py -e distillation_001,true_batch_001 -s final_test_bonus

# Final predictions will be there
cd data/predictions/distillation_001,true_batch_001
```

### Trained model weights

You can skip the training step by downloading model weights (9.5 GB) using [torrent file](data/experiments.torrent).  

Place the files in the data directory so that the folder structure is as follows:

```
data
├── experiments
│   ├── distillation_001
│   └── true_batch_001
└── sensorium_all_2023
    ├── dynamic29156-11-10-Video-8744edeac3b4d1ce16b680916b5267ce
    ├── dynamic29228-2-10-Video-8744edeac3b4d1ce16b680916b5267ce
    ├── dynamic29234-6-9-Video-8744edeac3b4d1ce16b680916b5267ce
    ├── dynamic29513-3-5-Video-8744edeac3b4d1ce16b680916b5267ce
    ├── dynamic29514-2-9-Video-8744edeac3b4d1ce16b680916b5267ce
    ├── dynamic29515-10-12-Video-9b4f6a1a067fe51e15306b9628efea20
    ├── dynamic29623-4-9-Video-9b4f6a1a067fe51e15306b9628efea20
    ├── dynamic29647-19-8-Video-9b4f6a1a067fe51e15306b9628efea20
    ├── dynamic29712-5-9-Video-9b4f6a1a067fe51e15306b9628efea20
    └── dynamic29755-2-8-Video-9b4f6a1a067fe51e15306b9628efea20
```
