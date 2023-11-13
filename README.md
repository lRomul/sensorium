# Solution for Sensorium 2023 Competition

![header](data/readme_images/header.png)

This repository contains the code to reproduce the winning solution to the Sensorium 2023, part of the NeurIPS 2023 competition track.
The competition aims to find the best model that can predict the activity of neurons in the primary visual cortex of mice in response to videos.
The competition introduced a temporal component using dynamic stimuli (videos) instead of static stimuli (images) used in Sensorium 2022, making the task more challenging.

The primary metric of the competition was a single trial correlation.
You can read about the metric, the data, and the task in the [competition paper](https://arxiv.org/abs/2305.19654) [1].
It is important to note that additional data for five mice was introduced during the competition, which doubled the dataset's size ([old](https://gin.g-node.org/pollytur/Sensorium2023Data) and [new](https://gin.g-node.org/pollytur/sensorium_2023_dataset) data).

## Solution

Key points:
* [DwiseNeuro](src/models/dwiseneuro.py) - novel model architecture for prediction neural activity in the mouse primary visual cortex
* Solid cross-validation strategy with splitting folds by perceptual video hash
* Training on all mice with an option to fill unlabeled samples via distillation

## Model Architecture

During the competition, I dedicated most of my time to designing the model architecture since it significantly impacted the solution's outcome compared to other components.
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

The first layer of the module is the stem. It's a point-wise convolution for increasing the number of channels, followed by batch normalization.
The rest of the core consists of inverted residual blocks [2, 3] with a `narrow -> wide -> narrow` channel structure. 

#### Techniques

Several methods were added to the core:
* **Absolute Position Encoding** [4] - summing the encoding to the input of each block allows convolutions to accumulate position information. It's quite important because of the subsequent spatial pooling after the core.
* **Factorized (2+1)D convolution** [5] - 3D depth-wise convolution was replaced with a spatial 2D depth-wise convolution followed by a temporal 1D depth-wise convolution. There are spatial convolutions with stride two in some blocks to compress output size.
* **Shortcut Connections** - completely parameter-free residual shortcuts. 
    * Identity mapping if input and output dimensions are equal. It's the same as the connection proposed in ResNet [6].
    * Nearest interpolation in case of different spatial sizes. 
    * Cycling repeating of channels if they don't match.
* **Squeeze-and-Excitation** [7] - dynamic channel-wise feature recalibration.
* **DropPath** [8, 9] - regularization that randomly drops the block's main path for each sample in batch.

#### Hyperparameters

I found that the number of core blocks and their parameters dramatically affect the outcome.
It's possible to tune channels, strides, expansion ratio, and spatial/temporal kernel sizes.
Obviously, it is almost impossible to start experiments with optimal values.
The problem is mentioned in the EfficientNet [3] paper, which concluded that it is essential to carefully balance model width, depth, and resolution.

After conducting a lot of experiments, I chose the following parameters:
* Four blocks with 64 output channels, three with 128, and two with 256.
* Three blocks have stride two. They are the first in each subgroup from the point above.
* Expansion ratio of the inverted residual block is six.
* Kernel of spatial depth-wise convolution is (1, 3, 3).
* Kernel of temporal depth-wise convolution is (5, 1, 1).

### Cortex

Compared with related works [10], I added a new part of the architecture - the cortex.
It is also common for all mice as the core.
The cortex receives features that have only channels and temporal dimensions.
Spatial information was accumulated thanks to position encoding previously applied in the core and compressed by average pooling after the core.
The primary purpose of the cortex is to smoothly increase the number of channels, which the readouts will further use.

The building element of the module is a grouped 1D convolution followed by the channel shuffle operation [11]. Shortcut connections with stochastic depth similar to the core are also applied.

#### Hyperparameters

Hyperparameters of the cortex were also important:
* Convolution with two groups and kernel size one. Bigger kernel size over temporal dimension has not led to better results.
* Three layers with 1024, 2048, and 4096 channels.

As you can see, the number of channels is quite large.
Groups help optimize computation and memory efficiency.
Channel shuffle operation allows the sharing of information between groups of different layers.

### Readouts

The readout is a single grouped 1D convolution, followed by Softplus activation.
Each of the ten mice has its readout with the number of output channels equal to the number of neurons (7863, 7908, 8202, 7939, 8122, 7440, 7928, 8285, 7671, 7495, respectively).

#### Softplus

Keeping the response positive by using Softplus was essential in my pipeline.
It works much better than `ELU + 1` [10], especially when I tune the Softplus beta parameter.
In my case, the optimal beta value was about 0.07.
The value can depend on target normalization.
I didn't use any normalization for target and input tensors during the training.

You can see a comparison of `ELU + 1` and Softplus in the plot below:

![softplus](data/readme_images/softplus.png)

#### Learnable Softplus

I also conducted an experiment where the beta parameter was trainable.
Interestingly, the trained value converged approximately to the optimal, which I found by grid search.
I omitted the learnable Softplus from the solution because it resulted in a slightly worse score.
But this may be an excellent way to quickly and automatically find a good beta.

Here's a numerical stable implementation of learnable Softplus in PyTorch:

```Python
import torch
from torch import nn


class LearnableSoftplus(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(beta)))

    def forward(self, x):
        xb = x * self.beta
        return (torch.clamp(xb, 0) + torch.minimum(xb, -xb).exp().log1p()) / self.beta
```

## Training
* CutMix
* MicePoissonLoss, all mice in batch
* Fill unlabeled samples via distillation
* 7k fold Cross-validation  
* Split folds by videos perceptual hashes 

## Prediction
* Predict each frame, mean blending overlaps
* Mean blend folds

[1] Dynamic Sensorium competition https://arxiv.org/abs/2305.19654  
[2] MobileNetV2 https://arxiv.org/abs/1801.04381  
[3] EfficientNet https://arxiv.org/abs/1905.11946  
[4] Attention Is All You Need https://arxiv.org/abs/1706.03762  
[5] R(2+1)D https://arxiv.org/abs/1711.11248v3  
[6] ResNet https://arxiv.org/abs/1512.03385  
[7] Squeeze-and-Excitation https://arxiv.org/abs/1709.01507  
[8] DropPath https://arxiv.org/abs/1605.07648v4  
[9] Stochastic Depth https://arxiv.org/abs/1603.09382  
[10] Generalization in data-driven models of primary visual cortex https://openreview.net/forum?id=Tp7kI90Htd  
[11] ShuffleNet https://arxiv.org/abs/1707.01083v2  

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
