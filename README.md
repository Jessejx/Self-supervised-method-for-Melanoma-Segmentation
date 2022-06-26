# Self-supervised-method-for-Melanoma-Segmentation
**Paper Title**: _JIANet: Jigsaw-Invariant Self-supervised Learning of Auto-encoder Based Reconstruction for Melanoma Segmentation_

## Table of Contents

- [Background and Method](#background)

- [Configurations](#Configurations)

- [Pretrained weights](#pretrained_weights)



- [Install](#install)


## Background and Method
Self-supervised learning (SSL) is a novel method of deep learning, can be regarded as an intermediate form between unsupervised and supervised learning, which can avoid expen- sive and time-consuming data annotations and collections. It learns in two steps. First, SSL methods were learned visual features with automatically generated labels (i.e., pretext task). Second, the learned features serve as a pre-trained model and are transferred to downstream tasks with few human-annotated labels (i.e., downstream task). Downstream tasks are multiple different tasks that are used to evaluate the quality of features such as melanoma segmentation by fine-tuning. In this repositories, we present pretext task (i.e., proposed model) and downstream task (i.e., pspnet)

<div align=center>
<img src="https://github.com/Jessejx/Self-supervised-method-for-Melanoma-Segmentation/blob/main/2.svg" width="500px">
</div>

## Configurations

Please install the following libraries, or load the provided ‘environment.yml’ file

1. python 3.7.0
2. pytorch 1.7.0 + cu110
3. albumentations 0.5.2
4. tqdm

## Pretrained weights
pretrained weights are in: [Pretraining weights](https://pan.baidu.com/s/1vSGG4etOjx0_aFuq1qqwQw) Password is: bvq9

## Datasets
The pretraining dataset can be downloaded from the following URLs:

1. [Pretraining dataset](https://challenge.isic-archive.com/data/)
2. [segmentation dataset](https://challenge.isic-archive.com/data/)

### Segmentation 
In this repositories, we present pspnet from downstream segmentation task:

1. Download the pretrained weights:
2. Load the pretrained weitghts from backbone network in the 'pspnet.py':
3. Afterward, please run the 'baseline_pspnet.py' to segment skin lesion

## Pretraining



### Any optional sections

## Usage

```
```

Note: The `license` badge image link at the top of this file should be updated with the correct `:user` and `:repo`.

### Any optional sections

## API

### Any optional sections

## More optional sections

# Results

<div align=center>
<img src="https://github.com/Jessejx/Self-supervised-method-for-Melanoma-Segmentation/blob/main/1.svg" width="750px">
</div>

## Contributing
