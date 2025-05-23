# Video Individual Counting With Implicit One-to-many Matching (ICIP 2025)

This repository includes the official implementation of the paper:

[**Video Individual Counting With Implicit One-to-many Matching**](https://arxiv.org/abs/2308.13814)

International Conference on Image Processing (ICIP), 2025

Xuhui Zhu<sup>1</sup>, Jing Xu<sup>2</sup>,Bingjie Wang<sup>1</sup>,Huikang Dai<sup>2</sup>,[Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, China
<sup>2</sup>FiberHome Telecommunication Technologies Co., Ltd., China
<sup>3</sup>University of Rochester, Rochester, USA

[[Paper]](TODO) | [[CODE]](TODO)

![OMAN](pics/Pipeline.PNG)

## Overview

TODO

## Installation

Clone and set up the CGNet repository:

```
git clone TODO
cd OMAN
conda create -n OMAN python=3.9
conda activate OMAN
pip install -r requirements.txt
```


## Data Preparation

- SenseCrowd: Download the dataset from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm#list/path=%2F) or from the original dataset [link](https://github.com/HopLee6/VSCrowd-Dataset).


## Training

TODO

## Inferrence

- Download ImageNet pretrained [ConvNext](TODO), and put it in ```pretrained``` folder. Or you can define your pre-trained model path in [models/backbones/backbone.py](models/backbones/backbone.py)
- To test OMAN on SenseCrowd dataset, run

```
python test.py
```


## Evaluation

- To evaluate the results after testing, run

```
python eval_metrics.py
```


## Pretrained Models

- Environment:

```
python==3.9
pytorch==2.0.1
torchvision==0.15.2
```

- Models:

| Dataset | Model Link | MAE | MSE | WRAE |
| :-- | :-- | :-- | :-- | :-- |
| SenseCrowd | [SENSE.pth](TODO) | 8.58 | 16.80 | 10.89% |

## Citation

If you find this work helpful for your research, please consider citing:

```
TODO
```


## Permission

This code is for academic purposes only. Contact: Xuhui Zhu (XuhuiZhu@hust.edu.cn)

## Acknowledgement

We thank the authors of [CGNet](https://github.com/streamer-AP/CGNet) and [PET](https://github.com/cxliu0/PET) for open-sourcing their work.

