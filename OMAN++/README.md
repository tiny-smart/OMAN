# OMAN++: Enhanced Video Individual Counting with Multi-Dataset Support

This repository includes the enhanced implementation of OMAN with support for multiple datasets:

**OMAN++** extends the original OMAN framework to support additional datasets and provides improved evaluation metrics and visualization tools.

Based on: [**Video Individual Counting With Implicit One-to-many Matching**](https://arxiv.org/abs/2506.13067)

Xuhui Zhu<sup>1</sup>, Jing Xu<sup>2</sup>, Bingjie Wang<sup>3</sup>, Huikang Dai<sup>2</sup>, [Hao Lu](https://sites.google.com/site/poppinace/)<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>FiberHome Telecommunication Technologies Co., Ltd., China

<sup>3</sup>University of Rochester, Rochester, USA

[[Paper]](https://arxiv.org/abs/2506.13067) | [[Original Code]](https://github.com/tiny-smart/OMAN)

## Overview

OMAN++ is an enhanced version of OMAN (One-to-Many mAtchiNg) for Video Individual Counting (VIC). This implementation extends the original framework to support multiple benchmark datasets and provides comprehensive evaluation tools.

**Key Features:**
- Support for multiple datasets: SenseCrowd, Drone, HT21, and Metro
- Enhanced evaluation metrics for each dataset
- Advanced visualization tools for result analysis
- Improved training and testing pipelines
- Dataset-specific configurations and optimizations

## Supported Datasets

OMAN++ provides full support for the following datasets:

- **SenseCrowd**: Original benchmark dataset for VIC
- **Drone**: Aerial view pedestrian counting dataset
- **HT21**: Head tracking dataset
- **Metro**: Metro station crowd counting dataset

## Installation

Clone and set up the OMAN++ repository:

```bash
git clone https://github.com/tiny-smart/OMAN
cd OMAN/OMAN++
conda create -n OMAN++ python=3.9
conda activate OMAN++
pip install -r requirements.txt
```

## Data Preparation

### SenseCrowd
Download the dataset from [Baidu disk](https://pan.baidu.com/s/1OYBSPxgwvRMrr6UTStq7ZQ?pwd=64xm#list/path=%2F) or from the original dataset [link](https://github.com/HopLee6/VSCrowd-Dataset).

### Drone Dataset
Please prepare the Drone dataset according to its official instructions.

### HT21 Dataset
Please prepare the HT21 dataset according to its official instructions.

### Metro Dataset
Please prepare the Metro dataset according to its official instructions.

Place the prepared datasets in the `datasets/` directory following the structure specified in the dataset configuration files.

## Training

- Download ImageNet pretrained ConvNext [[baidu disk]](https://pan.baidu.com/s/1oxxcD6h-JiRdJ4VItHJIUQ?pwd=ubqt) [[Google drive]](https://drive.google.com/file/d/1tDGb3DAEITajJ5xlzYSxCa5x4dnTbfJ-/view?usp=sharing), and put it in `pretrained` folder.

- To train OMAN++ on different datasets, modify the configuration in `train.py` and run:

```bash
python train.py --dataset [DATASET_NAME] --epochs 10
```

Or use the training script:

```bash
bash train.sh
```

## Inference

To test OMAN++ on different datasets:

### SenseCrowd
```bash
python test.py
```

### HT21 Dataset
```bash
python test_HT21.py
```

### Metro Dataset
```bash
python test_Metro.py
```

## Evaluation

To evaluate the results after testing, use the corresponding evaluation script for each dataset:

### SenseCrowd
```bash
python eval_metrics.py
```

### Drone Dataset
```bash
python eval_metrics_Drone.py
```

### HT21 Dataset
```bash
python eval_metrics_HT21.py
```

### Metro Dataset
```bash
python eval_metrics_Metro.py
```

## Visualization

OMAN++ provides advanced visualization tools for result analysis:

### SenseCrowd
```bash
python my_plot.py
```

### Drone Dataset
```bash
python my_plot_Drone.py
```

### HT21 Dataset
```bash
python my_plot_HT21.py
```

### Metro Dataset
```bash
python my_plot_Metro.py
```

## Environment

- Recommended environment:

```
python==3.9
pytorch==2.0.1
torchvision==0.15.2
```

## Model Architecture

OMAN++ uses the same architecture as OMAN:
- Backbone: ConvNext (pretrained on ImageNet)
- Implicit context generator
- One-to-many pairwise matcher
- Transformer-based decoder with 2 layers

## Citation

If you find this work helpful for your research, please consider citing:

```bibtex
@INPROCEEDINGS{11084398,
  author={Zhu, Xuhui and Xu, Jing and Wang, Bingjie and Dai, Huikang and Lu, Hao},
  booktitle={2025 IEEE International Conference on Image Processing (ICIP)}, 
  title={Video Individual Counting with Implicit One-to-Many Matching}, 
  year={2025},
  volume={},
  number={},
  pages={61-66},
  keywords={Legged locomotion;Pedestrians;Sensitivity;Codes;Image processing;Semantics;Benchmark testing;Generators;Standards;Context modeling;Video individual counting;pedestrian flux;semantic correspondence;one-to-many matching},
  doi={10.1109/ICIP55913.2025.11084398}}
```

## Permission

This code is for academic purposes only. Contact: Xuhui Zhu (XuhuiZhu@hust.edu.cn)

## Acknowledgement

We thank the authors of [CGNet](https://github.com/streamer-AP/CGNet) and [PET](https://github.com/cxliu0/PET) for open-sourcing their work. We also thank the creators of the SenseCrowd, Drone, HT21, and Metro datasets for making their data publicly available.
