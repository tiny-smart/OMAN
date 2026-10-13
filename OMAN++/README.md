# Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors (OMAN++)

This repository includes the implementation of the manuscript:

[**Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors**](https://github.com/tiny-smart/OMAN) (TIP 2026)

Hao Lu<sup>1</sup>, Xuhui Zhu<sup>1</sup>, Wenjing Zhang<sup>1</sup>, Yanan Li<sup>2</sup>, Xiang Bai<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, China

<sup>2</sup>Wuhan Institute of Technology, China

[[Manuscript]](doc/OMAN++.pdf) | Transaction Version (todo) | [[Previous Conference Version (OMAN, ICIP 2025)]](https://arxiv.org/abs/2506.13067) | [[Code]](https://github.com/tiny-smart/OMAN)


![OMAN++](pics/Pipeline.png)

## Highlights

- **WuhanMetroCrowd dataset**: Our WuhanMetroCrowd dataset features crowded, dynamic pedestrian flows from 7 typical metro station scenes, with severe occlusions and large density variations. Each pedestrian is manually annotated as `pedestrian`, `inflow`, `outflow`, or `both`, and indiscernible regions are masked for fair evaluation.
<p align="center">
  <img src="pics/dataset_teaser.png" width="850">
</p>

- **Effective across all benchmarks**: OMAN++ outperforms all existing methods and achieves robust counting performance by leveraging the social grouping and spatial-temporal displacement priors, especially in crowded scenarios like WuhanMetroCrowd.
<p align="center">
  <img src="pics/qualitative_results.png" width="850">
</p>

<p align="center">
  <img src="pics/qualitative_benchmarks.png" width="850">
</p>




## Overview

Video Individual Counting (VIC) aims to estimate foot traffic by counting unique pedestrians in a video, which is essentially a correspondence problem between frames. Existing VIC approaches mostly follow a one-to-one (O2O) matching strategy conditioned on appearance only, and therefore underperform in crowded scenes such as metro commuting.

This work rethinks the nature of VIC and recognizes two informative priors:

- **Social grouping prior**: pedestrians tend to gather in groups, which inspires relaxing the O2O matching to a one-to-many (O2M) matching, implemented by an Implicit Context Generator (ICG) and a One-to-Many Pairwise Matcher (OMPM);
- **Spatial-temporal displacement prior**: an individual cannot teleport physically, which facilitates a Displacement Prior Injector (DPI) that strengthens O2M matching, feature extraction, and model training via a Displacement-Aware Self-Attention (DASA) block, a displacement modulator, and a displacement-informed Optimal Transport (D-OT) loss.

These designs jointly form **OMAN++**, a simple but strong VIC baseline. To fill the data gap of crowded scenes, we further build **WuhanMetroCrowd**, one of the first VIC datasets characterized by crowded, dynamic pedestrian flows: 80 surveillance videos from 15 metro stations, 11,925 frames (2s sampling), and 223,662 manually annotated pedestrians with inflow/outflow labels.

### What's New Compared with OMAN (ICIP 2025)

- A novel crowded VIC benchmark, WuhanMetroCrowd, featuring long sequences, large density/flow variations, and severe occlusions;
- The spatial-temporal displacement prior with three synergistic modules: DASA, the displacement-modulated matcher, and the D-OT loss;
- Cross-frame displacement-aware self-attention and a displacement-modulated O2M matcher;
- Comprehensive experiments on four benchmarks (SenseCrowd, CroHD, MovingDroneCrowd, WuhanMetroCrowd) with consistent improvements over the state of the art.

## Repository Structure

```
OMAN++
├── train.py                  # training entry
├── test.py                   # inference on SenseCrowd
├── test_Metro.py             # inference on WuhanMetroCrowd
├── engine.py                 # training / evaluation loops
├── eval_metrics_Drone.py     # evaluation on MovingDroneCrowd
├── eval_metrics_Metro.py     # evaluation on WuhanMetroCrowd
├── models/
│   ├── pet.py                # locator (PET) + VIC pipeline
│   ├── vic.py                # ICG + OMPM (coarse-to-fine matcher) + displacement encoder
│   ├── tri_sim_ot_b.py       # displacement-informed optimal transport (D-OT) loss
│   ├── transformer/          # transformer encoder with DASA
│   └── backbones/            # ConvNeXt-S / VGG16 backbones
└── datasets/                 # SenseCrowd / HT21 / Metro / UAVVIC / MovingDroneCrowd loaders
```

## Model Zoo
Pretrained models are available at [[Baidu disk]](https://pan.baidu.com/s/1Cfyq3mV-CleJwDlvXlww5A?pwd=g9j2) and [[Google disk]](https://drive.google.com/drive/folders/1lvLWBbbTAPY9jOxBj9MMkYXXLQ7CYuQN?usp=share_link)

## Installation

Clone and set up the OMAN repository:

```
git clone https://github.com/tiny-smart/OMAN
cd OMAN
conda create -n OMAN++ python=3.9
conda activate OMAN++
pip install -r requirements.txt
```

> Note: OMAN++ shares the environment and the `requirements.txt` (e.g., `torch==2.0.1`, `torchvision==0.15.2`, `timm==0.9.2`, `geomloss==0.2.6`) at the repository root..

## Data Preparation

| Dataset | View | Frame Interval σ | Annotation |
| :-- | :-- | :-- | :-- |
| SenseCrowd | Surveillance | 15 (3s) | ID |
| CroHD (HT21) | Drone | 75 (3s) | ID |
| MovingDroneCrowd | Drone | 4 | ID |
| UAVVIC | Drone | 1 (3s) | In/Out |
| **WuhanMetroCrowd (Ours)** | Surveillance | 1 (2s) | In/Out |

- SenseCrowd / CroHD / MovingDroneCrowd / UAVVIC: please prepare the datasets following their official instructions.
- **WuhanMetroCrowd [Final Version]**: [[Baidu disk]](https://pan.baidu.com/s/108pYcxp1JkNaV27K1O6Ivg?pwd=jt3h) or [[Google disk]](https://drive.google.com/file/d/1qin0o5TGU0QrFR0AA3weFYY62rWDgyTh/view?usp=share_link). Based on the preview version reported in the paper, we have thoroughly re-inspected the dataset and rectified annotation errors. We will report OMAN++'s result on final version in this repo.


## Inference

- To test OMAN++ on different datasets, run

```
# SenseCrowd
python test.py

# WuhanMetroCrowd
python test_Metro.py
```

Checkpoint paths can be changed via `--resume`. Results are saved to `outputs/json/video_results_test.json`.

## Evaluation

- To evaluate the results after testing, run the corresponding evaluation script:

```
# MovingDroneCrowd
python eval_metrics_Drone.py

# WuhanMetroCrowd
python eval_metrics_Metro.py
```

## Results

Here are OMAN++'s Results on *WuhanMetroCrowd-Preview (80 clips, 11,925 frames, 223,662 annotations)* and *WuhanMetroCrowd-Final (80 clips, 11,925 frames, 207,545 annotations)*. Final version results will be updated soon.

| Dataset | MAE ↓ | MSE ↓ | WRAE ↓ | Checkpoint |
| :--: | :--: | :--: | :--: | :-- |
| **WuhanMetroCrowd-Preview** | **87.1** | **160.2** | **19.8%** | [[Baidu]](https://pan.baidu.com/s/1Cfyq3mV-CleJwDlvXlww5A?pwd=g9j2) [[Google]](https://drive.google.com/drive/folders/1lvLWBbbTAPY9jOxBj9MMkYXXLQ7CYuQN?usp=share_link) |
| **WuhanMetroCrowd-Final** | **-** | **-** | **-** | [[Baidu]]() [[Google]]() |



## Citation

If you find this work helpful for your research, please consider citing:

```
todo
```

## Permission

This code is for academic purposes only. Contact: Xuhui Zhu (XuhuiZhu@hust.edu.cn)

## Acknowledgement

We thank FiberHome Telecommunication Technologies Co., Ltd. and Wuhan Metro Group Co., Ltd. for sharing the data of WuhanMetroCrowd.
