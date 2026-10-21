# Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors (OMAN++)

This repository includes the implementation of the manuscript: 
[**Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors**](https://github.com/tiny-smart/OMAN),
Transaction on Image Processing (TIP 2026)

Hao Lu<sup>1</sup>, Xuhui Zhu<sup>1</sup>, Wenjing Zhang<sup>1</sup>, Yanan Li<sup>2</sup>, Xiang Bai<sup>1</sup>
<sup>1</sup>Huazhong University of Science and Technology, China
<sup>2</sup>Wuhan Institute of Technology, China

[[Manuscript]](doc/OMAN++.pdf) | [Transaction Version (OMAN++, TIP 2026)](https://ieeexplore.ieee.org/document/11739646) | [[Previous Conference Version (OMAN, ICIP 2025)]](https://arxiv.org/abs/2506.13067) | [[Code]](https://github.com/tiny-smart/OMAN)

![WuhanMetroCrowd](pics/WuhanMetroCrowd.png)

## Highlights
- **First Surveillance-view VIC Dataset**: WuhanMetroCrowd dataset features crowded, dynamic pedestrian flows from 7 metro station scenes, with severe occlusions and large variations. Each pedestrian is manually annotated as `pedestrian`, `inflow`, `outflow`, or `both`, and indiscernible regions are masked.
<p align="center">
  <img src="pics/dataset_teaser.png" width="850">
</p>

- **High Performance**: OMAN++ outperforms all existing methods and achieves robust counting performance, especially in crowded scenarios.
<p align="center">
  <img src="pics/qualitative_results.png" width="850">
</p>

<p align="center">
  <img src="pics/qualitative_benchmarks.png" width="850">
</p>


## Overview

Video Individual Counting (VIC) estimates foot traffic by counting unique pedestrians in a video. Existing methods primarily rely on appearance-based O2O matching, limiting performance in crowded scenes.

This work revisits VIC through two informative priors:
- Social grouping prior: pedestrians often form groups, motivating O2M matching through ICG and OMPM;
- Spatial-temporal displacement prior: pedestrians cannot teleport, motivating DPI that extracts motion through DASA, a displacement modulator, and D-OT loss.

![OMAN++](pics/Pipeline.png)

### What's New Compared with OMAN (ICIP 2025)

- A novel crowded VIC benchmark, WuhanMetroCrowd, featuring long sequences, large density/flow variations, and severe occlusions;
- The spatial-temporal displacement prior, working in synergy with the social grouping prior;
- Comprehensive experiments on four benchmarks with consistent improvements over the state of the art.

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
- **WuhanMetroCrowd**: [[Baidu disk]](https://pan.baidu.com/s/1e5f0AiE0sKn3-oCgf99jKQ?pwd=aiqj) or [[Google disk]](https://drive.google.com/file/d/1ORvmQWTRrVWCNWKDczAud7078kPtJkVE/view?usp=sharing). (80 clips, 11,925 frames, 207,545 annotations)

## A Quick Demo
Place pretrained models in ./pretrained. Use our provided video clip or place your video clips in ./examples and run the following command for a quick demonstration of OMAN++. Results are saved in ./examples/results.

```bash
python demo.py --input ./examples/test.mp4 --interval 50
```



## Inference

To test OMAN++ on different datasets, run

```
# SenseCrowd
python test.py

# WuhanMetroCrowd
python test_Metro.py
```

Checkpoint paths can be changed via `--resume`. Results are saved to `outputs/json/video_results_test.json`.

## Evaluation

To evaluate the results after testing, run the corresponding evaluation script:

```
# MovingDroneCrowd
python eval_metrics_Drone.py

# WuhanMetroCrowd
python eval_metrics_Metro.py
```

## Results
- **WuhanMetroCrowd**

| Dataset | MAE ↓ | MSE ↓ | WRAE ↓ | Checkpoint |
| :--: | :--: | :--: | :--: | :--: |
| CGNet | 164.0 | 344.6| 35.0% | [[Baidu]](https://pan.baidu.com/s/1lwI3x74RLV4PVAy0JANDPA?pwd=wkxd) [[Google]](https://drive.google.com/file/d/19R-5bkEcQ3fayhF56kCvfvoxNCHS0q_f/view?usp=sharing) |
| SDNet | 151.8 | 415.4 | 31.1% | [[Baidu]](https://pan.baidu.com/s/1X9vXIa2xn43DloIC_xQ5cQ?pwd=ta8s) [[Google]](https://drive.google.com/file/d/17d7ztkN_C_n1PQ-JiOqRo9xKg3Avuiuk/view?usp=sharing) |
| OMAN (Ours) | 130.3 | 257.3 | 34.4% | [[Baidu]](https://pan.baidu.com/s/1b28hN-v7ZYM4e7ZQGL7Nwg?pwd=han1) [[Google]](https://drive.google.com/file/d/1UtjEay840eG_RkqncPXryX0VAK_IQ05H/view?usp=sharing) |
| **OMAN++ (Ours)** | 45.4 | 62.3 | 18.1% | [[Baidu]](https://pan.baidu.com/s/1FEYFcIgAWDGyOltktJn8mQ?pwd=5fjr) [[Google]](https://drive.google.com/file/d/1TxzG4PUvHTpkpCLLZpX5BwmueHOPGMqh/view?usp=sharing) |


- **SenseCrowd**

| Dataset | MAE ↓ | MSE ↓ | WRAE ↓ | Checkpoint |
| :--: | :--: | :--: | :--: | :--: |
| DRNet | 12.3 | 24.7 | 12.7 | - |
| CGNet | 8.9 | 17.7 | 12.6 | - |
| FMDC | 14.3 | 28.9 | 14.9 | - |
| PDTR | 9.6 | 17.6 | 11.4 | - |
| SDNet | 8.6 | 15.9 | 11.0 | - |
| VIC-SSL | 7.6 | 12.6 | 10.4 | - |
| OMAN (Ours) | 8.3 | 15.4 | 11.1 | - |
| **OMAN++ (Ours)** | 7.7 | 14.6 | 10.4 | [[Baidu]](https://pan.baidu.com/s/1sgSFb3cT0LnElgjvoY1D7Q?pwd=9e6t) [[Google]](https://drive.google.com/file/d/1uMBPOSRsrmyAH9JiNQIXeCzz_7FY1cbD/view?usp=sharing) |


## Citation

If you find this work helpful for your research, please consider citing:

```
@article{OMAN++,
  author={Hao Lu, Xuhui Zhu, Wenjing Zhang, Yanan Li, Xiang Bai},
  booktitle={IEEE Transaction on Image Processing}, 
  title={Crowded Video Individual Counting Informed by Social Grouping and Spatial-Temporal Displacement Priors}, 
  year={2026},
  volume={},
  number={},
  pages={},
  keywords={Video individual counting, foot traffic, one-to-many matching, displacement priors},
  doi={doi={10.1109/TIP.2026.3737046}}}
```

## Permission

The code and dataset are for academic purposes only. Contact: Xuhui Zhu (XuhuiZhu@hust.edu.cn);

For commercial use, Contact: Hao Lu (hlu@hust.edu.cn)

## Acknowledgement

We thank FiberHome Telecommunication Technologies Co., Ltd. and Wuhan Metro Group Co., Ltd. for sharing the data of WuhanMetroCrowd.
