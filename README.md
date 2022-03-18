# Unifying Local, Long-range, and Multi-scale Contexts for Semantic Segmentation

## Content
* [Abstract](#Abstract)
* [Motivation](#Motivation)
* [Network](#Network)
* [Environment](#Our_Environment)
* [Installation](#Installation)
* [Results](#Results)
* [Acknowledgement](#Acknowledgement)

## Abstract
Modeling effective contextual information is crucial for accurate prediction in semantic segmentation. Most existing studies have made efforts to learn multi-scale or long-range context, yet their explorations of modeling context are not comprehensive enough that they cannot build an efficient mechanism to capture local, long-range, and multi-scale context at the same time. In this paper, we propose to unify these three types of contexts for semantic segmentation. Firstly, we introduce a novel attention module named IIWA (Intra- and Inter- Window Attention) which combines intra-window attention and inter-window attention to model local and long-range contextual information respectively. Moreover, to capture long-range dependencies across multiple scales and obtain multi-scale context simultaneously, we utilize the IIWA to construct a feature fusion module and a segmentation decoder. Among them, the feature fusion module aims to aggregate the three contexts from the features with two adjacent scales and provide multiple fused features for the next context aggregation. On the other hand, the segmentation decoder allows the query feature to interact with the enhanced features at multiple scales simultaneously. Therefore, it can build relationships across different scales to obtain multi-scale context while aggregating the local and long-range contexts. With the help of strong contextual modeling, our network surpasses state-of-the-art performances on three
benchmarks: ADE20K, Cityscapes, and Pascal Context. Especially on Pascal Context, we exceed the previous state-of-the-art method by 7.7% mIoU.

## Motivation

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/101050701/157171381-2bec4b09-d996-482a-a5d7-5db603e79845.png" width="80%"/>
</div>

## Network

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/101050701/157171430-d621c4b6-db5e-491c-920c-1f9682ff62e5.png" width="80%"/>
</div>

## Our_Environment
```
torch==1.7.1
torchvision==0.8.2
mmsegmentation==0.20.0
mmcv-full==1.4.1
scipy==1.6.0
einops==0.3.0
```

## Installation

Our project is developed based on [MMsegmentation](https://github.com/open-mmlab/mmsegmentation). Please follow the official MMsegmentation [INSTALL.md](docs/install.md) and [getting_started.md](docs/getting_started.md) for installation and dataset preparation.

To install pytorch, please follow the official [pytorch](https://pytorch.org/get-started/locally/).

## Results 

The model weights will be public gradually. The extracted code is `2050`.

### ADE20K

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (ss) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 512x512  | 16          | 160000   | -        |  48.30 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swins_512x512_160k_ade20k_type1.py)  | - &#124; -     |
| IIWA | Swin-B | 512x512  | 16          | 160000   | -        | 51.85 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinb_512x512_160k_ade20k_type1.py)  | [model not support]() &#124; [log](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swinb_ade20k.log.json)     |
| IIWA | Swin-L | 512x512  | 8           | 160000   | -              | 53.20 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinl_512x512_160k_ade20k_type1.py)  | [model](https://pan.baidu.com/s/1XRVS17NxuHP6S3c7TFLnxw) &#124; [log not support]()     |

### Cityscapes

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (val) (ms+f) | mIoU (test) (ms+f) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 512x1024 | 16         | 80000 | -  | 83.1 | 82.5 | [config_val](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swins_512x1024_80k_cityscapes_val_type2.py) &#124; [config_test](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swins_512x1024_80k_cityscapes_test_type3.py)  | [model_test](https://pan.baidu.com/s/1Vv8OlPcCwYEOFu5HPOTMHw) &#124; [log_val](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swins_cityscapes_val.log.json)     |
| IIWA | Swin-B | 512x1024 | 16         | 80000   | -  | 84.0 | 83.2 | [config_val](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinb_512x1024_80k_cityscapes_val_type2.py) &#124; [config_test](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinb_512x1024_80k_cityscapes_test_type3.py)  | - &#124; -     |

### Pascal Context

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (ss) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 480x480 | 32         | 40000 | -  | 55.8 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swins_480x480_40k_pascal_context_59_type2.py)  | [model](https://pan.baidu.com/s/1QQduoRi1me0Jmuer3jgjwQ) &#124; [log](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swins_pascal.log.json)     |
| IIWA | Swin-B | 480x480 | 24         | 40000   | -  | 62.9 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinb_480x480_40k_pascal_context_59_type1.py)  | [model](https://pan.baidu.com/s/15SHxfSZdGo7nEgKgWj5URw) &#124; [log](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swinb_pascal.log.json)     |

## Acknowledgement

Thanks to previous open-sourced repo:
[MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
