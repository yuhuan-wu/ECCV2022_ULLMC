# Unifying Local, Long-range, and Multi-scale Contexts for Semantic Segmentation

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

## Our Environment
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

## Results and models 

The model weights will be public gradually. The extracted code is `2050`.

### ADE20K

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (ss) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 512x512  | 16          | 160000   | -        |  48.30 | [config]()  | [model]() &#124; [log]()     |
| IIWA | Swin-B | 512x512  | 16          | 160000   | -        | 51.85 | [config]()  | [model]() &#124; [log]()     |
| IIWA | Swin-L | 512x512  | 8           | 160000   | -              | 53.20 | [config]()  | [model]() &#124; [log]()     |

### Cityscapes

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (val) (ms+f) | mIoU (test) (ms+f) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | ----------------- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 512x1024 | 16         | 80000 | -  | 83.1 | 82.5 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swins_512x1024_80k_cityscapes_test_type3.py)  | [model]() &#124; [log]()     |
| IIWA | Swin-B | 512x1024 | 16         | 80000   | -  | 84.0 | 83.2 | [config]()  | - &#124; -     |

### Pascal Context

| Method | Backbone | Crop Size | Batch Size | Lr schd | Mem (GB) | mIoU (ss) | config                                                                                                                          | download                                                                                                                                                                                                                                                                                                                                                     |
| ------ | -------- | --------- | ---------- | ------- | -------- | --------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| IIWA | Swin-S | 480x480 | 32         | 40000 | -  | 55.8 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/config/IIWA/iiwa_swins_480x480_40k_pascal_context_59_type2.py)  | [model](https://pan.baidu.com/s/1QQduoRi1me0Jmuer3jgjwQ) &#124; [log](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swins_pascal.log.json)     |
| IIWA | Swin-B | 480x480 | 24         | 40000   | -  | 62.9 | [config](https://github.com/AnonymousECCV/ECCV2022_ULLMC/blob/main/configs/IIWA/iiwa_swinb_480x480_40k_pascal_context_59_type1.py)  | [model]() &#124; [log](https://github.com/AnonymousECCV/ECCV2022_ULLMC/releases/download/v1.0/swinb_pascal.log.json)     |

## Acknowledgement

Thanks to previous open-sourced repo:
[MMsegmentation](https://github.com/open-mmlab/mmsegmentation)
