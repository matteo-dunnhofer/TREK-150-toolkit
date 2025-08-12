# [Is Tracking really more challenging in First Person Egocentric Vision?](https://machinelearning.uniud.it/datasets/vista/)
## The VISTA Benchmark Dataset and Toolkit
## ICCV 2025 ✨ Highlight ✨

<!-- start badges -->
[![arXiv-2507.16015](https://img.shields.io/badge/arXiv-2209.13502-red.svg)](https://arxiv.org/abs/2507.16015)
<!-- end badges -->

![VISTA](vista.jpg)

> Visual object tracking and segmentation are becoming fundamental tasks for understanding human activities in egocentric vision. Recent research has benchmarked state-of-the-art methods and concluded that first person egocentric vision presents challenges compared to previously studied domains. However, these claims are based on evaluations conducted across significantly different scenarios. Many of the challenging characteristics attributed to egocentric vision are also present in third person videos of human-object activities. This raises a critical question: how much of the observed performance drop stems from the unique first person viewpoint inherent to egocentric vision versus the domain of human-object activities? To address this question, we introduce a new benchmark study designed to disentangle such factors. Our evaluation strategy enables a more precise separation of challenges related to the first person perspective from those linked to the broader domain of human-object activity understanding. By doing so, we provide deeper insights into the true sources of difficulty in egocentric tracking and segmentation, facilitating more targeted advancements on this task.

## Authors
Matteo Dunnhofer (1,2)
Zaira Manigrasso (1)
Christian Micheloni (1)

* (1) Machine Learning and Perception Lab, University of Udine, Italy
* (2) Centre for Vision Research, York University, Canada

**Contact:** [matteo.dunnhofer@uniud.it](mailto:matteo.dunnhofer@uniud.it)


## Citing
When using the dataset or toolkit, please reference:

```
@InProceedings{vista,
author = {Dunnhofer, Matteo and Manigrasso, Zaira and Micheloni, Christian},
title = {Is Tracking really more challenging in First Person Egocentric Vision?},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {Oct},
year = {2025}
}
```

## The VISTA Dataset

The annotations produced for this dataset are contained in [this archive](https://machinelearning.uniud.it/datasets/vista/VISTA-annotations.zip) (you will find a json file with train/test annotations long/short-term sequences contained in VISTA).
Video frames of the VISTA's sequences cannot be directly re-distributed due to the EgoExo4D policy. So you won't directly find them in the annotations folder, but they will be automatically downloaded for you.

The full VISTA dataset can be built just by running
```
pip install got10k
git clone https://github.com/matteo-dunnhofer/fpv-tracking-toolkit
cd fpv-tracking-toolkit
python download_vista.py
```
This will download the original EgoExo4D MP4 take videos, extract the frames of interest using ```opencv```. After the whole process is completed, you will find frames in the ```frames``` folder. Each one defines a take as annotated in EgoExo4D. To perform the download, you have to accept and sign the [EgoExo4D's license](https://ego4ddataset.com/egoexo-license/). The code to download the take videos is based on the [EgoExo4D's cli](https://docs.ego-exo4d-data.org/download/). Make sure to follow the original instructions to setup the cli.

Each annotation JSON file contains a dictionary with the following fields:

- ```annotation_id```: Unique identifier for the annotation sequence.
- ```take```: Name of the original video take from which the sequence is extracted.
- ```fpv_camera_name```: Name of the first-person (egocentric) camera used for the sequence.
- ```tpv_camera_name```: Name of the third-person (exocentric) camera used for the sequence.
- ```frame_annotations```: Dictionary mapping each frame index to its annotation data. For each frame:
    - ```fpv``` (optional): Present if first-person annotation exists for the frame.
        - ```box```: List of bounding box coordinates `[x, y, w, h]` for the target object in the FPV frame.
        - ```mask```: Encoded segmentation mask for the target object in the FPV frame (COCO RLE format).
        - ```attributes``` (optional, test split and long-term mode): List of attribute strings describing tracking situations for the frame.
    - ```tpv``` (optional): Present if third-person annotation exists for the frame.
        - ```box```: List of bounding box coordinates `[x, y, w, h]` for the target object in the TPV frame.
        - ```mask```: Encoded segmentation mask for the target object in the TPV frame (COCO RLE format).
        - ```attributes``` (optional, test split and long-term mode): List of attribute strings describing tracking situations for the frame.

If no FPV and TPV are present, it means no annotation is present for that frame. In VISTA, video frames are sampled at 5 FPS and annotations at 1 FPS.

The code was tested with Python 3.11.7. All the temporary files (e.g. ```*.mp4``` video takes) generated during the download procedure will be removed automatically after the process is completed. The download process can be resumed from the last downloaded sequence if prematurely stopped.

The download process could take some quite to complete depending on connection quality.

## Toolkit
The code available in this repository allows you to replicate the experiments and results presented in our paper. Our code is built upon the [```got10k-toolkit```](https://github.com/got-10k/toolkit) toolkit and inherits the same tracker definition. Please check such a GitHub repository to learn how to use our toolkit. The only difference is that you have to change the name of the toolkit when importing the python sources (e.g. you have to use ```from toolkit.experiments import ExperimentVISTA``` instead of ```from got10k.experiments import ExperimentVISTA```). Otherwise, you can try to integrate the original ```got10k-toolkit``` with the sources of this repository (it should be easy).


### Evaluate Bounding-box Trackers
In the following, we provide examplar code to run an evaluation of the bounding-box [SiamFC tracker](https://github.com/got-10k/siamfc) on the VISTA benchmark.
```
git clone https://github.com/matteo-dunnhofer/fpv-tracking-toolkit
cd fpv-tracking-toolkit

# Clone the GOT-10k pre-trained SiamFC
pip install torch opencv-python got10k
git clone https://github.com/matteo-dunnhofer/siamfc-pytorch.git siamfc_pytorch
wget -nc --no-check-certificate "https://drive.google.com/uc?export=download&id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4" -O siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth
      
python example_vista_box.py
```

### Evaluate Video Object Segmentation Trackers 
In the following, we provide examplar code to run an evaluation of the VOS method [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) on the VISTA benchmark.
```
git clone https://github.com/matteo-dunnhofer/fpv-tracking-toolkit
cd fpv-tracking-toolkit

# Clone DAM4SAM and create environment
git clone https://github.com/jovanavidenovic/DAM4SAM.git
cd DAM4SAM
conda create -n dam4sam_env python=3.10.15
conda activate dam4sam_env
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install opencv-python got10k

cd checkpoints && \
./download_ckpts.sh 
cd ..
      
python example_vista_mask.py
```


## Tracker Results
The raw results of the trackers benchmarked in our paper can be downloaded from [this link](https://machinelearning.uniud.it/datasets/vista/VISTA-results.zip).

## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommercial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.

Copyright © Machine Learning and Perception Lab - University of Udine - 2021 - 2025
