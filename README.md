# [Is First Person Vision Challenging for Object Tracking?](https://machinelearning.uniud.it/datasets/trek150/)
## The TREK-150 Benchmark Dataset and Toolkit

<!-- start badges -->
[![arXiv-2011.12263](https://img.shields.io/badge/arXiv-2011.12263-red.svg)](https://arxiv.org/abs/2011.12263)
<!-- end badges -->

> Understanding human-object interactions is fundamental in First Person Vision (FPV). Tracking algorithms which follow the objects manipulated by the camera wearer can provide useful cues to effectively model such interactions. Visual tracking solutions available in the computer vision literature have significantly improved their performance in the last years for a large variety of target objects and tracking scenarios. However, despite a few previous attempts to exploit trackers in FPV applications, a methodical analysis of the performance of state-of-the-art trackers in this domain is still missing. In this paper, we fill the gap by presenting the first systematic study of object tracking in FPV. Our study extensively analyses the performance of recent visual trackers and baseline FPV trackers with respect to different aspects and considering a new performance measure. This is achieved through TREK-150, a novel benchmark dataset composed of 150 densely annotated video sequences. Our results show that object tracking in FPV is challenging, which suggests that more research efforts should be devoted to this problem so that tracking could benefit FPV tasks.

## Authors
Matteo Dunnhofer (1)
Antonino Furnari (2)
Giovanni Maria Farinella (2)
Christian Micheloni (1)

* (1) Machine Learning and Perception Lab, University of Udine, Italy
* (2) Image Processing Laboratory, University of Catania, Italy

**Contact:** [matteo.dunnhofer@uniud.it](mailto:matteo.dunnhofer@uniud.it)


## Citing
When using the dataset, please reference:

```
@InProceedings{TREK150,
author = {Dunnhofer, Matteo and Furnari, Antonino and Farinella, Giovanni Maria and Micheloni, Christian},
title = {Is First Person Vision Challenging for Object Tracking?},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
month = {Oct},
year = {2021}
}
```

## Details on the benchmark

The annotations built for this dataset are contained in the ```annotations``` folder (you will find a zip archive for every sequence contained in TREK-150).
Video frames of the TREK-150's sequences cannot be directly re-distributed due to the EK-55 policy. So you won't directly find them in this folder, but they will be automatically downloaded for you.

The full TREK-150 dataset can be built just by running
```
git clone https://github.com/matteo-dunnhofer/TREK-150-toolkit
cd TREK-150-toolkit
python download.py
```
This will download the original EK-55 MP4 videos, extract the frames of interest using ```ffmpeg```, and prepare the annotation files that will be extracted from the zip archives. After the whole process is completed, you will find 100 directories in the ```dataset``` folder. Each one defines a video sequence.

Each sequence folder will contain a directory

 - ```img/```: Contains the video frames of the sequence as ```*.jpg``` files.

and the following ```*.txt``` files:

 - ```groundtruth_rect.txt```: Contains the ground-truth trajectory of the target object. The comma-separated values on each line represent the bounding-box locations [x,y,w,h] (coordinates of the top-left corner, and width and height) of the target object at each respective frame (1st line -> target location for the 1st frame, last line -> target location for the last frame). A line with values -1,-1,-1,-1 specifies that the target object is not visible in such a frame.
 - ```action_target.txt```: Contains the labels for the action performed by the camera wearer (as verb-noun pair) and the target object category. The file reports 3 line-separated numbers. The first value is the action verb label, the second is the action noun label, the third is the noun label for the target object (action noun and target noun do not coincide on some sequences). The verb labels are obtained considering the ```verb_id``` indices of [this file](https://github.com/epic-kitchens/epic-kitchens-55-annotations/blob/master/EPIC_verb_classes.csv). The noun labels and target noun labels are obtained considering the ```noun_id``` indices of [this file](https://github.com/epic-kitchens/epic-kitchens-55-annotations/blob/master/EPIC_noun_classes.csv).
 - ```attributes.txt```: Contains the tracking attributes of the sequence. The file reports line-separated strings that depend on the tracking situations happening in the sequence. The strings are acronyms and explanations can be found in Table 2 of the main paper.
 - ```frames.txt```: Contains the frame indices of the sequence with respect to the full EK-55 video.
 - ```anchors.txt```: Contains the frame indices of the starting points (anchors) and the direction of evaluation (0 -> forward in time, 1 -> backward in time) to implement the MSE (multi-start evaluation) protocol.

The code was tested with Python 3.7.9 and ```ffmpeg``` 4.0.2. All the temporary files (e.g. ```*.MP4``` files, not relevant frames) generated during the download procedure will be removed automatically after the process is completed. The download process can be resumed from the last downloaded sequence if prematurely stopped.

The download process could take up to 24h to complete.

## Toolkit
The toolkit available in this repositery allows you to replicate the results presented in our paper. Our code is built upon the [got10k-toolkit](https://github.com/got-10k/toolkit) and inherits the same tracker definition. Please check such a GitHub repository to learn how to use our toolkit. The only difference is that you have to change the name of the toolkit when importing the python sources (e.g. you have to use ```from toolkit.experiments import ExperimentTREK150``` instead of ```from got10k.experiments import ExperimentTREK150```)

To run an examplar evaluation of the [SiamFC tracker](https://github.com/got-10k/siamfc) on the TREK-150 benchmark run the following commands.
```
git clone https://github.com/matteo-dunnhofer/TREK-150-toolkit
cd TREK-150-toolkit

# Clone a the GOT-10k pre-trained SiamFC
pip install torch opencv-python got10k
git clone https://github.com/matteo-dunnhofer/siamfc-pytorch.git siamfc_pytorch
wget -nc --no-check-certificate "https://drive.google.com/uc?export=download&id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4" -O siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth
      
python example_trek150.py
```
This will download and prepare the TREK-150 dataset if you have not done before.

You can proceed similarly to perform experiments on the [OTB benchmarks](http://cvlab.hanyang.ac.kr/tracker_benchmark/benchmark_v10.html) using our performance measures.
```
git clone https://github.com/matteo-dunnhofer/TREK-150-toolkit
cd TREK-150-toolkit

# Clone a the GOT-10k pre-trained SiamFC
pip install torch opencv-python got10k
git clone https://github.com/matteo-dunnhofer/siamfc-pytorch.git siamfc_pytorch
wget -nc --no-check-certificate "https://drive.google.com/uc?export=download&id=1UdxuBQ1qtisoWYFZxLgMFJ9mJtGVw6n4" -O siamfc_pytorch/pretrained/siamfc_alexnet_e50.pth

python example_otb100.py
```

## Tracker Results
The raw results of the trackers benchmarked in our paper can be downloaded from [this link](https://uniudamce-my.sharepoint.com/:u:/g/personal/matteo_dunnhofer_uniud_it/EbnWz8FPqetPgXErg1SNNhABeBpTrlMMqKFr6xIxreD6UQ?e=2Z8w2C).

## License
All files in this dataset are copyright by us and published under the 
Creative Commons Attribution-NonCommercial 4.0 International License, found 
[here](https://creativecommons.org/licenses/by-nc/4.0/).
This means that you must give appropriate credit, provide a link to the license,
and indicate if changes were made. You may do so in any reasonable manner,
but not in any way that suggests the licensor endorses you or your use. You
may not use the material for commercial purposes.

Copyright © Machine Learning and Perception Lab - University of Udine - 2021