## audio2face MM 2023

This is the official code for MM2023 paper: *Speech-Driven 3D Face Animation with Composite and Regional Facial Movements.*

Given a template 3D face, driven 3D face sequence, and driven speech audio, we synthesize 3D face sequence synchronized with the speech audio, and modulated by speech-independent factors of the driven 3D face sequence.

### Installation
------
We train and test based on Python3.8 and Pytorch. To install the dependencies run:

```
pip install -r requirements.txt
```
Additionallly, you need to install pytorch3d following these [instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).

### Dataset preparation
------
We provide the processed voca data in `voca.zip`. Please unzip the zip file in the root folder. The zip file can be downloaded in the following link:

[Download](https://pan.baidu.com/s/1IEUbIOfFXFthnQE5rp4YgA)

Extraction code: fgqi


### Training
------
```
bash train_meshtalk.sh
```

We train the backbone with a two-stage manner. In the first step, we freeze the HuBERT model and train the ResNet1D. In the second step, we simultaneously fine-tune all of the models.

- Training data structure (Taking the VOCASET as example)
```
├── audio   # training audio with wav format
│  ├──FaceTalk_170725_00137_TA  # user id    
│  │  ├──sentence01.wav
│  │  ├──...
│  ├──...
├── eyelid_weight_mask.npy
├── face_axis_mean.npy
├── face_axis_std.npy
├── mouth_weight_mask.npy
├── templates
│  ├──FaceTalk_170725_00137_TA.ply  # the neutral face of specific id
│  ├──...
├── upper_weight_mask.npy
└── verts_data.pkl  # the pickle obj of all sequences data

*_weight_mask.npy is 0/1 weight mask of facial regions

face_axis_mean.npy(size 3) and face_axis_std.npy(size 1) are overall mean/std of the whole dataset, which are used for normalization.
```


### Inference
------
```
bash test_meshtalk.sh 
```

The pretrained VOCASET, meshtalk dataset, and BIWI datset models can be found in this link:

[Download](https://pan.baidu.com/s/1h2kVQ24xxo5m6wdBY1VUdQ)

Extraction code: tmi7

### Citation
------

```
@inproceedings{wu2023audio2face,
  title={Speech-Driven 3D Face Animation with Composite and Regional Facial Movements},
  author={Wu, Haozhe and Zhou, Songtao, and Jia, Jia and Xing, Junliang and Wen, Qi and Wen, Xiang},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  year={2023}
}
```