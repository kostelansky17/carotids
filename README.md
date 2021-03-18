# Carotids

The Project Carotids was created as a Master thesis project
 "Localization and segmentation of in-vivo ultrasound carotid artery images" at the Faculty
of Electrical engineering at CTU in Prague. In this work is explored dataset
of carotid ultrasound images. The project itself is divided into three main 
parts:

* Classification
* Localization
* Segmentation


## Examples

### Classification

A classification model was selected the
ResNet50 model and the training process
is described by **categorization_train.py**
Th model is able to predict the type
of the ultrasound image. The four categories
are longitudinal, transverse, Doppler, and conical. How to predict categories of sample images is described in **categorization_use.py**

### Localization

To localize the common carotid artery on an ultrasound image,
two Faster R-CNNs are trained in **localization_train.py**. The trained model can be run as is shown in the file **localization_use.py**.

### Segmentation

For segmentation, there are two example files as well.
The training script **segmentation_train.py**, 
creates two U-nets, each for segmenting different type
(longitudinal and transverse) of ultrasound carotid artery images. The possible later usage is 
shown in **segmentation_use.py**.

## Trained models

The best models for all tasks as defined in 
the thesis are available for download from  
[Google Drive](https://drive.google.com/drive/folders/1gRT2sJv0F5efB3eZsnWPdG_CpzvjUcYS?usp=sharing).
The simple use case with sample data can be found in this repository as well
(sample data are in folder data_samples).

## Implementation details, requirements & installation

The work is primarily implemented in **Python 3** and **PyTorch**. 
The recommended version of **Python** is **>=3.7**. To a standard conda
environment, you will need **PyTorch** (version **>=1.6** id needed to load the 
models properly) and **Torchvision** (version **>=0.9**).
The package carotids can be then installed with command:
```
pip install git+https://github.com/kostelansky17/carotids.git
```


Created by Martin Kostelansky (*martin.kostelansky@fel.cvut.cz*)  in 2020 and 2021.
