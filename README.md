# Cartroids

Carotis were created as a software-research semester project at the Faculty
of Electrical engineering at CTU in Prague. In this work we explore dataset
of ultrasound carotid images. The project itself is divided into three main 
parts:

* Categorization
* Localization
* Segmentation
  
For each of these segments one can find the codebase defines in particularly 
named sub-folder of the folder carotids and running script. The reports to each 
of this tasks are located in the folder reports.

## Examples

### Categorization

The sample use the project for training the models described in the particular 
report can be found in **categorization_train.py** and later usage of a trained
model in **categorization_use.py**

### Localization

For localization, the training of the compared model is shown in file
**localization_train.py** and later usage of the best selected model in
 **localization_use.py**.

## Implementation details

The work is implemented primarily in **Python 3.7** and **PyTorch 1.6**. The 
PyTorch version needs to be met to load the models properly.

Created by Martin Kostelansky in 2020.
