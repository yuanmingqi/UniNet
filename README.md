# UniNet
Towards More Accurate Iris Recognition Using Deeply Learned Spatially Corresponding Features

<div align='center'>
    <img src= 'https://github.com/Mingqi-Yuan/UniNet/blob/master/reference/1.png' width=800px>
</div>

# Project structure
This project is organized as follows:
* dataset: contains the training data, validation data and test data.
* reference: contains the reference materials.
* report: contains the LaTex files of the project report.
* snapshots: for saving model weights.
* static: contains the pretrained model (MaskNet) and so on.
* data.py: for constructing the data generator.
* eval.py: for calculating TAR, FAR, EER.
* loss.py: the implementation of the Extended Triplet Loss.
* mask.py: for predicting masks for all the images in dataset using the pretrained MaskNet.
* match.py: functions for the iris matching (e.g. Hanmming distance).
* model.py: model class of the UniNet.
* network.py: the implementation of the FeatNet and the MaskNet.
* train.py: training file.

# Packages required
The moudle below is required for the project:
* PyTorch
* Pandas
* NumPy
* PIL

# Dataset preparation
For conducting training, your dataset should be organized as follows:

```
Train.txt

.dataset/Train/person1_001 0 
.dataset/Train/person1_002 0
...
.dataset/Train/personN_001 N
...
```
