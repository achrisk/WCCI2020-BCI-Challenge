# Clinical BCI Challenge WCCI 2020: Team iBCI

**Update (24th July 2020):** We got 2nd position in the challenge.

This is the submission of team iBCI for the Clinical BCI Challenge organized by World Congress on Computational Intelligence (WCCI) 2020. The [dataset](https://github.com/5anirban9/Clinical-Brain-Computer-Interfaces-Challenge-WCCI-2020-Glasgow) provided consists of EEG data recorded on 10 hemiparetic stroke patients using a 12 channel device with a sampling rate of 512 Hz.

## Description of Methods used

Preprocessing includes band-passing each trial in the band 8-35 Hz and trimming each trial to keep last 3s (as per experiment protocol). Each trial is a 12x1536 matrix where 12 is the number of channels and 1536 is the number of time samples in 3s. We have used two categories of techniques for classifying the data:
1. Riemannian Geometry
2. Convolutional Neural Network (CNN)

### Riemannian Geometry

Covariance matrix is calculated for each trial and then these matrices are projected on the tangent space in the Riemannain Space using geometric mean as the reference point as described in "Riemannian geometry applied to BCI classification" by Barachant et. al. available [here](https://hal.archives-ouvertes.fr/hal-00602700/document). This transforms each trial into a 78-dimensional vector. Further various classifiers have been tested including Dense Neural Network (DNN), Support Vector Machine (SVM) and Linear Discriminant Analysis (LDA).

### Convolutional Neural Network (CNN)

We have used EEGNet described in "EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces" by Lawhern et. al. available [here](https://arxiv.org/abs/1611.08024).

### Within Subject Classification

Several methods were compared using 5-fold cross validation kappa scores. DNN with two layers: 16 hidden units and 1 sigmoid output unit on Riemannian features is used to produce final results.

### Cross Subject Classification

For cross subject classification, an ensemble of EEGNet and Riemannian Geometry with DNN and SVM is used. These three methods have been ensembled by taking a majority vote.

## Instructions to run code

Run the IPython notebook ```Classify.ipynb```.

Packages required:
- Pyriemann
- Tensorflow
- scikit-learn
- SciPy
- NumPy
