# Brain Tumor Classifier & Live Prediction in Browser.
Deep learning experiments for the Brain image classification, to distinguish between tumor contaning brain scan vs normal brain scan.

## Overview
For this project, I built a Binary-Classifier to identify normal MRI brain scans vs with tumor MRI brain scans. This project aim was to achieve above 90% top-1 accuracy.
I was able to get the model to predict the class of the scans with 97% accuracy with out using transfer learning. To get these results I build a non-complex Convolutional Neural Network (CNN) model. This created time efficiencies and solid results. <br/><br/>
I started this project with binary classification problem by making a model that was able to classify normal vs tumor brains. This experiment was done on a
simple CNN models, having no more then 3 Conv2D (32, 32, 64 filters respectively) hidden layers, MaxPooling layer, dense layer, and dropout layer. These models were able to achieve above 95% accuracy.But from the loss and accuracy curve between train data and validation data, it was clear that model was overfitting because the distance between the train and val graph lines were large. Due to less limited time I stoped the project and didn't regularize or fine-tuned the model.

## Code and Resources Used
- Python Version: 3.8
- Tensorflow Version: 2.7.0
- Tensorflow. JS
- Packages: pandas, numpy, sklearn, matplotlib, seaborn
- Editor:  Google Colab

## Dataset and EDA
The experiment data was downloaded from <a href="https://www.kaggle.com/ahmedhamada0/brain-tumor-detection">Kaggle Br35H</a>.
It contained total 3000+ images, the data was small as compared to the data needed for ANN training. The dataset contained 3 folder, 
'yes' -> with tumor, 'no' -> no tumor and 'pred'. So I used tensorflow data API to load and preprocess the data and split the dataset
using sklearn train_test_split into 80-20 train and val datasets. The 'pred' images were used in the testing process.
                                                        

|               | Experiment #1 | 
| ------------- |:-------------------:|
| Dataset source|  Kaggle| 
| Classes |  2 = (yes / no) Tumor| 
| Individual Class |	1500 (x 2) = 3,000|
| Train data| 2,388 Images|
| Test data | 612 Images|
|Normalization|Yes|
|Data Augmentation|	No|
|Data loading|	tf.data API|

some examples from the Train datasets:

![alt text](https://github.com/ozzmanmuhammad/Brain-Tumor-Classification/blob/main/images/tumor_examples.png "Train data examples")

## Model Building and Performance

The experiments was done on CNN architecture model, which was trainied 10 epochs (less epochs to avoid overfitting as it was trained only on 2,388 images)

|               | Experiment #1 | 
| ------------- |:-------------------:|
|Architecture|	CNN|
|Hidden layers|	3 Conv2D, 3 MaxPooling, Flatten, Dense, Dropout|
|Hidden Width|	(32+ 32+ 64+ 64), dropout = 0.5|
|Callbacks|	None|
|Epochs|	10|

<br/><br/>

## Analysis
Model achieved 97% accuracy which was not surprising as it was trained on less data and epochs, and it was overfitting. The overfitting was clear when I build the graph
between loss and accuracy curves ( training and validation ) because the perfect generalized model's training and validation curves are always to each other and follow similar pattern. As the below graph clearly shows the distance between curves, Nevertheless it was able to achieve the accuracy as of our target goal 90+% top-1 accuracy.
<img src="https://github.com/ozzmanmuhammad/Brain-Tumor-Classification/blob/main/images/accuracy.png" alt="Accuracy Curve"  width="500"/>


## Predictions on Custom Images
For prediction I used the 'pred' images which were unseen for the model. These images were also preprocessed to match there dimension, shape and scale. Out of 9 model predicted all correct with high probability scores.
<img src="https://github.com/ozzmanmuhammad/Brain-Tumor-Classification/blob/main/images/tumor_preds.png" alt="Custom Predictions" width="700"/>

## Live Predictions in Browser.
Model was download in Keras .h5 format and then was converted for tensorflow java script. Upload any MRI brain scan image and test the prediction. The predictions are not always right. To try it yourself please visit project page on my <a href="https://ozzmanmuhammad.github.io/project-BrainTumor.html" target="_blank">"Portfolio."</a>
