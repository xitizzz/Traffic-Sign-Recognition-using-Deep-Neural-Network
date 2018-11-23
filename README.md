# Traffic-Sign-Recognition-using-Deep-Neural-Network
An implementation of CNN using Keras to recognize GTSRB traffic signs.

## Introduction
This project compares the performance of two popular CNN architectures on traffic sign recognition task.

## Structure
The network can be trained and tested by running [main.py](/main.py). The training hyperparameters like learning rate, batch size, momentum are also located in [main.py](/main.py). Upon execution it asks to select one of the three available variants.

[preprocess.py](/preprocess.py) contains the code for image preprocessing. [model.py](/model.py) contains computational graph for all three variants of the model. [setting.py](/settings.py) stores the model setting, however it's not suppose to be modified as the hyperparameter setting are fetched from [main.py](/main.py) at runtime.

The files with prefix train, trains a specific model. There are separate files for some models, because there is enough variation in training code to have a separate file. This in not ideal, but it makes each of them more readable.

The files with prefix test performs the testing. [test_baseline.py](/test_baseline.py) takes trained model as argument and performs the testing on predefined test data. [test_loadmodel.py](/test_loadmodel.py) can load model from file before testing.

## Data
The data comes from [German Traffic Sign Recognition Benchmark dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). The data contains over 50,000 images (>38,000 training, >12,000 testing) images of 43 different types of traffic signs. The data sits in directory [GTSRB](/GTSRB), however the actual data is not included. The dataset can be downloaded from [here](http://benchmark.ini.rub.de/Dataset/GTSRB_Online-Test-Images.zip).

## Models
For this project I use three different models.
1. Baseline model
2. VGG16 Model
3. Baseline model with batch normalization

### Baseline model
Baseline model consists of four blocks in total. Three convolution blocks each consisting two convolution layers, a maxpooling layer and a dropout layer. Last block consists a fully connected layer and an output layer.

### VGG16 Model
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is a well known network designed at Oxford University. It contains 5 convolution blocks and final block with two fully connected layers. It has 16 layers in total (hence the name). It was developed to classify objects in 1000 classes, here it is modified to classify in 43 classes.

### Baseline with Batch Normalization
This is same as baseline model, however a batch normalization layer is added after convolution layers.

## Results
The highest accuracy achieved is 97.74% using VGG16.

Check full project report [here](./Project Report.pdf)
