# Traffic-Sign-Recognition-using-Deep-Neural-Network
An implementation of CNN using Keras to recognize GTSRB traffic signs

## Models
For this project I use three different models.
1. Baseline model
2. VGG16 Model
3. Baseline model with batch normalization

### Baseline model
Baseline model consists of four blocks in total. Three convolution blocks each consisting two convolution layers, a maxpooling layer and a dropout layer.Last block consists a fully connected layer and an output layer.

### VGG16 Model
[VGG16](http://www.robots.ox.ac.uk/~vgg/research/very_deep/) is a well known network designed at Oxford University. It contains 5 convolution blocks and final block with two fully connected layers. It has 16 layers in total (hence the name). It was developed to classify objects in 1000 classes, here it is modify to classify in 39 classes.

### Baseline with Batch Normalization
This is same as baseline model, however a batch normalization layer added after convolution layers.

## Results
The highest accuracy achieved is 97.74% using VGG16. Highest accuracy achieved using baseline is 97.34%

Full project report can be found [here](Traffic-Sign-Recognition-using-Deep-Neural-Network/Report and Answers.pdf).

