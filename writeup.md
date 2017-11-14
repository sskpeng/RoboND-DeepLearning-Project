# Project: Deep Learning - Follow Me
### Introduction
In this Follw Me project, we are simulating a quadcopter equiped with a camera flying in a virtual city with many pedestains. The main task is to implement the deep learning techniques so that the quadcopter is able to detect and follow a specific person walking in the city.
![alt text][image0] 
---
[image0]: ./docs/misc/sim_screenshot.png
[image1]: ./writeup_pic/pic01_fcn.png
[image2]: ./writeup_pic/pic02_encoder.png
[image3]: ./writeup_pic/pic03_decoder.png
[image4]: ./writeup_pic/pic04_fcn_model.png
[image5]: ./writeup_pic/pic05_training_curves.png
[image6]: ./writeup_pic/pic06_result1.png
[image7]: ./writeup_pic/pic07_result2.png
[image8]: ./writeup_pic/pic08_result3.png
[image9]: ./writeup_pic/pic09_result_scores.png

## Neural Network Architecture
There are two major goals for the learning network to achieve the object recognition:

1. To identify if the person is in the image (carmera's field of view)
2. Find out where is the person in the image

The first task is a very common classification question, and **Convolutional Neural Network (Covnet or CNN)** is commenly used for this task. The advantage of covnets is, that the numbers of parameters can be largely reduced. As we know that it is not important to know in which corner of the image our hero is, to know the he is the hero. The weights are shared across all patches in an input layer.

However, for the second task, the spatial information of the image is for the locating the target so that the quadcopter can move accordingly. In this case, the **Fully Convolution Network (FCN)** is recommended. Fully Convolutional Network (FCN) preserves the spatial information in an image and enables us to get an understanding of the scene through semantic segmantation. 

A typical FCN is composited of the following three blocks:

1. Encoder
2. 1 by 1 convolution
3. Decoder

The **encoder blocks** extract features for the image. It is extracting more complex features from layer to layer. For example, the first layer might only look at lines or edges, next layer might look for circules, boxes, and the following layer might look for more complex geometries. 

**1 by 1 convolution** used on every pixel to transform feature maps to class-wise predictions. This helps in pixel-wise classification. In the conventional CNNs, we usually flatten the output into a fully connected layer into a 2D tensor, which results in the loss of spatial information, because no information about the location of the pixels is preserved. Hence, the 1x1 convolution is used to replace the fully connected layer and connect the encoder and decoder layer.

The **decoder blocks** upscale the output back to the size of the original image. The result is a segmentation information for every pixel in the image. So called skip connections is used to connect corresponding encoder layer and decoder layer (to match the layer size) to retrieves lost spatial information during Encoding. 

![alt text][image1] 

## Neural Network Parameters

The Hyperparameters of this FCN includes:

1. Learning Rate
2. Batch Size
3. Number of Epochs
4. Steps per Epoch
5. Number of Validation Steps

### Learning Rate
The learning rate is the amount of change applied to the weights in each iteration when training the neural network. It is an indicator how quickly the network can change its mind. The larger the rate, the faster the network approaches the optimum weights. However, applying too large rate will result in instability. Thus it is usually a low value.

### Batch Size
In order to reduce the amount of the training inputs in one run, the input is divided into small subsets called batches. The input is randomly shuffled an then put into the batches.

### Epochs
An epoch is a full iteration of a neural network. Performing this multiple times increases the network accuracy. This advantage will cease over time, there comes a point when the accuracy stops increasing. It is connected to the learning rate: smaller learning rates need more epochs to get a good accuracy. This makes sense as the changes are more subtle and therefore need more time to develop.

### Steps per Epoch
Steps per epoch is the number of training image batches which pass the network in 1 epoch. There is no need to put every image through the network in every epoch and not putting everything in everytime also helps with overfitting.

### Validation Steps
Validation steps is the number of validation image batches which pass the network in 1 epoch. This is the same as steps per epoch with validation images.

## FCN Implementation

FCN is implemeted in the provided notebook and the `encoder_block`, `decoder_block`, and the main `fcn_model` functions are shown as following:
![alt text][image2]
![alt text][image3]
![alt text][image4] 

## Experiments Results
The following hyperparameters are used to traing the network, and the corresponding test results are shown. Note that the run number shown on the table below is not the same as the training data set in the repository.

|                | run 1  | run 2  | run 3  | run 4  |
|:---------------|:------:|:------:| ------:| ------:|
|Learning rate   | 0.02   | 0.005  | 0.005  | 0.003  |
|Batch size      | 128    | 32     | 32     | 32     |
|Num of Epochs   | 5      | 10     | 50     | 100    |
|Steps per Epochs| 200    | 100    | 100    | 100    |
|Validation Steps| 50     | 50     | 50     | 50     |
|Workers         | 2      | 5      | 5      | 5      |
|Final IoU       | 0.5389 | 0.4650 | 0.5213 | 0.5576 |
|Final Score     | 0.3785 | 0.3255 | 0.3889 | 0.4028 |

The following validation results were generated in run 4.

Here is the training curves showing the loss in 100 epochs.
![alt text][image5]

Testing of following the target.
![alt text][image6]

Testing of patrol without target in the image.
![alt text][image7] 

Testing of patrol with target in the image.
![alt text][image8]

Final score and IoU calculation.
![alt text][image9]

### Improvement
The hyoerparameters above are able to achieve 0.4 to pass the target score. It is still very possible to improve the proformance by tuning the parameters. For example, increasing the num of epochs,  reducing the learning rate, and reducing the steps per epochs. It is also possible to gain better training scores by using better quality pictures with higher resolution.

### Changes needed for identifying new objects
The model trained in this exercise identifies humans in red shirt. To identify any new objects like car, cat, dog, new classes of data for those models will be needed. Even if multiple classes of models are provided for cars, animals, or other objects, and the models are pre-trained, the current network architecture might still not be able to process the computation simutaniously. More components or some massive cloud system is needed to achieve the novel classifier in this level. 