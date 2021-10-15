# Multi-class image classification using Convolutional Neural Networks (CNN)
This project is made for the Analytical Service Development course of Master of Engineering in Big Data Analytics at Arcada University of Applied Sciences, Helsinki. The aim of this work was to build a model that can classify whether an image is handgun, rifle or other using convolutional neural networks (CNN). The project also aimed to identify the best image augmentation parameter in improving the multi-class classifier or CNN model.

Dataset The project uses weapons data from the Department of Business Management and Analytics, Arcada University of Applied Science, provided by Stagezero Oy. This is the first work using these dataset. The dataset consisted of 26,542 images from 3 classes : handgun, rifle and other. We first started with organizing the data: splitting the data into two folders namely train and test folders. We have imported the necessary libraries for analysis. We applied Keras with a Tensorflow backend for the CNN model and the ImageDataGenerator class of Keras deep learning library for image data augmentation.

Modelling approach The approach was to first build a simple neural network with only two hidden layers. This was used as baseline model (sequential model of Keras library). The architecture of the baseline model was as a Conv2D layer of 32 channel of 3x3 kernel followed by flatten layer and output layer. We used softmax activation function as we have 3 classes in the dataset. In the tuned model we added MaxPool2D layer, Conv2D of 64 channel and Dropout layer.

We observed that tuning of sequential model parameter and image augmentation technique improve the accuracy from 0.66 to 0.86 for the multi-level classification of weapons.
