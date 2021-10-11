# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/modeling//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Project of Analytics Service Development Course

# ### Project Title: The effect of data augmentation in classifying weapons using Convolutional Neural Network
# ##### Presented by: Rabbil Bhuiyan
# ##### Presented on 29.05.2020

Image(filename= "gun_rifle_other_image.jpg", width=300, height=200)

# ## Introduction

# There is an increasing demand for automated surveillance services for numerous security reasons. The main aim of automated surveillance is to alert the CCTV (Closed Circuit Television) operator when there happens a dangerous situation (by providing certain mesaures like proximity alarm, lighting, intruder detection systems). Such dangerous situation of carrying a weapon like handgun, rifle ot other with the intention of harming people in the public place, can be identified using machine learning classification algorithms in the field of computer vision. Thus, the classification of weapons is important for the CCTV operational requirements for a wider security and safety aspects. This project proposed machine learning algorithms using artificial neural network for classification of weapons  either handgun, rifle or other. The objectives of this project to be achieved are- 
#
# - To design a multi-class classifier to detect three types of weapons : handgun, rife or other
# - To identify the best augmentaion parameter in improving the multi-class classifer or CNN model
#
# To do this, the project will utilize an artificial neural network based classifier (for details see the method section).

# #### Importing packages

# +
# Import necessary libraies
import numpy as np
import pandas as pd
import os
import cv2

import keras
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import confusion_matrix

import itertools
from IPython.display import Image
import matplotlib.pyplot as plt
# %matplotlib inline
# -

# #### Reading data into the memory

# +
import os
gun_files = os.listdir('C:/Users/Rabbil/Documents/New_folder_weapons_dataset/handgun')
rifle_files = os.listdir('C:/Users/Rabbil/Documents/New_folder_weapons_dataset/rifle')
other_files = os.listdir('C:/Users/Rabbil/Documents/New_folder_weapons_dataset/other')

print('Total images of gun: ', len(gun_files))
print('Total images of rifle: ', len(rifle_files))
print('Total images of other: ', len(other_files))
# -

# #### Organization of data 

# +
TRAIN_DATADIR= "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/train_data"
TEST_DATADIR= "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/test_data"


CATEGORIES= ["handgun", "rifle", "other"]

for category in CATEGORIES:
    path=os.path.join(TRAIN_DATADIR, category) # path to handgun, rifle and other dir
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array, cmap="gray")
        #plt.imshow(numpy.real(rink))
        #plt.imshow(im.asnumpy())

        plt.show()
        break
    break
# -

print(img_array)

print(img_array.shape)

# +
# normalizing the images as images are in different shape e.g landscape or portrait
IMG_SIZE=60

new_array= cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap='gray')
plt.show()
# -

print(new_array.shape)

# +
# creating training data set (#Build the dataset!)
training_data=[]

def create_training_data():
    #iterate over the dataset
    for category in CATEGORIES:
        path=os.path.join(TRAIN_DATADIR, category) # path to handgun, rifle and other dir (#path to test or train)
        class_num= CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array= cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
            
create_training_data()
# -

print(len(training_data))

# +
# Testing data set

# +
CATEGORIES= ["handgun", "rifle", "other"]

for category in CATEGORIES:
    path=os.path.join(TEST_DATADIR, category) # path to handgun, rifle and other dir
    for img in os.listdir(path):
        img_array_test=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array_test, cmap="gray")
        #plt.imshow(numpy.real(rink))
        #plt.imshow(im.asnumpy())

        plt.show()
        break
    break
# -

print(img_array_test)

print(img_array_test.shape)

# +
# normalizing the images as images are in different shape e.g landscape or portrait
IMG_SIZE=60

new_array_test= cv2.resize(img_array_test, (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array_test, cmap='gray')
plt.show()
# -

print(new_array_test.shape)

# +
# Building the testing dataset

# +
# creating testdata set (#Build the dataset!)
test_data=[]

def create_test_data():
    #iterate over the dataset
    for category in CATEGORIES:
        path=os.path.join(TEST_DATADIR, category) # path to handgun, rifle and other dir (#path to test or train)
        class_num= CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array_test=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array_test= cv2.resize(img_array_test, (IMG_SIZE,IMG_SIZE))
                test_data.append([new_array_test, class_num])
            except Exception as e:
                pass
            
create_test_data()
# -

print(len(test_data))

# +
# Lenght of total data both training and test data
# -

len(training_data), len(test_data)

# shuffle the dataset to avoid everything is gun or rifle and so on
import random
random.shuffle(training_data)

#Let's check that we shuffled properly
for sample in training_data[:10]:
    print(sample[1])

# #### Assigining features and label variables for training dataset 

# assigning features and label variables
X= [] #in general, capital X is feature set
y= [] #lowercase y is label

# +
for features, label in training_data:
    X.append(features)
    y.append(label)

#Currently, cannot pass lists to the neural network
#Need to convert X to a numpy array
X = np.array(X).reshape(-1,#how many features?
                        IMG_SIZE, IMG_SIZE, 1)#dims of data and 1indicates grayscale
y = np.array(y)
# -

X.shape, y.shape

# +
# We can see that our image is a tensor of rank 4,
# or we can say that our data have 4 dimensional array 
#with dimensions 4000 x 150 x 150 x 3 which correspond to the batch size, height, width and channels respectively.
# -

print(y.shape)

#Recode labels using to_categorical to get the correct shape of inputs:
from keras.utils import to_categorical
y = to_categorical(y)

print(y)

print(y.shape)
# So we see 26542 observations, 3 labels as 'one hot'

# #### Splitting the training dataset into train and validation set

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 2)
#(train_x.shape, train_y.shape), (val_x.shape, val_y.shape)
print("Shape of train images is:", X_train.shape)
print("Shape of validation images is:", X_val.shape)
print("Shape of labels is:", y_train.shape)
print("Shape of labels is:", y_val.shape)

# +
# Lenght of train and validation set
n_train=len(X_train)
n_val= len(X_val)

n_train, n_val

# +
# Defining the sub set of sample 
n_train_sample=len(X_train[0:5000])
n_val_sample = len(X_val[0:1000])

n_train_sample, n_val_sample

# +
# lets create our ImageDataGenerator object for training and validation set
# -

train_datagen= ImageDataGenerator(rescale=1./255) # scale the image between 0 and 1
val_datagen = ImageDataGenerator(rescale=1./255)

'''Rescaling normalizes the image pixel values to have zero mean and standard deviation of 1. 
It helps your model to generally learn and update its parameters efficiently.'''

# +
# Now that we have the ImageDataGenerator objects, 
#let's create python generators from them by passing our train and validation set.

# Create the image generators
batch_size = 20
epochs = 30
train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size= batch_size)
# -

# ### Methods (Convolutional Neural Network)

# Classification of image as handgun, rifle or other can be done with the help of artificial neural networks trained through supervised or unsupervised learning algorithms. In this project, back-propagation algorithm which is a supervised learning algorithm is chosen for its simplicity. Frist, we build a simple neural network with only two hidden layers. This was used as baseline model.
#
# For the baseline model we create a sequential model. Sequential model means that all the layers of the model will be arranged in sequence. Here we create the first layer by calling the add() function on the model and pass a Conv2D layer of 32 channel of 3x3 kernel. This first layer is called the input layer. We then add flatten layer. The last layer is output layer and has an output size of 3 and a different activation function called softmax.
#
# Then we tuned our base model by adding more layers in order to improve the accuracy and reduce the overfitting as well. Here we add a MaxPool2D layer. Its function is to reduce the spatial size of the incoming features and therefore helps reduce the number of parameters and computation in the network, thereby helping to reduce overfitting. 
#
# We then repeat the earlier layer with different filters as 64 (channel as 64) and also MaxPooling2D. 
# Then we add a Flatten layer. A conv2D layers extract and learn spatial features which are then passed to a dense
# layer after it has been flattened. This is the work of the flatten layer. 
# We also add a Dropout layer with value 0.5. Dropout randomly drops some layers in a neural networks and then learns with the reduced network. 
#
# The last layer is output layer with a size of 3 and a different activation function called softmax. After the creation of softmax layer the model is finally prepared. 
#
# Now we need to compile the model. Here we will be using Adam optimiser to reach to the global minima while training out model. We will also specify the learning rate of the optimiser, here in this case it is set at 0.0001. Now the final model is ready. We are using model.fit_generator as we are using ImageDataGenerator to pass data to the model. We will pass train and validation data to fit_generator. 
#

# #### Building the baseline model

# +
# Conv2D : two dimensional convolutional model
# 32: output for next layer
#(3,3): convolutional window size

model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape= X.shape[1:]),
    Flatten(),
    Dense(3, activation='softmax'),
    ])
# -

model.summary()

# #### Compile the model

# compiling the model
model.compile(optimizer=Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# #### Train the simple CNN model

# +
#Use the fit_generator method of the ImageDataGenerator class to train the network
history = model.fit_generator(train_generator, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history.history['val_loss'][-1], 
    history.history['val_accuracy'][-1]))

# +
# Saving model
# -

model.save('BaselineCnnModel.h5')

# #### Vizualizing training and validation accuracy and loss

# +
#get the details from the history object
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range,  loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# -

# -  We see that training accuracy and validation accuracy are off by large margin 
# -  The model has achieved only around 63% accuracy on the validation set.
# -  The difference in accuracy between training and validation is noticeable—a sign of overfitting.
# -  It means that the model will have a difficult time generalizing on a new dataset due to overfitting.
# -  One way to overcome overfitting is data augmentation.

# #### Fine-tunning of the model

# +
model_tuned=Sequential()
# Conv2D : two dimensional convulational model
# 64: input for next layer
#(3,3): convulational window size


model_tuned.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu", input_shape= X.shape[1:]))
model_tuned.add(Conv2D(filters=32,kernel_size=(3,3), activation="relu"))
model_tuned.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model_tuned.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model_tuned.add(Conv2D(filters=64,kernel_size=(3,3), activation="relu"))
model_tuned.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

model_tuned.add(Flatten())
model_tuned.add(Dense(64))
model_tuned.add(Activation('relu'))
model_tuned.add(Dropout(0.5))
model_tuned.add(Dense(3, activation='softmax'))
# -

model_tuned.summary()

# +
# Now train this modified model on our images of handguns and rifles
# -

# # compiling the model as before
model_tuned.compile(optimizer=Adam(lr=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# +
# Now train the model using model.fit_generator() (same as original model)

# +
history_tuned = model_tuned.fit_generator(train_generator, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned.history['val_loss'][-1], 
    history_tuned.history['val_accuracy'][-1]))

# +
# saving model
# -

model_tuned.save('tunedCNNmodel.h5')

# +
#### <font color= #a569bd >Vizualizing training and validation accuracy and loss of tunned model </font>

# +
#get the details from the history object
acc = history_tuned.history['accuracy']
val_acc = history_tuned.history['val_accuracy']

loss=history_tuned.history['loss']
val_loss=history_tuned.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range,  loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
# -

# - It look likes that the model is not overfitting as the train and validation accuracy are pretty close and following each other.
# - We can also notice that the accuracy keeps increasing as the epoch increases, 
# - It gives us the intuition that increasing the epoch size will likely give us a higher accuracy.
# - The loss curves are still overfitting as both train and validation loss are not close to each other as accuracy curve, but the loss will likely go lower if we increase the epoch size.

# #### Testing the model

# +
# shuffling of data
random.shuffle(test_data)

#Let's check that we shuffled properly
for sample in test_data[:10]:
    print(sample[1])

# +
# assigning features variables for test data set and we donot need the y label here
X_test= []# we put small x here 
y_test = []

for features, label in test_data:
    X_test.append(features)
    y_test.append(label)
    

#Currently, cannot pass lists to the neural network
#Need to convert X to a numpy array
X_test = np.array(X_test).reshape(-1,#how many features?
                        IMG_SIZE,IMG_SIZE, 1)#dims of data and 1indicates grayscale

y_test = np.array(y_test)
y_test = to_categorical(y_test)
# -

X_test.shape, y_test.shape

n_test=len(X_test)
n_test

n_test_sample=len(X_test[0:1000])
n_test_sample


# Function of plotting images
def plot_images(imgs=X_test, labels=y_test, rows=1, figsize=(20,8), fontsize=14):
    figure = plt.figure(figsize=figsize)
    cols = max(1,len(imgs) // rows-1)
    labels_present = False
    # checking if labels is a numpy array
    if type(labels).__module__ == np.__name__:
        labels_present=labels.any()
    elif labels:
        labels_present=True
    for i in range(len(imgs)):
        subplot = figure.add_subplot(rows, cols+1, i+1)
        # axis off, but leave a bounding box
        plt.tick_params(
            axis='both',
            which='both',
            bottom='off',
            top='off',
            left='off',
            right='off',
            labelbottom='off',
            labelleft='off')
        # plot labels if present
        if labels_present:
            subplot.set_title(labels[i], fontsize=fontsize)
        plt.imshow(imgs[i][:,:,0], cmap='viridis')
        
    plt.show()


plot_images(X_test[0:10], rows=1, figsize=(16,12))

test_labels = y_test[:,0]
test_labels[:10]

test_datagen = ImageDataGenerator(rescale = 1./255)
test_generator = test_datagen.flow(X_test, y_test, batch_size=batch_size)

# predicting on test data
Predictions = model_tuned.predict_generator(generator=test_generator, 
                                      steps= None, 
                                      verbose=0)

Predictions = np.round(Predictions[:,0])
Predictions

y_true = test_labels
y_pred= Predictions
cm = confusion_matrix(y_true, y_pred)


def plot_confusion_matrix(cm, CATEGORIES,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(CATEGORIES))
    plt.xticks(tick_marks, CATEGORIES, rotation=45)
    plt.yticks(tick_marks, CATEGORIES)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm_plot_labels = ['handgun','rifle','other']
plot_confusion_matrix(cm=cm, CATEGORIES=cm_plot_labels, title='Confusion Matrix')

# -  Blue cells are correctly predicted while white cells are incorrectly predicted
# -  The model correctly predicted that an image was a handgun 2910 times when it actually was a handgun and incorrectly predicted that an image was a rifle 1512 times when it was not a handgun and vice versa
# -  Poor predictions could be due to multi level classfication
# -  The 'other' image is problematic during runing the predction model which indicates faulty images

# ### Augmentation technique of images

# Image data augmentation is a technique that can be used to artificially expand the size of a training dataset by creating modified versions of images in the dataset. Training deep learning neural network models on more data can result in more powerful models, and the augmentation techniques can create variations of the images that can improve the ability of the fit models to generalize. Thus image augmentation techniques also reduce the overfitting of the model. The Keras deep learning library provides the ability to use data augmentation automatically when training a model. This is achieved by using the ImageDataGenerator class.
#
# Image augmentation parameters that are generally used to increase the data sample count are zoom, shear, rotation, preprocessing_function and so on. We will focus on 9 differernt types of data augmentation techniques for image data; specifically:
# -  Image rotations via the rotation_range argument
# -  Image zoom via the zoom_range argument
# -  Image shear via shear_range argument
# -  Image shifts via the width_shift_range  
# -  Image shifts via height_shift_range argument
# -  Image channel via channel_shift argument
# -  Image flips via the horizontal_flip and 
# -  Image flips via vertical_flip arguments.
# -  Image fill via fill_mode argument
#

# +
# Create train datagenerator for different augmentation features

train_datagen_rr=ImageDataGenerator(rescale=1./255, rotation_range=40)
train_datagen_sr=ImageDataGenerator(rescale=1./255, shear_range=0.2)
train_datagen_zr=ImageDataGenerator(rescale=1./255, zoom_range=0.5)
train_datagen_wsr=ImageDataGenerator(rescale=1./255, width_shift_range=0.2)
train_datagen_hsr=ImageDataGenerator(rescale=1./255, height_shift_range=0.2)
train_datagen_csr=ImageDataGenerator(rescale=1./255, channel_shift_range=10)
train_datagen_hf=ImageDataGenerator(rescale=1./255, horizontal_flip=True)
train_datagen_vf=ImageDataGenerator(rescale=1./255, vertical_flip=False)
train_datagen_fm=ImageDataGenerator(rescale=1./255, fill_mode='constant')


#Create validation data generator
val_datagen = ImageDataGenerator(rescale=1./255)

# +
# Now that we have the ImageDataGenerator objects, 
#let's create python generators from them by passing our train and validation set.

# Create the image generators
batch_size = 20
epochs = 30

train_generator_rr = train_datagen_rr.flow(X_train, y_train, batch_size=batch_size)
train_generator_sr = train_datagen_sr.flow(X_train, y_train, batch_size=batch_size)
train_generator_zr = train_datagen_zr.flow(X_train, y_train, batch_size=batch_size)
train_generator_wsr = train_datagen_wsr.flow(X_train, y_train, batch_size=batch_size)
train_generator_hsr = train_datagen_hsr.flow(X_train, y_train, batch_size=batch_size)
train_generator_csr = train_datagen_csr.flow(X_train, y_train, batch_size=batch_size)
train_generator_hf = train_datagen_hf.flow(X_train, y_train, batch_size=batch_size)
train_generator_vf = train_datagen_vf.flow(X_train, y_train, batch_size=batch_size)
train_generator_fm = train_datagen_fm.flow(X_train, y_train, batch_size=batch_size)


val_generator = val_datagen.flow(X_val, y_val, batch_size= batch_size)
# -

# #### Training the model for each augmentation parameter 

# ##### Rotation range

# +
# Now train the model using model.fit_generator()in the tunned model
history_tuned_rr = model_tuned.fit_generator(train_generator_rr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_rr.history['val_loss'][-1], 
    history_tuned_rr.history['val_accuracy'][-1]))
# -

# ##### Shear range

# +
history_tuned_sr = model_tuned.fit_generator(train_generator_sr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_sr.history['val_loss'][-1], 
    history_tuned_sr.history['val_accuracy'][-1]))
# -

# ##### Zoom range 

# +
history_tuned_zr = model_tuned.fit_generator(train_generator_zr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_zr.history['val_loss'][-1], 
    history_tuned_zr.history['val_accuracy'][-1]))
# -

# ##### Width_shift_range 

# +
history_tuned_wsr = model_tuned.fit_generator(train_generator_wsr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_wsr.history['val_loss'][-1], 
    history_tuned_wsr.history['val_accuracy'][-1]))
# -

# ##### Height_shift_range

# +
history_tuned_hsr = model_tuned.fit_generator(train_generator_hsr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_hsr.history['val_loss'][-1], 
    history_tuned_hsr.history['val_accuracy'][-1]))
# -

# ##### Channel_shift_range

# +
history_tuned_csr = model_tuned.fit_generator(train_generator_csr, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_csr.history['val_loss'][-1], 
    history_tuned_csr.history['val_accuracy'][-1]))
# -

# ##### Horizontal_flip 

# +
history_tuned_hf = model_tuned.fit_generator(train_generator_hf, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_hf.history['val_loss'][-1], 
    history_tuned_hf.history['val_accuracy'][-1]))
# -

# ##### Vertical_flip 

# +
history_tuned_vf = model_tuned.fit_generator(train_generator_vf, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_vf.history['val_loss'][-1], 
    history_tuned_vf.history['val_accuracy'][-1]))
# -

# ##### Fill_mode

# +
history_tuned_fm = model_tuned.fit_generator(train_generator_fm, 
                    steps_per_epoch=n_train_sample // batch_size,
                    epochs = epochs,
                    validation_data=val_generator, 
                    validation_steps= n_val_sample // batch_size, 
                    )

print("Final loss: {0:.4f}, final accuracy: {1:.4f}".format(
    history_tuned_fm.history['val_loss'][-1], 
    history_tuned_fm.history['val_accuracy'][-1]))
# -
# #### Plotting the accuracy and loss

# +
figure = plt.figure(figsize=(20,9))
 
subplot = figure.add_subplot(1, 2, 1)

plt.plot(history_tuned.history['accuracy'], color='#d35400', label='Original data')
plt.plot(history_tuned_rr.history['accuracy'], color='#1b2631', label='rotation data')
plt.plot(history_tuned_sr.history['accuracy'], color='#283747', label='shear sample')
plt.plot(history_tuned_zr.history['accuracy'], color='#2e4053', label='zoom samples')
plt.plot(history_tuned_wsr.history['accuracy'], color='#34495e', label='width_shift samples')
plt.plot(history_tuned_hsr.history['accuracy'], color='#5d6d7e', label='height_shift samples')
plt.plot(history_tuned_csr.history['accuracy'], color='#85929e', label='channel_shift samples')
plt.plot(history_tuned_hf.history['accuracy'], color='#aeb6bf', label='horizontal_flip samples')
plt.plot(history_tuned_vf.history['accuracy'], color='#d6dbdf', label='vertical_flip samples')
plt.plot(history_tuned_fm.history['accuracy'], color='#ebedef', label='fill_mode samples')
plt.ylim(0.2,1.09)

tick_marks = np.arange(epochs)
plt.xticks(tick_marks, range(epochs), rotation=45)
plt.title("Training accuracy", fontsize=13, 
             fontweight=0, color='black', style='italic', y=1.02)
plt.legend(loc='lower right')


subplot = figure.add_subplot(1, 2, 2)
plt.plot(history_tuned.history['val_accuracy'], color='#d35400', label='Original data')
plt.plot(history_tuned_rr.history['val_accuracy'], color='#1b2631', label='rotation data')
plt.plot(history_tuned_sr.history['val_accuracy'], color='#283747', label='shear sample')
plt.plot(history_tuned_zr.history['val_accuracy'], color='#2e4053', label='zoom samples')
plt.plot(history_tuned_wsr.history['val_accuracy'], color='#34495e', label='width_shift samples')
plt.plot(history_tuned_hsr.history['val_accuracy'], color='#5d6d7e', label='height_shift samples')
plt.plot(history_tuned_csr.history['val_accuracy'], color='#85929e', label='channel_shift samples')
plt.plot(history_tuned_hf.history['val_accuracy'], color='#aeb6bf', label='horizontal_flip samples')
plt.plot(history_tuned_vf.history['val_accuracy'], color='#d6dbdf', label='vertical_flip samples')
plt.plot(history_tuned_fm.history['val_accuracy'], color='#ebedef', label='fill_mode samples')
plt.ylim(0.2,1.09)
 
tick_marks = np.arange(epochs)
plt.xticks(tick_marks, range(epochs), rotation=45)
plt.title("Validation accuracy", fontsize=13, 
             fontweight=0, color='black', style='italic', y=1.02)
plt.legend(loc='lower right')
 
 
 
plt.show()
# -

# #### Table for comparasion of all the accuracy and loss

# +
# Assigning variables to each model's accuracy and loss

Original_ac = history_tuned.history['accuracy'][-1]
rr_ac = history_tuned_rr.history['accuracy'][-1]
sr_ac = history_tuned_sr.history['accuracy'][-1]
zr_ac = history_tuned_zr.history['accuracy'][-1]
wsr_ac =history_tuned_wsr.history['accuracy'][-1]
hsr_ac =history_tuned_hsr.history['accuracy'][-1]
csr_ac =history_tuned_csr.history['accuracy'][-1]
hf_ac = history_tuned_hf.history['accuracy'][-1]
vf_ac = history_tuned_vf.history['accuracy'][-1]
fm_ac = history_tuned_fm.history['accuracy'][-1]

Original_val_ac = history_tuned.history['val_accuracy'][-1]
rr_val_ac = history_tuned_rr.history['val_accuracy'][-1]
sr_val_ac = history_tuned_sr.history['val_accuracy'][-1]
zr_val_ac = history_tuned_zr.history['val_accuracy'][-1]
wsr_val_ac =history_tuned_wsr.history['val_accuracy'][-1]
hsr_val_ac =history_tuned_hsr.history['val_accuracy'][-1]
csr_val_ac =history_tuned_csr.history['val_accuracy'][-1]
hf_val_ac = history_tuned_hf.history['val_accuracy'][-1]
vf_val_ac = history_tuned_vf.history['val_accuracy'][-1]
fm_val_ac = history_tuned_fm.history['val_accuracy'][-1]

Original_loss = history_tuned.history['loss'][-1]
rr_loss = history_tuned_rr.history['loss'][-1]
sr_loss = history_tuned_sr.history['loss'][-1]
zr_loss = history_tuned_zr.history['loss'][-1]
wsr_loss =history_tuned_wsr.history['loss'][-1]
hsr_loss =history_tuned_hsr.history['loss'][-1]
csr_loss =history_tuned_csr.history['loss'][-1]
hf_loss = history_tuned_hf.history['loss'][-1]
vf_loss = history_tuned_vf.history['loss'][-1]
fm_loss = history_tuned_fm.history['loss'][-1]

Original_val_loss = history_tuned.history['val_loss'][-1]
rr_val_loss = history_tuned_rr.history['val_loss'][-1]
sr_val_loss = history_tuned_sr.history['val_loss'][-1]
zr_val_loss = history_tuned_zr.history['val_loss'][-1]
wsr_val_loss =history_tuned_wsr.history['val_loss'][-1]
hsr_val_loss =history_tuned_hsr.history['val_loss'][-1]
csr_val_loss =history_tuned_csr.history['val_loss'][-1]
hf_val_loss = history_tuned_hf.history['val_loss'][-1]
vf_val_loss = history_tuned_vf.history['val_loss'][-1]
fm_val_loss = history_tuned_fm.history['val_loss'][-1]


# Initialise data to lists 
Data=[{'Accuracy':Original_ac, 'Val_accuracy':Original_val_ac, 'Loss':Original_loss, 'Val_loss':Original_val_loss},
      {'Accuracy':rr_ac, 'Val_accuracy':rr_val_ac, 'Loss':rr_loss, 'Val_loss':rr_val_loss},
      {'Accuracy':sr_ac, 'Val_accuracy':sr_val_ac, 'Loss':sr_loss, 'Val_loss':sr_val_loss},
      {'Accuracy':zr_ac, 'Val_accuracy':zr_val_ac, 'Loss':zr_loss, 'Val_loss':zr_val_loss},
      {'Accuracy':wsr_ac, 'Val_accuracy':wsr_val_ac, 'Loss':wsr_loss, 'Val_loss':wsr_val_loss},
      {'Accuracy':hsr_ac, 'Val_accuracy':hsr_val_ac, 'Loss':hsr_loss, 'Val_loss':hsr_val_loss},
      {'Accuracy':csr_ac, 'Val_accuracy':csr_val_ac, 'Loss':csr_loss, 'Val_loss':csr_val_loss},
      {'Accuracy':hf_ac, 'Val_accuracy':hf_val_ac, 'Loss':hf_loss, 'Val_loss':hf_val_loss},
      {'Accuracy':vf_ac, 'Val_accuracy':vf_val_ac, 'Loss':vf_loss, 'Val_loss':vf_val_loss},
      {'Accuracy':fm_ac, 'Val_accuracy':fm_val_ac, 'Loss':fm_loss, 'Val_loss':fm_val_loss}
       ] 
  
# Creates DataFrame. 
df = pd.DataFrame(Data, index =['Original', 'Rotation_range', 'Shear_range', 'Zoom_range',
                               'Width_shift_range', 'Height_shift_range', 'Channel_shift_range',
                               'Horizontal_flip', 'Vertical_flip','Fill_mode']) 
   
# Print the Data 
df

# +
# val_loss starts decreasing, val_acc starts increasing(Correct, means model build is learning and working fine)
# val_loss starts increasing, val_acc also increases.(could be case of overfitting or diverse probability values in cases softmax is used in output layer)
# val_loss starts increasing, val_acc starts decreasing(means model is cramming values not learning)
# -

# ### Conclusions

# After tuning, by adding more layers, we have improved the baseline model which was based on just two layers. We still observed that both training and validation accuracy is not fairly good for value proposition of weapon dataset. This poor accuracy is might be due to the multi-class weapon classification, which we found dispropotionate number of images in ‘other’ image class. However, few images in 'other' class also seems to be out of the boundary in case of resized images. We assume that the accuracy will fairly increase while considering binary classification between handgun and rifle. However, further tuning or tailoring of the CNN modelling parameter might improve the multi-class weapon classification in terms of both accuracy and loss. As we observed the with the increases of epochs the accuracy improves, applying large number of epoch could improve the model.
#
# The augmentation techniques we applied gave us promising results. We observed that 'fill_mode' parameter improved the model by minimizing the loss and maximizing the accuracy significantly. However, Height_shift_range parameter looks promimsing in improving the model accuracy as the deviations among the accuracy levels are least. So, combination of augmentation parameters might improve the model accuracy further.
#

# ### Business Value propositions

# ####  Data-sourced propostions
# - Binary classfication vs multi-class classfication 
# - Very small pixel image particularly in 'other' image data
# - Dispropotionate images across the classes (Handguns : 9104; Rifles : 14244; Other : 3194)
#
# ####  Analytics-driven propostions
# - Tailoring image size (resizing of image)
#   - Image size vs Total params in CNN model
# - Further tuning of CNN model
# - Increasing the number of epochs
# - Combinations of augmentation parameters
#
# ####  Take-way propositions based on this project study
# - Best models minimize loss and maximize accuracy
# - Augmentation of images affect the model accuracy (not all parameter !)
# - Horizontal flip is most effective (as both loss and val_loss are lower and accuracy and val_accuracy are higher)
# - Fill mode parameter : highest accuracy (0.86) and lowest loss (0.3)
# - Fill mode alone improved the model accuracy from 0.66 to 0.86 (<font color=orange>Awesome !</font>)

# ## THANK YOU !

Image(filename= "question_image.jpg", width=900, height=400)


