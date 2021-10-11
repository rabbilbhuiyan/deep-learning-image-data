# ---
# jupyter:
#   jupytext:
#     formats: ipynb,../scripts/preparation//py
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

# ### Creating train and test folder out of full dataset which is consisted of guns, rifles and others

import os
import numpy as np
import shutil

# ### Gun data

# +
# Creating Train / Test folders
root_dir = 'C:/Users/Rabbil/Documents/New_folder_weapons_dataset'
gunCls = '/handgun'
rifleCls = '/rifle'
otherCls = '/other'

os.makedirs(root_dir +'/train_data' + gunCls)
os.makedirs(root_dir +'/train_data' + rifleCls)
os.makedirs(root_dir +'/train_data' + otherCls)
#os.makedirs(root_dir +'/val' + gunCls)
#os.makedirs(root_dir +'/val' + rifleCls)
#os.makedirs(root_dir +'/val' + otherCls)
os.makedirs(root_dir +'/test_data' + gunCls)
os.makedirs(root_dir +'/test_data' + rifleCls)
os.makedirs(root_dir +'/test_data' + otherCls)


# Creating partitions of the data after shuffeling
currentCls = gunCls
src = 'C:/Users/Rabbil/Documents/New_folder_weapons_dataset'+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8)])

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
#print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/train_data"+currentCls)

#for name in val_FileNames:
    #shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/val"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/test_data"+currentCls)

# -

# ### Rifles data

# +
# Creating partitions of the data after shuffeling
currentCls = rifleCls
src = 'C:/Users/Rabbil/Documents/New_folder_weapons_dataset'+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8)])

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
#print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/train_data"+currentCls)

#for name in val_FileNames:
    #shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/val"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/test_data"+currentCls)

# -

# ### Other images

# +
# Creating partitions of the data after shuffeling
currentCls = otherCls
src = 'C:/Users/Rabbil/Documents/New_folder_weapons_dataset'+currentCls # Folder to copy images from

allFileNames = os.listdir(src)
np.random.shuffle(allFileNames)
train_FileNames, test_FileNames = np.split(np.array(allFileNames),[int(len(allFileNames)*0.8)])

train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
#val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames))
print('Training: ', len(train_FileNames))
#print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/train_data"+currentCls)

#for name in val_FileNames:
    #shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/val"+currentCls)

for name in test_FileNames:
    shutil.copy(name, "C:/Users/Rabbil/Documents/New_folder_weapons_dataset/test_data"+currentCls)

# -

