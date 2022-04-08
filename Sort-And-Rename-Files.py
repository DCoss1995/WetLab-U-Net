#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 10:58:25 2022

@author: wetlab
"""
'''
    - SISTEMARE IL RENAME DEI FILE
    - SORTARE I FILE NELLE DIRECTORY
'''
#%% IMPORT SECTION
import IPython.display as display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import datetime, os
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from IPython.display import clear_output
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import tifffile as tif
import random
import shutil
import math
from tensorflow.keras.layers import *
#%% CHECK DIRECTORIES
# Get the current working directory and print files in directory.
currWD = os.getcwd()
print(currWD)
currWDFiles = os.listdir(currWD)
print('Files in the directory:',currWDFiles)

# Ask user if is the correct one. If not, select folder where all images, grayscale
# and annotations, have been stored.
root = tk.Tk()
root.withdraw()
title = 'Current Working Directory'
message = 'Is this the correct working directory?\n'  + currWD
workDirQuest = tk.messagebox.askquestion(title, message)
if workDirQuest == 'yes':
    yesTitle = 'YES!'
    yesMessage = 'Ok!'
    messagebox.showinfo(yesTitle, yesMessage)
else:
    noTitle = 'NO!'
    noMessage = 'Ok! Select the correct one'
    messagebox.showinfo(noTitle, noMessage)
    newWorkDir = tk.filedialog.askdirectory()
    newDir = os.chdir(newWorkDir)
    print(newDir)
    newDirFiles = os.listdir(newDir)
    print('Files in the directory:', newDirFiles)
    
# Get the number of elements. It will be used to split images into train and
# validation datasets. 
currWD = os.getcwd()
listFiles = os.listdir(currWD)
numEl = len(listFiles)
print(numEl)
#%% SORT IMAGES
# Images have to be split into grayscale and annotations.
# Then, images have to be split into training and validation.

# Create a list of numbers and sample (the number is present only one time)--
# a number equal to the number of images you want to use for testing.
# PAY ATTENTION: there are k original images and k mask images.
numbersList = []
for i in range(numEl): # change range number on the basis of the number of available images
    numbersList.append(i)

# From examples, validation dataset is about the 39% of total images.
kVal = int(numEl*0.39)
valNumbers = random.sample(numbersList,k=kVal)

# Put the rest of numbers as training numbers
for num in valNumbers:
    numbersList.remove(num)

trainNumbers = numbersList

# Add zeros to numbers in lists
# Validation
numberValid = []
for num in valNumbers:
    num = str(num)
    if len(num) == 1:
        num = '000'+(num)
        numberValid.append(num)
    elif len(num) == 2:
        num = '00'+num
        numberValid.append(num)
    elif len(num) == 3:
        num = '0' + num
        numberValid.append(num)
 
# Training
numberTrain = []
for num in trainNumbers:
    num = str(num)
    if len(num) == 1:
        num = '000'+(num)
        numberTrain.append(num)
    elif len(num) == 2:
        num = '00'+num
        numberTrain.append(num)
    elif len(num) == 3:
        num = '0' + num
        numberTrain.append(num)
    
# Check if any number is present in other lists
print([i for i in numberTrain if i in numberValid])

#%% MOVE FILES TO RESPECTIVE DIRECTORIES
# Now, we move files to respective directories. I suggest to call your images annotations (for masks) and
# gray for your grayscale images.
# Select the directory for training and validation.
message = tk.messagebox.showinfo('Annotations Training Images', 
                                 'Select the annotation training directory')
annTrainImages = tk.filedialog.askdirectory()

message = tk.messagebox.showinfo('Annotations Validation Images',
                                 'Select the annotation validation directory')
annValidImages = tk.filedialog.askdirectory()

message = tk.messagebox.showinfo('Grayscale Training Images',
                                 'Select the grayscale training directory')
imgTrainImages = tk.filedialog.askdirectory()

message = tk.messagebox.showinfo('Grayscale Validation Images',
                                 'Select the grayscale validation directory')
imgValidImages = tk.filedialog.askdirectory()
#%%
#Get current working directory (it should be the one with all images, both annotations and grayscale)
currWD = os.getcwd()
# Move files to directories
sourceList = os.listdir(currWD)

# Grayscale training directory
grayTrain = []
sourceList = os.listdir(currWD)
for file in sourceList:
    for num in numberTrain:
        if file.endswith('Grayscale'+num+'.png'):
            grayTrain.append(file)

for trainGray in grayTrain:
    source = currWD + '/' + trainGray
    destination = imgTrainImages + '/' + trainGray
    shutil.move(source,destination)   

# Grayscale validation directory
grayValid = []
sourceList = os.listdir(currWD)    
for file in sourceList:
    for num in numberValid:
        if file.endswith('Grayscale'+num+'.png'):
            grayValid.append(file)

for validGray in grayValid:
    source = currWD + '/' + validGray
    destination = imgValidImages + '/' + validGray
    shutil.move(source,destination)
 
# Annotations training directory
maskTrain = []
sourceList = os.listdir(currWD)
for file in sourceList:
    for num in numberTrain:
        if file.endswith('Annotations'+num+'.png'):
            maskTrain.append(file)

for trainMask in maskTrain:
    source = currWD + '/' + trainMask
    destination = annTrainImages + '/' + trainMask
    shutil.move(source,destination)
  
# Annotations validation directory
maskValid = []
sourceList = os.listdir(currWD)
for file in sourceList:
    for num in numberValid:
        if file.endswith('Annotations'+num+'.png'):
            maskValid.append(file)

for validMask in maskValid:
    source = currWD + '/' + validMask
    destination = annValidImages + '/' + validMask
    shutil.move(source,destination)

#%% RENAME FILES
# Rename annotation train images
os.chdir(annTrainImages)
currFiles = sorted(os.listdir(annTrainImages))
count = 0
for filename in currFiles:
    if 'Annotations' in filename:
        old_name = filename
        new_name = filename.replace(filename,'Training_'+str(count)+'.png')
        os.rename(old_name,new_name)
        count += 1

# Rename annotation validation images
os.chdir(annValidImages)
currFiles = sorted(os.listdir(annValidImages))
count = 0
for filename in currFiles:
    if 'Annotations' in filename:
        old_name = filename
        new_name = filename.replace(filename,'Validation_'+str(count)+'.png')
        os.rename(old_name, new_name)
        count += 1
    
# Rename grayscale training images
os.chdir(imgTrainImages)
currFiles = sorted(os.listdir(imgTrainImages))
count = 0
for filename in currFiles:
    if 'Grayscale' in filename:
        old_name = filename
        new_name = filename.replace(filename,'Training_'+str(count)+'.png')
        os.rename(old_name, new_name)
        count += 1
    
# Rename grayscale validation images
os.chdir(imgValidImages)
currFiles = sorted(os.listdir(imgValidImages))
count = 0
for filename in currFiles:
    if 'Grayscale' in filename:
        old_name = filename
        new_name = filename.replace(filename,'Validation_'+str(count)+'.png')
        os.rename(old_name, new_name)
        count += 1
#%%