#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 15:32:20 2022

@author: wetlab
"""
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
#%% CREATE THE DATALOADER
# Set some parameters.
''' 
START FROM HERE IF YOUR FILES ARE READY.
'''
root = tk.Tk()
root.withdraw()
# Set some parameters that will be useful.
imgHeight = tk.simpledialog.askinteger('Image Height',
                                       'What value do you want to use for the image height?',
                                       initialvalue = '128')
imgWidth = tk.simpledialog.askinteger('Image Width',
                                      'What value do you want to use for the image width?',
                                      initialvalue = '128')
nChannels = tk.simpledialog.askinteger('Number of Channels',
                                       'How many channels do you want?',
                                       initialvalue = '3')
nClasses = tk.simpledialog.askinteger('Number of Classes',
                                      'How many classes do you have?',
                                      initialvalue = '2')
seed = tk.simpledialog.askinteger('Seed',
                                  'What value do you want to use as seed?',
                                  initialvalue = '42')

batchSize = tk.simpledialog.askinteger('Batch Size',
                                       "What value do you want to use as batch size?",
                                       initialvalue = '32')

# Ask user if the directory is the correct one. If not, select the grayscale folder,
# not the annotations one.
currWD = os.getcwd()
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

# Get infos about paths of: images dataset, training dataset, validation dataset.
# All images dataset
dataset_path = os.getcwd()
print(dataset_path)
# Training dataset
training_dataset = dataset_path + "/training/"
trainList = os.listdir(training_dataset)
trainSize = len(trainList)
print(trainSize)
# Validation dataset
validation_dataset = dataset_path + "/validation/"
validationList = os.listdir(validation_dataset)
valSize = len(validationList)
print(valSize)

# Function to retrieve the image and corresponding annotation (=mask).
def parse_image(img_path: str) -> dict:
    # Load an image and the corresponding annotation, returning them into dictionary
    image = tf.io.read_file(img_path) # read the image from the folder
    image = tf.image.decode_png(image, channels=3) # decode the png image as an rgb, even if it is a grayscale image. It will be normalized
    # in range [0 1] later.
    image = tf.image.convert_image_dtype(image, tf.uint8) # the png image is converted to unsigned int 8 bit.
    
    mask_path = tf.strings.regex_replace(img_path,'images','annotations')
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask,channels=1)
    #mask = tf.where(mask == 255, np.dtype('uint8').type(0), mask)
    
    return {'image': image, 'segmentation_mask': mask}

# Create datasets from folder using the parse_image function
train_dataset = tf.data.Dataset.list_files(training_dataset + '*.png', seed = seed)
train_dataset = train_dataset.map(parse_image)

val_dataset = tf.data.Dataset.list_files(validation_dataset + '*.png', seed = seed)
val_dataset = val_dataset.map(parse_image)

# Generate a single dataset, as dictionary, with train and test dataset
dataset = {'train': train_dataset, 'val': val_dataset}

print('################')
print('General Info about dataset:')
print(dataset)
print('################')
print('Info about training dataset:')
print(dataset['train'])
print('################')
print('Info about validation dataset:')
print(dataset['val'])

# If any warning appears in IPython console, it can be ignored. https://github.com/tensorflow/tensorflow/issues/37144
#%% PREPROCESSING THE IMAGE
# Define the functions for the preprocessing of images and annotations.
@tf.function
def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (imgHeight, imgWidth), method="nearest")
    input_mask = tf.image.resize(input_mask, (imgHeight, imgWidth), method="nearest")

    return input_image, input_mask 


@tf.function
def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask = tf.cast(input_mask,tf.float32) / 255.0
    #input_mask -= 1
  
    return input_image, input_mask

# Define function for loading train and test images (with annotations), and preprocess them.
@tf.function
def load_image_train(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

@tf.function
def load_image_test(datapoint):
    input_image = datapoint["image"]
    input_mask = datapoint["segmentation_mask"]
    input_image, input_mask = resize(input_image, input_mask)
    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

# Preprocess train dataset
dataset['train'] = dataset['train'].map(load_image_train, num_parallel_calls = tf.data.AUTOTUNE)
dataset['val'] = dataset['val'].map(load_image_test, num_parallel_calls = tf.data.AUTOTUNE)

print(dataset['train'])
train_batches = dataset['train'].cache().shuffle(buffer_size = 1000).batch(batchSize).repeat()
train_batches = train_batches.prefetch(buffer_size = tf.data.AUTOTUNE)
print(train_batches)
'''
dataset['train'] = dataset['train'].shuffle(buffer_size = 1000, seed = seed)
dataset['train'] = dataset['train'].repeat()
dataset['train'] = dataset['train'].batch(batchSize)
dataset['train'] = dataset['train'].prefetch(buffer_size = tf.data.AUTOTUNE)
'''
numOfTest = 28
# Process validation dataset
validation_batches = dataset['val'].take(190).batch(batchSize)
test_batches = dataset['val'].skip(190).take(numOfTest).batch(batchSize)

print(validation_batches)
print(test_batches)

# For informations about what buffer_size does: 
# https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
#%% DATA VISUALIZATION
# Display some data for ensuring that images and masks are in the dataset
def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ["Input Image", "True Mask", "Predicted Mask"]

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
    plt.axis("off")
  plt.show()

sample_batch = next(iter(train_batches))
random_index = np.random.choice(sample_batch[0].shape[0])
sample_image, sample_mask = sample_batch[0][random_index], sample_batch[1][random_index]
display([sample_image, sample_mask])
#%% U-NET BUILDING BLOCKS
def double_conv_block(x, n_filters):
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   # Conv2D then ReLU activation
   x = layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
   return x

def downsample_block(x, n_filters):
   f = double_conv_block(x, n_filters)
   p = layers.MaxPool2D(2)(f)
   p = layers.Dropout(0.3)(p)
   return f, p

def upsample_block(x, conv_features, n_filters):
   # upsample
   x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
   # concatenate
   x = layers.concatenate([x, conv_features])
   # dropout
   x = layers.Dropout(0.3)(x)
   # Conv2D twice with ReLU activation
   x = double_conv_block(x, n_filters)
   return x

def build_unet_model():
 # inputs
   inputs = layers.Input(shape=(128,128,3))
   # encoder: contracting path - downsample
   # 1 - downsample
   f1, p1 = downsample_block(inputs, 64)
   # 2 - downsample
   f2, p2 = downsample_block(p1, 128)
   # 3 - downsample
   f3, p3 = downsample_block(p2, 256)
   # 4 - downsample
   f4, p4 = downsample_block(p3, 512)
   # 5 - bottleneck
   bottleneck = double_conv_block(p4, 1024)
   # decoder: expanding path - upsample
   # 6 - upsample
   u6 = upsample_block(bottleneck, f4, 512)
   # 7 - upsample
   u7 = upsample_block(u6, f3, 256)
   # 8 - upsample
   u8 = upsample_block(u7, f2, 128)
   # 9 - upsample
   u9 = upsample_block(u8, f1, 64)
   # outputs
   outputs = layers.Conv2D(3, 1, padding="same", activation = "softmax")(u9)
   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
   return unet_model

unet_model = build_unet_model()
unet_model.summary()
tf.keras.utils.plot_model(unet_model, show_shapes = True)
#%% COMPILE AND TRAIN THE U-NET
unet_model.compile(optimizer = tf.keras.optimizers.Adam(),
                   loss = 'sparse_categorical_crossentropy',
                   metrics = "accuracy")

NUM_EPOCHS = 40

STEPS_PER_EPOCH = trainSize // batchSize

#VAL_SUBSPLITS = 5
VAL_STEPS = valSize // batchSize #// VAL_SUBSPLITS

model_history = unet_model.fit(train_batches,
                               epochs = NUM_EPOCHS,
                               steps_per_epoch = STEPS_PER_EPOCH,
                               validation_steps = VAL_STEPS,
                               validation_data = validation_batches)

#%% LEARNING CURVE
def display_learning_curves(history):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    epochs_range = range(NUM_EPOCHS)

    fig = plt.figure(figsize=(12,6))

    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label="train accuracy")
    plt.plot(epochs_range, val_acc, label="validataion accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(epochs_range, loss, label="train loss")
    plt.plot(epochs_range, val_loss, label="validataion loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")

    fig.tight_layout()
    plt.show()
# Display learning curves 
display_learning_curves(unet_model.history)
#%%
currWD = os.getcwd()
unet_model.save(currWD+'Model.tf',save_format = 'tf')
def create_mask(pred_mask):
  pred_mask = tf.argmax(pred_mask, axis=-1)
  pred_mask = pred_mask[..., tf.newaxis]
  for i , image in enumerate(pred_mask,1):
      tf.keras.utils.save_img(f'/home/wetlab/Desktop/Unet Vessels/Predictions/Predicted-image-{i}.png', image)

  return pred_mask[0]

def show_predictions(dataset=None,num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = unet_model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(unet_model.predict(sample_image[tf.newaxis, ...]))])
    
count = 0
for i in test_batches:
    count += 1
print('number of batches: ', count)

for i in range(0,numOfTest): # cambiare 29 con numOfTest + 1
    show_predictions(test_batches, 1)
    print(i)
