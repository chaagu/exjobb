# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:35:51 2018
for printing:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

@author: medvel
"""

# -------------------- Importing modules -----------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import SGD
from keras import backend as K
import numpy as np
import argparse

#if debugging is needed
#import pdb
#pdb.set_trace()

K.clear_session()

# --------------------- Variables -------------------------------------------
# First training
steps_per_epoch_1 = 30
epochs_1 = 50

# Second training
steps_per_epoch_2 = 30
epochs_2 = 100

# --------------------- Creating basemodel -----------------------------------

# create the base pre-trained model
base_model = VGG19(weights='imagenet', include_top=False)
#base_model = VGG19(weights='imagenet')
# without input_shape specified, expected input shape = 224,224,3


# -------------------- Modifying model ---------------------------------------


# add a global spatial average pooling layer
# Output tensor received via base_model.output
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
# inputs = list of inputs tensors, outputs = list of output tensors
model = Model(inputs=base_model.input, outputs=predictions)
# model.summary()




# --------- Modifying model by manually removing only the last layer ---------
#
##model = Sequential()
##for layer in base_model.layers:
##    model.add(layer)
#
##model.layers.pop()
#
# if only the final later is to be trained:
# for layers in model.layers:
#    layer.trainable = False
#
##model.add(Dense(2, activation = 'softmax'))

##for layer in model.layers[:19]:
##    layer.trainable = False
##for layer in model.layers[19:]:
##    layer.trainable = True

# ------------------------ Prepare data --------------------------------------



# ImageDataGenerator decides which transformations are to be performed
DATA_DIR_TRAIN = "/mnt/nvme/wounds/data/Binary_Wound_Train"
DATA_DIR_VAL = "/mnt/nvme/wounds/data/Binary_Wound_Val"
DATA_DIR_TEST = "/mnt/nvme/wounds/data/TestImage"

DATA_DIR_TRAIN_PIGMENT = "/mnt/nvme/wounds/data/pigment_bilder/train"
DATA_DIR_VAL_PIGMENT = "/mnt/nvme/wounds/data/pigment_bilder/val"

DATA_DIR_SAVE = "/mnt/nvme/wounds/data/Augmented_Images"

train_datagen_pigment = image.ImageDataGenerator(
            rescale=1./255,
            rotation_range=110,
            horizontal_flip=True,
            vertical_flip=True
            )

test_datagen_pigment = image.ImageDataGenerator(rescale=1./255)

train_generator_pigment = train_datagen_pigment.flow_from_directory(
            DATA_DIR_TRAIN_PIGMENT,
            target_size=(224, 224),
            batch_size=48,
            class_mode='categorical'
            )

validation_generator_pigment = test_datagen_pigment.flow_from_directory(
            DATA_DIR_VAL_PIGMENT,
            target_size=(224, 224),
            batch_size=10,
            class_mode='categorical')

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=110,
        horizontal_flip=True,
        vertical_flip=True
        )

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        DATA_DIR_TRAIN,
        target_size=(224, 224),
        batch_size=8,
        class_mode='categorical'
#        save_to_dir=DATA_DIR_SAVE,
#        save_prefix='im',
#        save_format='png'
        )

validation_generator = test_datagen.flow_from_directory(
        DATA_DIR_VAL,
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        DATA_DIR_TEST,
        target_size=(224, 224),
        batch_size=6,
        class_mode='categorical'
        )
# class_mode Determines the type of label arrays that are returned:
#"categorical" will be 2D one-hot encoded labels, "binary" will be 1D binary labels

#x_batch, y_batch = next(train_generator)
#x_val, y_val = next(validation_generator)

# ------------ Training the entire network on pigment images ------------------

for layer in model.layers:
       layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

first_train_pigment = model.fit_generator(
    train_generator_pigment,
    steps_per_epoch=16, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
    epochs=75,
    validation_data=validation_generator_pigment,
    validation_steps=20
    )

# ---------------- Training top (newly added) layers --------------------------


# first: train only the top layers (which were randomly initialized)
# i.e. freeze all convolutional InceptionV3 layers
for layer in base_model.layers:
    layer.trainable = False

# compile the model (should be done *after* setting layers to non-trainable)
#from keras.optimizers import rmsprop
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
#decrease learningrate: (lr=0.0001)
model.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

first_train = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_1, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=5
        )

# let's visualize layer names and layer indices to see how many layers
# we should freeze:
# Gives a list similar to:
# 0 layer 1
# 1 layer 2
# etc
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# -------------------- Training top (base) convlayers ------------------------

# at this point, the top layers are well trained and we can start fine-tuning
# convolutional layers. We will freeze the bottom N layers
# and train the remaining top layers. We will freeze
# the first 10 layers and unfreeze the rest:

for layer in model.layers[:17]:
   layer.trainable = False
for layer in model.layers[17:]:
   layer.trainable = True

# we need to recompile the model for these modifications to take effect
# we use SGD with a low learning rate
model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])

# we train our model again (this time fine-tuning the top 2 inception blocks
# alongside the top Dense layers
second_train = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_2, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs_2,
        validation_data=validation_generator,
        validation_steps=5)


# ---------------- Testing --------------------------------------------------

test_images, test_labels = next(test_generator)

preds = model.predict_generator(test_generator, 1, verbose = 1)
print('True labels: \n %s' %test_labels)
preds_int = (preds > 0.5)*1
print('Predicted labels: \n %s' %preds_int)

# --------------- Evaluating model ------------------------------------------

score = model.evaluate_generator(validation_generator, 2)
print("Loss: %s" %score[0])
print("Acc: %s" %score[1])

# ------------ Creating plot and saving image ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('imagename1')
parser.add_argument('imagename2')
parser.add_argument('filename')
args = parser.parse_args()

# summarize first_train for accuracy
plt.plot(first_train.history['acc'])
plt.plot(first_train.history['val_acc'])
plt.plot(first_train.history['loss'])
plt.plot(first_train.history['val_loss'])
plt.title('first_train')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='upper left')
# set axis
plt.ylim([0, 1])
plt.savefig(args.imagename1)
plt.clf()

# summarize second_train for accuracy
plt.plot(second_train.history['acc'])
plt.plot(second_train.history['val_acc'])
plt.plot(second_train.history['loss'])
plt.plot(second_train.history['val_loss'])
plt.title('second_train')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='upper left')
# set axis
plt.ylim([0, 1])
plt.savefig(args.imagename2)

#for i, layer in enumerate(model.layers):
#    print(i,layer.name)

# ------------- Saving results in textfile ------------------------------------
# parser = argparse.ArgumentParser()

#parser.add_argument('filename')
#args = parser.parse_args()
with open(args.filename, 'w') as file:
    file.write('\n Number of epochs in first training (top layers only): %s' %epochs_1)
    file.write('\n Number of steps in each epoch: %s' %steps_per_epoch_1)
    file.write('\n Number of epochs in second training: %s' %epochs_2)
    file.write('\n Number of steps in each epoch: %s' %steps_per_epoch_2)
    file.write('\n Final val. acc: %s' %score[1])
    file.write('\n Final val. loss: %s' %score[0])
    file.write('\n Test images, labels: \n %s' %test_labels)
    file.write('\n Test images, predicted labels: \n %s' %preds_int)
    file.write('\n Test images, predicted labels (exact): \n %s' %preds)

    file.write('\n First training, acc: \n %s' %first_train.history['acc'])
    file.write('\n First training, val. acc: \n %s' %first_train.history['val_acc'])
    file.write('\n First training, loss: \n %s' %first_train.history['loss'])
    file.write('\n First training, val. loss: \n %s' %first_train.history['val_loss'])

    file.write('\n Second training, acc: \n %s' %second_train.history['acc'])
    file.write('\n Second training, val. acc: \n %s' %second_train.history['val_acc'])
    file.write('\n Second training, loss: \n %s' %second_train.history['loss'])
    file.write('\n Second training, val. loss: \n %s' %second_train.history['val_loss'])
