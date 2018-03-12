# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 13:09:23 2018

"""

# -------------------- Importing modules --------------------------------------

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
import glob
import cv2
from PIL import Image as pil_image
from datetime import datetime
from imgaug import augmenters as iaa
K.clear_session()

#if debugging is needed
#import pdb
#pdb.set_trace()

# --------------------- Variables ---------------------------------------------

steps_per_epoch = 30
epochs = 10

# --------------------- Creating basemodel ------------------------------------

base_model = VGG19(weights='imagenet', include_top=False)
# without input_shape specified, expected input shape = 224,224,3

# -------------------- Modifying model ----------------------------------------

# add a global spatial average pooling layer
# Output tensor received via base_model.output
x = base_model.output
x = GlobalAveragePooling2D()(x)
# x = Dropout(0.2)(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
# inputs = list of inputs tensors, outputs = list of output tensors
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()

# ------------------------ Prepare data ---------------------------------------

# ImageDataGenerator decides which transformations are to be performed
DATA_DIR_TEST = "/mnt/nvme/wounds/data/pigment_bilder/test"
DATA_DIR_PIGMENT_TRAIN = "/mnt/nvme/wounds/data/pigment_bilder/train"
DATA_DIR_PIGMENT_VAL = "/mnt/nvme/wounds/data/pigment_bilder/val"

datagen = image.ImageDataGenerator(
        rescale=1./255
        )

train_pigment_generator = datagen.flow_from_directory(
        DATA_DIR_PIGMENT_TRAIN,
        target_size=(224, 224),
        batch_size=25,
        class_mode='categorical'
        )

validation_pigment_generator = datagen.flow_from_directory(
        DATA_DIR_PIGMENT_VAL,
        target_size=(224, 224),
        batch_size=18,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(
        DATA_DIR_TEST,
        target_size=(224, 224),
        batch_size=10,
        class_mode='categorical'
        )

# ---------------- Training entire network ------------------------------------

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train = model.fit_generator(
        train_pigment_generator,
        steps_per_epoch=steps_per_epoch, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs,
        validation_data=validation_pigment_generator,
        validation_steps=10
        )

# ---------------- Testing ----------------------------------------------------

test_images, test_labels = next(test_generator)

preds = model.predict_generator(test_generator, 1, verbose = 1)
print('True labels: \n %s' %test_labels)
preds_int = (preds > 0.5)*1
print('Predicted labels: \n %s' %preds_int)

# --------------- Evaluating model ------------------------------------------

score = model.evaluate_generator(validation_pigment_generator, 2)
print("Loss: %s" %score[0])
print("Acc: %s" %score[1])

# ------------ Creating plot and saving image ---------------------------------

parser = argparse.ArgumentParser()
parser.add_argument('imagename1')
parser.add_argument('filename')
args = parser.parse_args()

# summarize train for accuracy
plt.plot(train.history['acc'])
plt.plot(train.history['val_acc'])
plt.plot(train.history['loss'])
plt.plot(train.history['val_loss'])
plt.title('training')
plt.ylabel('accuracy/loss')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc', 'loss', 'val_loss'], loc='upper left')
# set axis
plt.ylim([0, 1])
plt.savefig(args.imagename1)
plt.clf()

# ------------- Saving results in textfile ------------------------------------

with open(args.filename, 'w') as file:
    file.write('\n Number of epochs in training: %s' %epochs)
    file.write('\n Number of steps in each epoch: %s' %steps_per_epoch)
    file.write('\n Final val. acc: %s' %score[1])
    file.write('\n Final val. loss: %s' %score[0])
    file.write('\n Test images, labels: \n %s' %test_labels)
    file.write('\n Test images, predicted labels: \n %s' %preds_int)
    file.write('\n Test images, predicted labels (exact): \n %s' %preds)

    file.write('\n acc: \n %s' %train.history['acc'])
    file.write('\n val. acc: \n %s' %train.history['val_acc'])
    file.write('\n loss: \n %s' %train.history['loss'])
    file.write('\n val. loss: \n %s' %train.history['val_loss'])


# ------------ Saving model ---------------------------------------------------

model.save('model.h5')
