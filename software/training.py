# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:35:51 2018

for printing:
for i, layer in enumerate(base_model.layers):
   print(i, layer.name)

if debugging is needed:
import pdb
pdb.set_trace()
"""

# -------------------- Importing modules -----------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.models import Model, load_model
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

# --------------------- Variables -------------------------------------------

steps_per_epoch = 30
epochs = 5

# --------------------- Importing model -----------------------------------

model = load_model('model.h5')

# ------------------------ Prepare data --------------------------------------

DATA_DIR_TRAIN_AUG = "/mnt/nvme/wounds/data/Train"
DATA_DIR_VAL = "/mnt/nvme/wounds/data/Binary_Wound_Val"
DATA_DIR_TEST = "/mnt/nvme/wounds/data/TestImage"
DATA_DIR_SAVE = "/mnt/nvme/wounds/data/Augmented_Images"

datagen = image.ImageDataGenerator(
        rescale=1./255
        )

train_generator = datagen.flow_from_directory(
        DATA_DIR_TRAIN_AUG,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
        )

validation_generator = datagen.flow_from_directory(
        DATA_DIR_VAL,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical')

test_generator = datagen.flow_from_directory(
        DATA_DIR_TEST,
        target_size=(224, 224),
        batch_size=6,
        class_mode='categorical'
        )

# ---------------- Training entire network --------------------------

for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])

train = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=5
        )

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

# --------------- Precision and recall ---------------------------------------

#from sklearn.metrics import precision_recall_curve
#import itertools
#batch_size_val = 20
#
#batch, true_labels = next(validation_generator)
#
#val_labels = []
#val_preds = []
#val_probs = []
#
#
#for idx, (batch, labels) in enumerate(validation_generator):
#    pred = model.predict(batch)
#    val_preds.append(pred.argmax(axis=1))
#    val_probs.append(pred)
#    val_labels.append(labels.argmax(axis=1))
#    if idx == (2256 // batch_size_val):
#        break
#
#val_preds = list(itertools.chain(*val_preds))
#val_labels = list(itertools.chain(*val_labels))
#
#[pos_class, neg_class] = validation_generator.class_indices
#classes = [pos_class, neg_class]
#
#valid_probs = list(itertools.chain(*[p[:, classes.index('Venous')] for p in val_probs]))
#valid_labels = np.array(val_labels) == classes.index('Venous')
#
#precision, recall, thresholds = precision_recall_curve(valid_labels, valid_probs)
#
#plt.style.use('seaborn-notebook')
#plt.figure(figsize=(8, 6))
#plt.plot(recall, precision)
#plt.plot(recall[precision > .999], precision[precision > .999], c='green')
#plt.plot(recall[recall > .999], precision[recall > .999], c='green')
#plt.ylabel('Precision')
#plt.xlabel('Recall')
#plt.title('Precision-Recall curve for cheetahs')
#plt.ylim([precision.min()-0.01, precision.max()+0.01])
#plt.xlim([-0.05, 1.05])
#
#plt.savefig('prec_rec_curve.png')
#plt.clf()

# ------------ Creating plot and saving image --------------------------------
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

