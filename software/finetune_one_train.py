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
import glob
import cv2
from PIL import Image as pil_image
from datetime import datetime
from imgaug import augmenters as iaa

seq = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.10*255),
    iaa.ContrastNormalization((0.5, 1.8)),
    iaa.Multiply((0.5, 1.5))
])

images = []
for name in glob.glob('/mnt/nvme/wounds/data/Binary_Wound_Train/NotVenous/*.jpg'):
        a = np.array(cv2.imread(name))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        images.append(a)

count = 0
images_aug = seq.augment_images(images)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/NotVenous/"+str(datetime.now())+'.jpg')
    count += 1

images = []
for name in glob.glob('/mnt/nvme/wounds/data/Binary_Wound_Train/Venous/*.jpg'):
        a = np.array(cv2.imread(name))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        images.append(a)

count = 0
images_aug = seq.augment_images(images)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/Venous/"+str(datetime.now())+'.jpg')
    count += 1

#if debugging is needed
#import pdb
#pdb.set_trace()

K.clear_session()

# --------------------- Variables -------------------------------------------
# First training
steps_per_epoch_1 = 20
epochs_1 = 40

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
##x = Dropout(0.1)(x)
# let's add a fully-connected layer
x = Dense(2048, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
predictions = Dense(2, activation='softmax')(x)

# this is the model we will train
# inputs = list of inputs tensors, outputs = list of output tensors
model = Model(inputs=base_model.input, outputs=predictions)
model.summary()




# --------- Modifying model by manually removing only the last layer ---------
#
##model = Sequential()
##for layer in base_model.layers:
##    model.add(layer)
#
##model.layers.pop()
#
##for layers in model.layers:
##    layer.trainable = False
#
##model.add(Dense(2, activation = 'softmax'))
#
##model.summary()


# ------------------------ Prepare data --------------------------------------



# ImageDataGenerator decides which transformations are to be performed
DATA_DIR_TRAIN_AUG = "/mnt/nvme/wounds/data/Train"
DATA_DIR_TRAIN = "/mnt/nvme/wounds/data/Binary_Wound_Train"
DATA_DIR_VAL = "/mnt/nvme/wounds/data/Binary_Wound_Val"
DATA_DIR_TEST = "/mnt/nvme/wounds/data/TestImage"
DATA_DIR_SAVE = "/mnt/nvme/wounds/data/Augmented_Images"

train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=110,
        horizontal_flip=True,
        vertical_flip=True
        )

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_aug_generator = test_datagen.flow_from_directory(
        DATA_DIR_TRAIN_AUG,
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical'
        )

train_generator = train_datagen.flow_from_directory(
        DATA_DIR_TRAIN,
        target_size=(224, 224),
        batch_size=20,
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

# ---------------- Training top (newly added) layers --------------------------

#for layer in model.layers[:10]:
#   layer.trainable = False
#for layer in model.layers[10:]:
#   layer.trainable = True
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])

first_train = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch_1, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs_1,
        validation_data=validation_generator,
        validation_steps=5
        )

#------------------ Training using augmented images from imgaug----------------

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay = 0.0001), loss='binary_crossentropy', metrics=['accuracy'])

first_train = model.fit_generator(
        train_aug_generator,
        steps_per_epoch=steps_per_epoch_1, # total number of steps or batches of samples to yield from the generator before an epoch is finished: size of dataset/number of samples in batch
        epochs=epochs_1,
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

# ------------ Creating plot and saving image ---------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('imagename1')
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

#for i, layer in enumerate(model.layers):
#    print(i,layer.name)

# ------------- Saving results in textfile ------------------------------------

with open(args.filename, 'w') as file:
    file.write('\n Number of epochs in training: %s' %epochs_1)
    file.write('\n Number of steps in each epoch: %s' %steps_per_epoch_1)
    file.write('\n Final val. acc: %s' %score[1])
    file.write('\n Final val. loss: %s' %score[0])
    file.write('\n Test images, labels: \n %s' %test_labels)
    file.write('\n Test images, predicted labels: \n %s' %preds_int)
    file.write('\n Test images, predicted labels (exact): \n %s' %preds)

    file.write('\n acc: \n %s' %first_train.history['acc'])
    file.write('\n val. acc: \n %s' %first_train.history['val_acc'])
    file.write('\n loss: \n %s' %first_train.history['loss'])
    file.write('\n val. loss: \n %s' %first_train.history['val_loss'])

