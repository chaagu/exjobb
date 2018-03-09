# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 10:11:09 2018

"""
import numpy as np
import glob
import cv2
from PIL import Image as pil_image
from datetime import datetime
from imgaug import augmenters as iaa

# ----------------------- Creating images -------------------------------

images_notvenous = []
for name in glob.glob('/mnt/nvme/wounds/data/Train/NotVenous/*.jpg'):
        a = np.array(cv2.imread(name))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        images_notvenous.append(a)

images_venous = []
for name in glob.glob('/mnt/nvme/wounds/data/Train/Venous/*.jpg'):
        a = np.array(cv2.imread(name))
        a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        images_venous.append(a)

# ------------------- Defining augmentation ------------------------------

seq_noise = iaa.Sequential([
    iaa.AdditiveGaussianNoise(scale=0.10*255),
    iaa.ContrastNormalization((0.5, 1.8)),
    iaa.Multiply((0.5, 1.5))
])

seq_rot = iaa.Sequential([
    iaa.Affine(rotate=(-45, 45)),
    iaa.Flipud(0.5),
    iaa.Fliplr(0.5)
])

# ------------------ Performing augmentation -------------------------------

count = 0
images_aug = seq_noise.augment_images(images_notvenous)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/NotVenous/"+str(datetime.now())+'.jpg')
    count += 1

count = 0
images_aug = seq_rot.augment_images(images_notvenous)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/NotVenous/"+str(datetime.now())+'.jpg')
    count += 1

count = 0
images_aug = seq_noise.augment_images(images_venous)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/Venous/"+str(datetime.now())+'.jpg')
    count += 1

count = 0
images_aug = seq_rot.augment_images(images_venous)
for number in range(len(images_aug)):
    image_array = images_aug[count]
    aug_image = pil_image.fromarray(image_array,mode='RGB')
    aug_image.save("/mnt/nvme/wounds/data/Train/Venous/"+str(datetime.now())+'.jpg')
    count += 1
