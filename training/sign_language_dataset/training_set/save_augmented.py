import os, cv2
import tensorflow as tf
import numpy as np
import np_utils
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from tensorflow.python.client import device_lib
from keras import backend as K
from albumentations import (
    GaussNoise, RandomGamma, RandomBrightnessContrast
)

###############################
##### DATA AUGEMENTATION ######
###############################
def applyRandonBrightnessContrastAugmentation(img):
    brightnessContrastFactor = A.Compose([ A.RandomBrightnessContrast(p=1),])
    augmentation = brightnessContrastFactor(image=img)
    augmented_image = augmentation["image"]
    return augmented_image


def applyRandomGamma(img, fromRange, to):
    gammaFactor = A.RandomGamma(gamma_limit=(fromRange, to), always_apply=True)
    augmentation = gammaFactor(image=img)
    augmented_image = augmentation["image"]
    return augmented_image


def applyGaussNoise(img, fromRange, to):
    gaussNoiseFactor = A.GaussNoise(var_limit=(fromRange, to), mean=0, always_apply=True, p=1)
    augmentation = gaussNoiseFactor(image=img)
    augmented_image = augmentation["image"]
    return augmented_image



##############################
# here we collect all the paths
# to our train / test images
pathToImages = []
for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("jpg"):
            pathToImages.append(path)
            #print(path)

lettersDict = {'A' :0, 'B':1, 'U':2, 'V':3} 


h = 1000



#Loops through imagepaths to load images and labels into arrays
for path in pathToImages:

    # Reads image and returns np.array
    image = cv2.imread(path) 
    image = cv2.resize(image, (128, 128)) 

    folderTosave = "/" + path.split("/")[1]
    label = path.split("/")[2].split(".")[0][0]

    letterToNumber = lettersDict.get(label)

    randomBrightnessandContrast2 = applyRandonBrightnessContrastAugmentation(image)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomBrightnessandContrast2)
    h+=1

    randomBrightnessandContrast3 = applyRandonBrightnessContrastAugmentation(image)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomBrightnessandContrast3)
    h+=1

    randomGamma1 = applyRandomGamma(image, 100, 200)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomGamma1)
    h+=1

    randomGamma2 = applyRandomGamma(image, 200, 300)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomGamma2)
    h+=1

    randomGaussNoise1 = applyGaussNoise(image, 100, 200)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomGaussNoise1)
    h+=1

    randomGaussNoise2 = applyGaussNoise(image, 200, 300)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomGaussNoise2)
    h+=1

    randomBrightnessandContrast1 = applyRandonBrightnessContrastAugmentation(image)
    cv2.imwrite(folderTosave + "/" + label + str(h) + ".jpg", randomBrightnessandContrast1)
    h+=1
