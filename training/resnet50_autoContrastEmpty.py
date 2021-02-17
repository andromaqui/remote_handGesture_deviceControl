from tensorflow.python.client import device_lib
import os, cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
import np_utils
import albumentations as A
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix # Helps present results as a confusion-matrix
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from tensorflow.python.client import device_lib
from keras import backend as K
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
#from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse


#############################
######## TRAINING ###########
############################# 

##############################
# here we collect all the paths
# to our train / test images
pathToImages = []
for root, dirs, files in os.walk(".", topdown=False): 
    for name in files:
        path = os.path.join(root, name)
        if path.endswith("jpg"):
            pathToImages.append(path)

print(len(pathToImages)) 
X = [] # Image data
y = [] # Labels
lettersDict = {'A' :0, 'B': 1, 'U':2, 'V' :3, 'E': 4} 


# Loops through imagepaths to load images and labels into arrays
for path in pathToImages:
    # Reads image and returns np.array
    img = cv2.imread(path) 
    img = cv2.resize(img, (128, 128)) 

    try:
     label = path.split("/")[3].split(".")[0][0]
     X.append(img)
   
    except Exception as e:
        print(e)
        print(path)
    letterToNumber = lettersDict.get(label)
    y.append(letterToNumber)


X = np.array(X, dtype="uint8")
y = np.array(y)

print("Images loaded: ", len(X))
print("Labels loaded: ", len(y))
print(tf.__version__)

ts = 0.3 # Percentage of images that we want to use for testing. The rest is used for training.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)

baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(128, 128, 3)))

# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(4, 4))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(5, activation="softmax")(headModel)

# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the training process
for layer in baseModel.layers:
    layer.trainable = False

# Construction of model
model = Sequential()
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(128, 128, 3))) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))

# Configures the model for training
model.compile(optimizer='adam', # Optimization routine, which tells the computer how to adjust the parameter values to minimize the loss function.
              loss='sparse_categorical_crossentropy', # Loss function, which tells us how bad our predictions are.
              metrics=['accuracy']) # List of metrics to be evaluated by the model during training and testing.

# Trains the model for a given number of epochs (iterations on a dataset) and validates it.
model.fit(X_train, y_train, epochs=15, batch_size=64, verbose=2, validation_data=(X_test, y_test))

# Save entire model to a HDF5 file
model.save('model_res50_5classes_autocontrast.h5')

print(X_test.shape)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy: {:2.2f}%'.format(test_acc*100))
predictions = model.predict(X_test) # Make predictions towards the test set
np.argmax(predictions[0]), y_test[0] # If same, got it right
