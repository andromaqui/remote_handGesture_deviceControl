# IoT Sign Language Remote Bulb Control

My motivation for this project started by my fascination for virtual assistant devices like Alexa and Google Assistant. At the same time, I noticed that there are only a few technologies around, which serve deaf and speech-impaired people. Therefore I decided to experiment with how far an AI could go with understanding sign-language and how this could be incomporated into a virtual assistant device.

This is a project being developed as part of my Bachelors thesis aiming to combine the areas of Machine Learning, Computer Vision and Web Development.  The app detects hand gestures from the user and sends the respective HTTP request to LIFX api in order to turn on / off the LIFX mini bulb. The detection of the hand gestures happens by passing each frame of the camera capture in a gesture classifier, trained on 848 images with the ResNet50 model.


- Palm gesture (B Letter): turns on the bulb
- V gesture (V letter): turns off the bulb

![Alt Text](demo.gif)

## Starting the project

First, after navigating to the project folder install the dependencies described in the `requirements.txt` file with the following command:
`pip3 install -r requirements.txt`

Afterwards, start the gui by running:
`python3 gui/app.py`


## Implementation

This project was built with Python 3. For the training of the classifier, **Tensorflow** with the **Keras API** was used. Moreover, the ResNet50 Convolutional Neural Network was used, pretrained on the ImageNet and fine-tuned with a 6 layer convolutional neural network.

## Dataset

### Training

Part of the utilized dataset originates from the following github page https://github.com/mon95/Sign-Language-and-Static-gesture-recognition-using-sklearn/blob/master/Dataset.zip, where images of the english sign language from 8 participants are contained. From there, images from **4 handgestures** (A, B, U & V) were extracted and the dataset was then extended by 6 additional participants. The final constracted training set contained images from 14 participants in total and for each participant there are 5 classes. 

- class letter A
- class letter B
- class letter U
- class letter V
- class empty (for when there is no gesture detected)

### Validation

Addinionally a validation dataset was used, which contains images the model has not seen before in order to calculate its accuracy. For the validation data-set 376 images were collected from another 3 participants.  

## Modules
- gui : the graphical user interface of the app. 
- backend : responsible for sending requests to LIFX api in order to control the Smart Bulb.
- training : includes the dataset used as well as the training / evaluation scripts, which were used for this project.


## Improvements
- document data collection and training process
- plot metrics for model
- update evaluation.ipynb
- speed up LIFX api requests
- add requirements.txt
- improve model 
