# IoT Hand Gesture Remote Bulb Control

This is a project being developed as part of my Bachelors thesis aiming to combine the areas of Machine Learning, Computer Vision and Web Development. The app detects hand gestures from the user and sends the respective HTTP request to LIFX api in order to turn on / off the LIFX mini bulb.

- V gesture: turns off the bulb
- Palm gesture: turns on the bulb

![Alt Text](demo.gif)



## Modules
- gui : the graphical user interface of the app 
- backend : responsible for sending requests to LIFX api in order to control the Smart Bulb
- training : includes the dataset used as well as the several training scripts and the model architectures, which were used for this project

## TODO
- document data collection and training process
- update evaluation.ipynb
