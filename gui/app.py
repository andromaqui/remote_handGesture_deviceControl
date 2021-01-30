from PIL import Image, ImageOps
from tensorflow import keras
import multiprocessing
import numpy as np
import threading, time, logging, cv2, sys
sys.path.insert(1, '../backend')
from api_requests import turn_off, turn_on

## if probability less than theshold
## count as no class
PROBABILITY_THRESHOLD = 0.5

## frames per second
TOTALFRAMES_PERSECOND = 20


blocking = True
model = keras.models.load_model("model_res50_5classes.h5")
cam = cv2.VideoCapture(0)

if not cam.isOpened():
  print ("Could not open cam")
  exit()

####region of interest dimensions
startX =  800
startY = 0
finishX = 1200
finishY = 400


#prevent numpy exponential 
np.set_printoptions(suppress=True)

currentFrame = 0
classPerFrame = []

counter = 0
## TODO
##calculate FPS
##

### take the last 20 frames -> for each frame map a class -> take most common value
while(1):
    turnOn_thread = threading.Thread(target=turn_on, name="turnBulbOn", args=[])
    turnOff_thread = threading.Thread(target=turn_off, name="turnBulbOff", args=[])

    ## RESET FRAME COUNT
    if(currentFrame == TOTALFRAMES_PERSECOND):
        currentFrame = 0
        #print(classPerFrame)
        averageFrameClass = max(set(classPerFrame), key=classPerFrame.count)

        if(averageFrameClass == 0):
            logging.info("Thread to TURN ON bulb started")
            turnOn_thread.start()
        elif(averageFrameClass == 3):
            logging.info("Thread to TURN OFF bulb started")
            turnOff_thread.start()

        #print(averageFrameClass)

    ret, frame = cam.read()
    currentFrame+=1
    if ret:
        frame = cv2.flip(frame,1)
        display = cv2.rectangle(frame.copy(),(startX,startY),(finishX,finishY),(0,0,255),2) 
        cv2.imshow('Total Input',display)
        ROI = frame[startY:finishY, startX:finishX].copy()
        cv2.imshow('Region of Interest', ROI)

        #cv2.imwrite("A" + str(counter)+".jpg", ROI) 
        #counter+=1
        """    
         ## add autocontrast
        img = Image.fromarray(display)
        # applying autocontrast method  
        img = ImageOps.autocontrast(img, cutoff = 1, ignore = 1) 
        numpyimg = np.array(img)
        """

        img = cv2.resize(ROI, (128, 128)) #R
        img = img.reshape(1, 128, 128, 3)
        
        predictions = model.predict(img) # Make predictions towards the test set
        predictions = predictions.flatten()

        predicted_label = np.argwhere(predictions > PROBABILITY_THRESHOLD) # Get index of the predicted label from prediction
        try:
            print(predicted_label[0][0])
            classPerFrame.append(predicted_label[0][0])
        except Exception as e:
            print(str(counter)+predicted_label)

        if cv2.waitKey(10) & 0xFF == ord('q'):
          break

cam.release()
cv2.destroyAllWindows()
