from PIL import Image, ImageOps
from tensorflow import keras
import multiprocessing
import numpy as np
import threading, time, logging, cv2, sys
sys.path.insert(1, '../backend')
from api_requests import turn_off, turn_on

## if probability less than theshold
## count as no class
PROBABILITY_THRESHOLD = 0.6

## frames per second
TOTALFRAMES_PERSECOND = 10


blocking = True
model = keras.models.load_model("model_res50_5classes_autocontrast.h5")
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
isOff = 0
isOn = 0


### take the last 20 frames -> for each frame map a class -> take most common value
while(1):
    turnOn_thread = threading.Thread(target=turn_on, name="turnBulbOn", args=[])
    turnOff_thread = threading.Thread(target=turn_off, name="turnBulbOff", args=[])
    

    ## RESET FRAME COUNT
    if(currentFrame == TOTALFRAMES_PERSECOND):
        currentFrame = 0
        #print(classPerFrame)
        averageFrameClass = max(set(classPerFrame), key=classPerFrame.count)
        print(averageFrameClass)
        try:
            if(averageFrameClass.item()==0 and isOn==0):
                print("turn on")
                is0n = 1
                isOff = 0
                turnOn_thread.start() 
                #turn_on()
                #logging.info("Thread to TURN ON bulb started")
            elif(averageFrameClass.item()==1 and isOff==0):
                print("turn off")
                is0ff = 1
                is0n = 0
                #logging.info("Thread to TURN OFF bulb started")
                turnOff_thread.start()
            else:
                print(averageFrameClass)
        except Exception as e:    
            print(e)   
        #print(averageFrameClass)
    
    ret, frame = cam.read()    
    if ret:
        currentFrame += 1
        frame = cv2.flip(frame,1)
        display = cv2.rectangle(frame.copy(),(startX,startY),(finishX,finishY),(0,0,255),2) 
        cv2.imshow('Total Input',display)
        ROI = frame[startY:finishY, startX:finishX].copy()
        cv2.imshow('Region of Interest', ROI)

        #cv2.imwrite("A" + str(counter)+".jpg", ROI) 
        #counter+=1
          
        
         ## apply autocontrast to images
         ## in order to equalize histogram
        img = Image.fromarray(ROI)
        # applying autocontrast method  
        img = ImageOps.autocontrast(img, cutoff = 1, ignore = 1) 
        numpyimg = np.array(img)
        

        img = cv2.resize(ROI, (128, 128)) #R
        img = img.reshape(1, 128, 128, 3)
        
        predictions = model.predict(img) # Make predictions towards the test set
        predictions = predictions.flatten()

        predicted_label = np.argwhere(predictions > PROBABILITY_THRESHOLD) # Get index of the predicted label from prediction
        try:
            #print(predicted_label[0][0])
            classPerFrame.append(predicted_label[0][0])
            """
            classItem = predicted_label[0][0]
            
            print(classItem)
            if((classItem.item())==0 and isOn==0):
                print("turn on")
                is0n = 1
                isOff = 0
                turnOn_thread.start() 
                #turn_on()
                #logging.info("Thread to TURN ON bulb started")
            elif(classItem.item() == 3 and isOn==0):
                print("turn off")
                is0ff = 1
                is0n = 0
                #turn_off()
                #logging.info("Thread to TURN OFF bulb started")
                turnOff_thread.start()
            else:
                print(classItem)
            """
        except Exception as e:
                continue
                """
                print("Exception")
                print(counter)
                print(predicted_label)
                classPerFrame.append(None)
                """


        if cv2.waitKey(10) & 0xFF == ord('q'):
          break

cam.release()
cv2.destroyAllWindows()
