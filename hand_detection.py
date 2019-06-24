import cv2
import datetime
import argparse
import imutils
from imutils.video import VideoStream

import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from keras.models import load_model
import keras
from keras.models import Sequential
from keras import regularizers
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.applications import ResNet50,VGG16
from keras.applications.resnet50 import preprocess_input
from keras import Model,layers
from keras.models import load_model,model_from_json

from utils import detector_utils as detector_utils

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                               'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                               'Z':25,'nothing':26,'space':27,'delete':28,'complete':29}

def create_model():
    
    model = Sequential()
    
    model.add(Conv2D(16, kernel_size = [3,3], padding = 'same', activation = 'relu', input_shape = (150,150,3)))
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(32, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(Conv2D(128, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = [3,3], padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = [3,3]))
    
    model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = 'relu', kernel_regularizer = regularizers.l2(0.001)))
    model.add(Dropout(0.5))
    model.add(Dense(26, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
    
    print("MODEL CREATED")
    model.summary()
    
    return model

def fit_model():
    model_hist = model.fit(X_train, Y_train, batch_size = 32, epochs = 20, validation_split = 0.1)
    return model_hist 

def get_labels_for_plot(predictions):
    predictions_labels = []
    for ins in labels_dict:
        if predictions == labels_dict[ins]:
            predictions_labels.append(ins)
            return ins

            #break
    #return predictions_labels

def load_test_dataa():
    images = []
    names = []
    size = 50,50
    temp = cv2.imread('./img_thr2.jpg')
    temp = cv2.resize(temp, size)
    #temp = cv2.cvtColor(temp,cv2.COLOR_RGB2GRAY)
    images.append(temp)
    names.append('img2')
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names

#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    thresholded = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, _) = cv2.findContours(thresholded.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (thresholded, segmented)

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60
    
    #model = create_model()
    #model.load_weights('Final_model_wn_weights.h5')

 	####################################################
 						#TODO (Add code here)

 	#Check if any .jpg files already exist or no. If yes, delete them

 	####################################################

    model = load_model('new_model.h5')
    # Get stream from webcam and set parameters)
    vs = VideoStream().start()

    # max number of hands we want to detect/track
    num_hands_detect = 1

    # Used to calculate fps
    start_time = datetime.datetime.now()
    num_frames = 0

    im_height, im_width = (None, None)

    arr = []
    final_arr = []
    Sent_arr = []
    recommended = "None"	
    counter = 0
    #print("Enter first gesture")
    gesture_no = 1

    bg = None
    aWeight = 0.5
    num_frames = 0
    try:

        while True:
            # Read Frame and process
            frame = vs.read()
            #frame = cv2.resize(frame, (320, 240))

            frame = cv2.flip(frame,1)

            clone = frame.copy()

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
            	cv2.imwrite('frame_copy.jpg',frame)
            	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            except:
                print("Error converting to RGB")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (7, 7), 0)    

            if num_frames < 30:
                run_avg(gray, aWeight)
            else:
            # segment the hand region
                hand = segment(gray)

            # check whether hand region is segmented
                if hand is not None:
                    # if yes, unpack the thresholded image and
                    # segmented region
                    (thresholded, segmented) = hand
                    cv2.imwrite('threshold.jpg',thresholded)

                    # draw the segmented region and display the frame
                    #cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))
                    #cv2.imshow("Thesholded", thresholded)
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        vs.stop()
                        break


            num_frames += 1

            font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX 
            kernel=np.ones((3,3))

        
            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)


            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)

            # When no hand is detected
            if(max(scores) < 0.5):
            	frame_copy = cv2.imread('frame_copy.jpg')
            	cv2.imwrite("img_thr2.jpg",frame_copy)


            #Preparing cropped image for prediction
            tester_img,tester_name = load_test_dataa()
            pred = model.predict_classes(tester_img)
            predictions_labels_plot = get_labels_for_plot(pred)

            if(max(scores) < 0.5):
            	predictions_labels_plot = "nothing"

            if(len(arr) > 15 ):
                for elmts in arr:
                    if predictions_labels_plot == elmts:
                        counter += 1
                        if counter == len(arr):
                            arr = []
                            counter = 0
                            if(predictions_labels_plot == 'nothing'):
                            
                                if(len(final_arr) != 0):
                                    #print("final_arr is :")
                                    #print(''.join(final_arr))
                                    gesture_no = 1
                                    Sent_arr.append(str(''.join(final_arr)))
                                    final_arr = []
                            elif(predictions_labels_plot == 'delete'):
                            
                                if(len(final_arr) != 0):
                                    #print("final_arr is :")
                                    #print(''.join(final_arr))
                                    final_arr.pop()
                                    gesture_no += 1
                            elif(predictions_labels_plot == 'complete'):
                            	
                                ######################################################
                                			#TODO( Add code here )

                                #Append recommended word to Sent_arr
                                #Set final_arr to []

                                ######################################################

                                gesture_no = 1
                            else:
                                final_arr.append(str(predictions_labels_plot))
                                gesture_no += 1
                   				#####################################################
                   							#TODO( Add code here )

                   				#Predict word based on word existing in str(''.join(final_arr)) currently and set it in recommended variable
                   				#Preferably create a fucntion for recommending and call that fxn here with str(''.join(final_arr)) as argument.
                   				# I have initialized the recommended variable to "None". Please make whatever changes necessary.

                   				#####################################################


                                break

            arr.append(predictions_labels_plot)

            cv2.putText(frame, 'Enter gesture ' + str(gesture_no) + ': '+ str(predictions_labels_plot),
                                    (int(im_width*0.01),int(im_height*0.1)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)


            cv2.putText(frame, 'Recommendation : ' +  recommended,
                        (int(im_width*0.01),int(im_height*0.7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)

            cv2.putText(frame, 'Word : ' +  str(''.join(final_arr)),
                        (int(im_width*0.01),int(im_height*0.8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)

            cv2.putText(frame, 'Sentence : ' +  str(''.join(Sent_arr)),
                        (int(im_width*0.01),int(im_height*0.9)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                #detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
