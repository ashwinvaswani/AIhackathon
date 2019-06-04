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
from keras.applications import ResNet50,VGG16
from keras.applications.resnet50 import preprocess_input
from keras import Model,layers
from keras.models import load_model,model_from_json

from utils import detector_utils as detector_utils

labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                               'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                               'Z':25,'space':26,'del':27,'nothing':28}


def get_labels_for_plot(predictions):
    predictions_labels = []
    for ins in labels_dict:
        if predictions == labels_dict[ins]:
            predictions_labels.append(ins)
            #return predictions_labels
            break
    return predictions_labels

def load_test_dataa():
    images = []
    names = []
    size = 64,64
    temp = cv2.imread('./img2.jpg')
    #temp = cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
    #cv2.imwrite('out.jpg',temp)
    temp = cv2.resize(temp, size)
    images.append(temp)
    names.append('img2')
    images = np.array(images)
    images = images.astype('float32')/255.0
    return images, names

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--display', dest='display', type=int,
                        default=1, help='Display the detected images using OpenCV. This reduces FPS')
args = vars(ap.parse_args())

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':
    # Detection confidence threshold to draw bounding box
    score_thresh = 0.60

    model=load_model('Final_model_asl.h5')

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
    counter = 0
    print("Enter first gesture")

    try:
        while True:
            # Read Frame and process
            frame = vs.read()
            frame = cv2.resize(frame, (320, 240))

            if im_height == None:
                im_height, im_width = frame.shape[:2]

            # Convert image to rgb since opencv loads images in bgr, if not accuracy will decrease
            try:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            except:
                print("Error converting to RGB")

            #print(frame)

            cv2.imwrite('1.jpg',frame)
            font=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX 
            kernel=np.ones((3,3))


            #cv2.putText(frame,' '+str(predictions_labels_plot),(0,50),font,0.5,(255,255,255),1)
            # Run image through tensorflow graph
            boxes, scores, classes = detector_utils.detect_objects(
                frame, detection_graph, sess)

            #print(boxes)

            # Draw bounding boxeses and text
            detector_utils.draw_box_on_image(
                num_hands_detect, score_thresh, scores, boxes, classes, im_width, im_height, frame)
            #print(scores)
            if(max(scores) <0.5):
                #img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                #img = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
                cv2.imwrite("img2.jpg",frame)

###############################################TODO#####################################################
            #TODO:  NOT SURE ABOUT THIS (Try filters)


            tester_img,tester_name = load_test_dataa()
            pred = model.predict_classes(tester_img)
            predictions_labels_plot = get_labels_for_plot(pred)
            #print(predictions_labels_plot[0]) 

            # new_frame = cv2.imread("img2.jpg")
            # roi=cv2.resize(new_frame,(64,64))
            # roi=cv2.cvtColor(roi,cv2.COLOR_RGB2BGR)
            # x=np.reshape(roi,(-1,64,64,3))
            # pred = (model.predict_classes(x))
            # predictions_labels_plot = get_labels_for_plot(pred)

            if(len(arr) > 10 ):
                for elmts in arr:
                    if predictions_labels_plot == elmts:
                        counter += 1
                        if counter == len(arr):
                            arr = []
                            counter = 0
                            if(predictions_labels_plot[0] == 'nothing'):
                                print("final_arr is :")
                                if(len(final_arr) != 0):
                                    print(''.join(final_arr))
                                    final_arr = []
                            else:
                                final_arr.append(str(predictions_labels_plot[0]))
                                #print(final_arr)
                                break
            #print(final_arr)
            #print()
            #print("Enter next gesture")

            




            arr.append(predictions_labels_plot)

            #print(predictions_labels_plot)

            cv2.putText(frame, str(predictions_labels_plot),
                        (int(im_width*0.65),int(im_height*0.1)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0), 2)
#########################################################################################################

            # Calculate Frames per second (FPS)
            num_frames += 1
            elapsed_time = (datetime.datetime.now() -
                            start_time).total_seconds()
            fps = num_frames / elapsed_time

            if args['display']:
                # Display FPS on frame
                detector_utils.draw_text_on_image("FPS : " + str("{0:.2f}".format(fps)), frame)
                cv2.imshow('Detection', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    vs.stop()
                    break

        print("Average FPS: ", str("{0:.2f}".format(fps)))

    except KeyboardInterrupt:
        print("Average FPS: ", str("{0:.2f}".format(fps)))
