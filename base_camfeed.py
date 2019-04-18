import cv2 as cv
import numpy as np
from utils import *
import sys 

'''
#For cv rectange documentation
cv.Rectangle(img, pt1, pt2, color, thickness=1, lineType=8, shift=0) → None¶
Parameters: 
img – Image.
pt1 – Vertex of the rectangle.
pt2 – Vertex of the rectangle opposite to pt1 .
rec – Alternative specification of the drawn rectangle.
color – Rectangle color or brightness (grayscale image).
thickness – Thickness of lines that make up the rectangle. Negative values, like CV_FILLED , mean that the function has to draw a filled rectangle.
lineType – Type of the line. See the line() description.
shift – Number of fractional bits in the point coordinates.
'''

#Function for getting video feed via webcam
def getVideoFeed():
    cam = cv.VideoCapture(0); #0 enables the webcam device
    print ("Press Esc key to exit..");
    while True:
        x,img_frame = cam.read();
        if not x:
            print('Unable to read from camera feed');
            sys.exit(0);

        gray_scale_frame = cv.cvtColor(img_frame,cv.COLOR_BGR2GRAY);
        
        faces_detected = face_detector.detectMultiScale(gray_scale_frame,1.3,5);

        for (xmin,ymin,width,height) in faces_detected:
            xmax = xmin + width;
            ymax = ymin + height;
            cv.rectangle(img_frame,(xmin,ymin),(xmax,ymax),(255,255,0),2)  

        cv.imshow('cam_feed',img_frame);

        key = cv.waitKey(1) & 0xFF;
        if key == 27: #27 for Esc Key
            break;

    cv.destroyAllWindows();
    cam.release();