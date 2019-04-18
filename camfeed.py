import cv2 as cv
import numpy as np
from utils import *
import sys 
from model import get_output_layers

#To detect the gender of person, we assume the face is enough & therefore we pass faces of people who are already under tracking 
def show_gender(img_frame, face_outs, gender_net, conf_threshold = 0.7):
    bboxes = []
    genderList = ['Male','Female'];
    Height, Width = img_frame.shape[0],img_frame.shape[1]
    for i in range(face_outs.shape[2]):
        confidence = face_outs[0, 0, i, 2]
        if confidence > conf_threshold:
            xmin = int(face_outs[0, 0, i, 3] * Width)
            ymin = int(face_outs[0, 0, i, 4] * Height)
            xmax = int(face_outs[0, 0, i, 5] * Width)
            ymax = int(face_outs[0, 0, i, 6] * Height)
            bboxes.append([xmin, ymin, xmax, ymax])
            cv.rectangle(img_frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), int(round(Height/150)), 8);
            pad_val = 10;

            #get face frame
            face = img_frame[max(xmin-pad_val,0) : min(xmax+pad_val, Width-1), max(ymin-pad_val,0) : min(ymax+pad_val, Height-1)];
            face_blob = cv.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
            
            #input face frame to gender model
            gender_net.setInput(face_blob)
            genderPreds = gender_net.forward()
            gender = genderList[genderPreds[0].argmax()]
            label = "{}".format(gender);
            print(label);
            cv.putText(img_frame, label, (xmin, ymin-pad_val), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA);
            
    return img_frame

def isVisible(pt, cam_position):
    #Assume a line x = yc where yc is the y cordinate of camera position
    #A point is visible to camera if it's x cordinate is strictly greater than xcordinate of camera 
    if(pt[0] > cam_position[0]):
        return True;
    return False;

#Room size a 4 tuple cordinates specifying rectangle: [0 0] followed by width and height
def map_to_room(img_frame,cam_position,roomsize,bboxes):
    height = img_frame.shape[0];
    width = img_frame.shape[1];

    room_width = roomsize[2]-roomsize[0];
    room_height = roomsize[3]-roomsize[1];
    print('Original==> width: ',width,' height: ',height);
    print('Room==> width: ',room_width,' height: ',room_height);
    points = [];
    for bbox in bboxes:
        x,y,w,h = bbox[0], bbox[1], bbox[2], bbox[3];
        orig_pt = [round(x + w/2.), round(y + h/2.)];
        scaled_pt = [0 , 0];
        scaled_pt[0] = (1.0*orig_pt[0]/width)*room_width;
        scaled_pt[1] = (1.0*orig_pt[1]/height)*room_height;
        if(isVisible(scaled_pt,cam_position)):
            points.append(scaled_pt);
            print('rescaled_room_pts: (',scaled_pt[0],',',scaled_pt[1],')');
        else:
            print("Sorry not visible pt: (",scaled_pt[0],',',scaled_pt[1],')');
    return points;

#Function for getting video feed via webcam
def getVideoFeed(people_net,gender_net,face_net,roomsize,cam_position,scale = 1.0/255):
    
    cam = cv.VideoCapture(0); #0 enables the webcam device
    print ("Press Esc key to exit..");

    prev_people_count = 0;
    isFirstTime = True;

    while True:
        x,img_frame = cam.read();
        #img_frame = cv.imread('images.jpeg');
        Height, Width = img_frame.shape[0],img_frame.shape[1]
        if not x:
            print('Unable to read from camera feed');
            sys.exit(0);

        ################################
        #       DATA PREPARATION
        ################################

        #documentation:      cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
        #size is spatial size of input expected by the model
        people_blob = cv.dnn.blobFromImage(img_frame, scale, (416,416), (0,0,0), True, crop=False);
        face_blob = cv.dnn.blobFromImage(img_frame, 1.0, (300, 300), [104, 117, 123], True, crop=False);

        people_net.setInput(people_blob);
        face_net.setInput(face_blob);

        ################################
        #       INFERENCE TIME
        ################################

        people_outs = people_net.forward(get_output_layers(people_net));
        face_outs = face_net.forward();


        img_frame = show_gender(img_frame,face_outs,gender_net);

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5 #confidence of bbox
        
        for out in people_outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    if class_id != 0:# 0 is the class id corresponding to person class
                        continue;
                    class_ids.append(class_id)
                    print ('class_ids: ',class_ids);
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        people_detected = boxes;

        #we get rescaled VISIBLE cordinates as list of centroids of bounding boxes tracking people 
        rescaled_room_pts = map_to_room(img_frame,cam_position,roomsize,boxes);

        num_people_detected = len(rescaled_room_pts);
        print('total people_detected: ',num_people_detected);
        if(prev_people_count < num_people_detected):
            print("--------------------------------");
            print("New Person Entered");
            print("--------------------------------");

        prev_people_count = num_people_detected;

        show_points(roomsize,rescaled_room_pts,cam_position);

        pad_val = 10;
        
        for temp_face in people_detected:
            xmin,ymin,width,height = int(temp_face[0]),int(temp_face[1]),int(temp_face[2]),int(temp_face[3]);
            xmax = xmin + width;
            ymax = ymin + height;
            if xmin>0 and ymin>0 and width>0 and height>0:
            #                                                (Blue,Green,Red)
                cv.rectangle(img_frame,(xmin,ymin),(xmax,ymax),(255,0,0),2); 
                print('point of interest: (',round(xmin+width/2),',',round(ymin+height/2),')');
                cv.circle(img_frame, (round(xmin+width/2), round(ymin+height/2)), 5, (0,255,0));
                cv.putText(img_frame, 'Person', (xmin, ymin-pad_val), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2, cv.LINE_AA);
                person_frame = img_frame[xmin-pad_val : xmax+pad_val ,ymin-pad_val : ymax+pad_val];


        cv.imshow('cam_feed',img_frame);

        key = cv.waitKey(1) & 0xFF;
        if key == 27: #27 for Esc Key
            break;

    cv.destroyAllWindows();
    cam.release();