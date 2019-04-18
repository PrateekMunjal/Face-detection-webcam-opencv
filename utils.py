import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from time import sleep
import numpy as np 

face_detector = cv.CascadeClassifier('haarcascade_frontalface_default.xml') 

def show_points(room_size,tracked_points,cam_position,pad=2):
 	
    room_width = room_size[2]-room_size[0];
    room_height = room_size[3]-room_size[1];
    rectangle = plt.Rectangle((0,0), room_width, room_height);#, fc='r')
    plt.plot((cam_position[0],cam_position[0]), (0,room_height),color='yellow',linewidth=5);
    plt.gca().add_patch(rectangle)
    plt.xlim((room_size[0]-pad,room_size[2]+pad));
    plt.ylim((room_size[1]-pad,room_size[3]+pad));
    plt.plot(cam_position[0],cam_position[1],marker='D',color='red',markersize=25,label=' Web Camera');
    
    temp_id = 0;
    colormap = plt.cm.gist_ncar;
    colorst = [colormap(i) for i in np.linspace(0, 0.9,len(tracked_points))];  #to get each person a different color
    for i in tracked_points:
        temp_id += 1;
        plt.plot(i[0],i[1],marker='o',color=colorst[temp_id-1]);
        plt.pause(0.5);

    plt.legend(loc='best');
    plt.pause(0.5);
    plt.clf();#to reuse the same plot and alleviates the use of explicit closing of plots.
    