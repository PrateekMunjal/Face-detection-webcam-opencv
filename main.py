from camfeed import getVideoFeed as getVideo
from model import load_model
import argparse

#load model to track people
people_net = load_model(model_weights = "yolov3.weights", model_config = "yolov3.cfg");

#load model to classify the gender
gender_net = load_model(model_weights = "gender_net.caffemodel",model_config = "gender_deploy.prototxt");

#load model to get the bounding box for face
face_net = load_model(model_weights = "opencv_face_detector_uint8.pb",model_config="opencv_face_detector.pbtxt");

parser = argparse.ArgumentParser();
parser.add_argument("-xc","--x_cam", help="x-cordinate of webcam",type=int,required=False,default=0);
parser.add_argument("-yc","--y_cam", help="y-cordinate of webcam",type=int,required=False,default=5);
parser.add_argument("-rw","--room_width", help="width of room",type=int,required=False,default=10);
parser.add_argument("-rh","--room_height", help="height of room",type=int,required=False,default=10);
args = parser.parse_args();


#input the webcam position (x-cordinate, y-cordinate)
cam_position = [args.x_cam,args.y_cam];

#input the room dimension as (width, height)
room_size  = [0,0,args.room_width,args.room_height];

#Start tracking people through webcam
getVideo(people_net,gender_net,face_net,room_size,cam_position);