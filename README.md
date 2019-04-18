# Problem Statement
Given front-view video feed of a room (webcam feed from a laptop sitting on a table), estimate & display in real-time, the top-view of people’s location in the room, along with their gender.

## Compatibility
* Python 3.6
* Ubuntu 18

## Dependencies
* cv2 -- *for opencv*
* matplotlib -- *for plotting purposes*
* To detect face we use: *opencv_face_detector_uint8.pb, opencv_face_detector.pbtxt*
* To track people we use: *yolov3.weights, yolov3.cfg*
* To detect gender of person we use: *gender_net.caffemodel, gender_deploy.prototxt*

## Assumptions
* When a person is not visible from webcam, we do not count it as present
in the room.
* For webcam visibility: Assume a line x = yc where yc is the y cordinate
of camera position. Now, a point is said to be visible with respect
to camera iff it’s x cordinate is strictly greater than x-cordinate of
camera.
* For better visualization, we mark a yellow line for defining web-cam
visibility boundary.

## Approach
* Input the room dimensions and web camera position.
* The web-camera frames are fed to detect the person.
* Now among the detected persons, we calculate the centroid of bounding
boxes of visible persons. The visibility is with respect to position of
web-camera.
* Next, we pass the person frames to detect faces which are further input
to detect the gender of person.
* In last we plot them on web-camera as well as on room plot in real
time.

## Relevant code files
main.py -- *the main file instantiating all the calls*

model.py -- *responsible for loading the model with the help of weights
and configuration file*

utils.py -- *responsible for showing the re-scaled tracked points on given
room*

camfeed.py -- *Heart of the program. It contains four functions as men-
tioned below:*
  * show gender: *responsible for detecting the gender of person based on the face of person as input*
  * isVisible: *Boolean function. outputs true if point is visible from web-cam and false otherwise*
  * map_to_room: *returns a list of visible points re-scaled to room coordinates. We re-scale coordinates by normalizing coordinate values by their corresponding axis length*
  * getVideoFeed: *responsible for calling above three functions with
web cam frames as input*

## Usage
For default parameters
```
python3 main.py
```
Generalized command line
```
python3 main.py –xc [x-cordinate of webcam] –yc [y-cordinate of webcam] -rw [width of room] -rh [height of room]
For example: python3 main.py -xc 5 -yc 2 -rw 20 -rh 10
```
## Sample Output
![](https://github.com/PrateekMunjal/Face-detection-webcam-opencv/blob/master/output.png)

## Dowmload pre-trained model weights
* Run wget https://pjreddie.com/media/files/yolov3.weights
* Run wget https://github.com/arunponnusamy/object-detection-opencv/blob/master/yolov3.cfg
* Run wget https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector_uint8.pb
* Run wget https://github.com/spmallick/learnopencv/blob/master/AgeGender/opencv_face_detector.pbtxt
* Download gender_net.caffemodel from https://www.dropbox.com/s/iyv483wz7ztr9gh/gender_net.caffemodel?dl=0
* Run wget https://github.com/spmallick/learnopencv/blob/master/AgeGender/gender_deploy.prototxt 

## References
1. https://github.com/spmallick/learnopencv/blob/master
2. https://github.com/arunponnusamy/object-detection-opencv
