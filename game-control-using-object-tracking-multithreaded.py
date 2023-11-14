'''This script can detect objects specified by the HSV color and also sense the
direction of their movement.Using this script a Snake Game which has been loaed
in the repo, can be played. Implemented using OpenCV.
Uses seperate thread for reading frames through OpenCV.'''

#Import necessary modules
import cv2
"""cv2: This is the OpenCV library, which provides computer vision and image processing functions."""
import imutils
"""imutils: It provides easy-to-use functions for resizing, rotating, and performing other common image processing operations."""
import numpy as np
"""numpy: It provides support for multi-dimensional arrays and various mathematical operations."""
from collections import deque
"""deque from collections:  It is used in this script to store a fixed number of object coordinates for tracking."""
import time
"""time: This module provides various time-related functions, such as pausing the execution of the script."""
import pyautogui
"""pyautogui: This module provides cross-platform control of the mouse and keyboard.
It is used in this script to simulate key presses based on the detected object's direction."""
from threading import Thread
"""Thread from threading module is imported to enable multi-threading functionality for reading frames from the webcam in a separate thread."""





#Class implemeting seperate threading for reading of frames.
class WebcamVideoStream:
    def __init__(self):
        """to capture the video from default webcam"""
        self.stream = cv2.VideoCapture(0)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
    def start(self):
        """for starting video stream in a seperate frame"""
        Thread(target = self.update, args=()).start()
        return self
    def update(self):
        """this the main loop for seperate frame,
           it reads the frame continuously from webcam
           until the stop attribute is false"""
        while True:
            if self.stopped:
                return
            self.ret, self.frame = self.stream.read()
    def read(self):
        """return the current frame"""
        return self.frame
    def stop(self):
        """stop the video"""
        self.stopped = True

#Define HSV colour range for green colour objects
"""specific green shades that the script captures"""
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

#Used in deque structure to store no. of given buffer points
#(to keep track of previous position of the object detected)
buffer = 20 
"""as the new points are added the older one will get
 discarded to maitain the buffer size"""

#Used so that pyautogui doesn't click the center of the screen at every frame
flag = 0

#Points deque structure storing 'buffer' no. of object coordinates
pts = deque(maxlen = buffer)
#Counts the minimum no. of frames to be detected where direction change occurs
counter = 0
#Change in direction is stored in dX, dY
(dX, dY) = (0, 0)
#Variable to store direction string
direction = ''
#Last pressed variable to detect which key was pressed by pyautogui
last_pressed = ''

#Sleep for 2 seconds to let camera initialize properly.
time.sleep(2)

#Use pyautogui function to detect width and height of the screen
width,height = pyautogui.size()

#Start video capture in a seperate thread from main thread.
vs = WebcamVideoStream().start()
#video_shower = VideoShow(vs.read()).start()

#Click on the centre of the screen, game window should be placed here.
pyautogui.click(int(width/2), int(height/2))

while True:
    #Store the readed frame in frame
    frame = vs.read()
    #Flip the frame to avoid mirroring effect
    frame = cv2.flip(frame,1)
    #Resize the given frame to a 600*600 window
    frame = imutils.resize(frame, width = 600)
    #Blur the frame using Gaussian Filter of kernel size 11, to remove excessivve noise
    blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
    #Convert the frame to HSV, as HSV allow better segmentation.
    """The HSV color space is often used for better color segmentation and analysis in computer vision applications"""
    hsv_converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

    #Create a mask for the frame, showing green values
    mask = cv2.inRange(hsv_converted_frame, greenLower, greenUpper)
    #Erode the masked output to delete small white dots present in the masked image
    mask = cv2.erode(mask, None, iterations = 2)
    #Dilate the resultant image to restore our target
    mask = cv2.dilate(mask, None, iterations = 2)

    #Find all contours in the masked image
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #Define center of the ball to be detected as None
    center = None

    #If any object is detected, then only proceed
    if len(contours) > 0:
        #Find the contour with maximum area
        c = max(contours, key = cv2.contourArea)
        #Find the center of the circle, and its radius of the largest detected contour.
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        #Calculate the centroid of the ball, as we need to draw a circle around it.
        M = cv2.moments(c)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        #Proceed only if a ball of considerable size is detected
        if radius > 10:
            #Draw circles around the object as well as its centre
            cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
            cv2.circle(frame, center, 5, (0,255,255), -1)
            #Append the detected object in the frame to pts deque structure
            pts.appendleft(center)

    #Using numpy arange function for better performance. Loop till all detected points
    for i in np.arange(1, len(pts)):
        #If no points are detected, move on.
        if(pts[i-1] == None or pts[i] == None):
            continue

        #If atleast 10 frames have direction change, proceed
        if counter >= 10 and i == 1 and len(pts) >= 10 and pts[-10] is not None:
            #Calculate the distance between the current frame and 10th frame before
            dX = pts[-10][0] - pts[i][0]
            dY = pts[-10][1] - pts[i][1]
            (dirX, dirY) = ('', '')

            #If distance is greater than 50 pixels, considerable direction change has occured.
            if np.abs(dX) > 50:
                dirX = 'West' if np.sign(dX) == 1 else 'East'

            if np.abs(dY) > 50:
                dirY = 'North' if np.sign(dY) == 1 else 'South'

            #Set direction variable to the detected direction
            direction = dirX if dirX != '' else dirY
            #Write the detected direction on the frame.
            cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        #Draw a trailing red line to depict motion of the object.
        thickness = int(np.sqrt(buffer / float(i + 1)) * 2.5)
        cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)

    #If deteced direction is East, press right button
    if direction == 'East':
        if last_pressed != 'right':
            pyautogui.press('right')
            last_pressed = 'right'
            print("Right Pressed")
            #pyautogui.PAUSE = 2
    #If deteced direction is West, press Left button
    elif direction == 'West':
        if last_pressed != 'left':
            pyautogui.press('left')
            last_pressed = 'left'
            print("Left Pressed")
            #pyautogui.PAUSE = 2
    #if detected direction is North, press Up key
    elif direction == 'North':
        if last_pressed != 'up':
            last_pressed = 'up'
            pyautogui.press('up')
            print("Up Pressed")
            #pyautogui.PAUSE = 2
    #If detected direction is South, press down key
    elif direction == 'South':
        if last_pressed != 'down':
            pyautogui.press('down')
            last_pressed = 'down'
            print("Down Pressed")
            #pyautogui.PAUSE = 2


    #video_shower.frame = frame
    #Show the output frame.
    cv2.imshow('Game Control Window', frame)
    key = cv2.waitKey(1) & 0xFF
    #Update counter as the direction change has been detected.
    counter += 1

    #If pyautogui has not clicked on center, click it once to focus on game window.
    if (flag == 0):
        pyautogui.click(int(width/2), int(height/2))
        flag = 1

    #If q is pressed, close the window
    if(key == ord('q')):
        break
#After all the processing, release webcam and destroy all windows
vs.stop()
cv2.destroyAllWindows()
