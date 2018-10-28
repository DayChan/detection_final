#!/usr/bin/python
#-*-coding:utf-8-*-

import zmq 
import cv2
import time
import pickle
context = zmq.Context()  
socket = context.socket(zmq.PUB)  
socket.bind("tcp://0.0.0.0:5000")
print("Port 5000 connect well")  
msg = cv2.imread('picture.jpg')
while True:  
    msg_pickle = pickle.dumps(msg)
    socket.send(msg_pickle)
    print("Send successfully")
    time.sleep(0.2)
